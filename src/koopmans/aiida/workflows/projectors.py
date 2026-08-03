"""Reading of the external projector files pw2wannier90 stages.

One ``<element>.dat`` radial-projector file per element, parsed the way
pw2wannier90 reads it (Fortran list-directed integers), yields the
per-element projector tables upstream's builder expects.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiida import orm

    from koopmans.input_file import KoopmansInput


def _int_or_none(word: str) -> int | None:
    """Parse an integer token, returning None when it is not one."""
    try:
        return int(word)
    except ValueError:
        return None


def _expand_repeat_token(token: str) -> list[int] | None:
    """Parse one list-directed integer token, expanding an ``r*v`` repeat count."""
    prefix, star, value = token.partition("*")
    if not star:
        parsed = _int_or_none(token)
        return None if parsed is None else [parsed]
    repeat = _int_or_none(prefix)
    parsed = _int_or_none(value)
    if repeat is None or parsed is None or repeat < 1:
        return None
    return [parsed] * repeat


def _read_list_directed_ints(
    records: list[str],
    start: int,
    count: int,
    what: str,
    malformed: Callable[[str], ValueError],
) -> tuple[list[int], int]:
    """Read ``count`` integers as one Fortran list-directed READ statement.

    Consume whole records from ``records[start:]``: blank records are
    skipped, values are separated by blanks and/or commas and may continue
    over records, and ``r*v`` repeat counts expand to ``r`` copies of
    ``v``. Return the values plus the index of the record after the last
    one consumed — the remainder of that record is discarded, because the
    next READ starts on a fresh record. A ``/`` terminator before the
    count is met, or a null value from adjacent commas within a record,
    would leave Fortran values undefined, so both are rejected rather
    than reproduced (nulls formed at the start of a record or across a
    record boundary are not detected and read as regular separators).
    """
    values: list[int] = []
    for index in range(start, len(records)):
        record = records[index]
        if re.search(r",[ \t]*,", record):
            raise malformed(f"adjacent commas in the {what} leave a value undefined.")
        for token in record.replace(",", " ").split():
            before_slash, slash, _ = token.partition("/")
            if before_slash:
                expanded = _expand_repeat_token(before_slash)
                if expanded is None:
                    raise malformed(f"the {what} value {before_slash!r} is not an integer.")
                values.extend(expanded)
            if len(values) >= count:
                return values[:count], index + 1
            if slash:
                raise malformed(
                    f"a `/` ends the {what} after {len(values)} of {count} values, "
                    "leaving the rest undefined."
                )
    raise malformed(f"it ends before the {what} is complete ({len(values)} of {count} values).")


def _read_projector_angular_momenta(projector_file: Path) -> list[int]:
    """Read the per-projector angular momenta of an external projector file.

    Follow pw2wannier90's reader: leading comment lines (first non-space
    character ``#``; a tab-indented ``#`` is not a comment there) are
    skipped, then one list-directed read takes the ``<ngrid> <nproj>``
    header and a second, starting on a fresh record, takes the ``nproj``
    angular momenta (:func:`_read_list_directed_ints`). The radial tables
    that follow are pw2wannier90's business. Deliberately stricter than
    the Fortran reader in three cases it lets through with undefined or
    unusable values: a zero projector count, a ``/`` terminating the read
    early, and negative angular momenta all raise.
    """

    def malformed(reason: str) -> ValueError:
        """Build the rejection error for this projector file."""
        return ValueError(f"The projector file {projector_file} is malformed: {reason}")

    records = list(
        itertools.dropwhile(
            lambda record: record.lstrip(" ").startswith("#"),
            projector_file.read_text().splitlines(),
        )
    )
    (_, nproj), momenta_start = _read_list_directed_ints(
        records, 0, 2, "`<ngrid> <nproj>` header", malformed
    )
    if nproj < 1:
        raise malformed(f"the projector count is {nproj}; at least one projector is required.")
    momenta, _ = _read_list_directed_ints(
        records, momenta_start, nproj, "angular-momentum list", malformed
    )
    negative = [angular_momentum for angular_momentum in momenta if angular_momentum < 0]
    if negative:
        raise malformed(f"it lists negative angular momenta: {negative}.")
    return momenta


def _load_external_projectors(
    structure: orm.StructureData,
    proj_dir: Path | None,
) -> tuple[dict[str, Any], str]:
    """Build the per-element projector tables from an external projector directory.

    The directory holds one ``<element>.dat`` radial-projector file per
    element — the filename pw2wannier90 stages and reads — and those files
    are the whole user-facing contract: each contributes 2l+1 projectors
    per listed angular momentum (:func:`_read_projector_angular_momenta`).

    The returned dict exists only to satisfy upstream's
    ``get_builder_from_protocol``, which demands ``external_projectors``
    tables: each entry carries the parsed ``l`` and ``frozen: False``, so
    every projector is Lowdin-orthonormalized. Partial freezing of the
    projector set is deliberately unsupported.

    The directory is ultimately consumed on the pw2wannier90 code's
    computer (it is staged into each calculation from there), but the
    files are parsed and validated here on the local filesystem — a check
    that coincides with the real one only when that computer shares this
    filesystem, i.e. the localhost setup. Projector directories that exist
    only on a remote computer are not supported yet.

    Returns the tables and the resolved directory path.
    """
    if proj_dir is None:
        raise ValueError(
            "`pw2wannier90.atom_proj_dir` must be set when `pw2wannier90.atom_proj_ext` "
            "is true: it locates the external projector files."
        )
    directory = Path(proj_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(
            f"`pw2wannier90.atom_proj_dir` does not exist on this machine: {directory}. "
            "The directory is read on the computer the pw2wannier90 code runs on, and "
            "this check assumes that computer shares the local filesystem (the "
            "localhost setup); projector directories that exist only on a remote "
            "computer are not supported yet."
        )
    elements = sorted(
        {structure.get_kind(site.kind_name).symbol for site in structure.sites}  # type: ignore[no-untyped-call]
    )
    missing_files = [
        f"{element}.dat" for element in elements if not (directory / f"{element}.dat").is_file()
    ]
    if missing_files:
        raise ValueError(
            f"{directory} is missing the projector files {missing_files}; "
            "pw2wannier90 reads one `<element>.dat` per element of the structure."
        )
    external_projectors = {
        element: [
            {"l": angular_momentum, "frozen": False}
            for angular_momentum in _read_projector_angular_momenta(directory / f"{element}.dat")
        ]
        for element in elements
    }
    return external_projectors, str(directory)


def _reject_unwired_external_projectors(koopmans_input: KoopmansInput, route: str) -> None:
    """Reject ``atom_proj_ext`` on a route that does not consume it.

    The singlepoint and trajectory routes build their Wannierizations
    without consulting the external projector keywords, so accepting the
    switch there would silently drop it.
    """
    if koopmans_input.calculator_parameters.pw2wannier90.atom_proj_ext:
        raise NotImplementedError(
            f"`pw2wannier90.atom_proj_ext` is not wired into the {route} route; "
            "external projectors are currently supported by the `wannierize` task only."
        )
