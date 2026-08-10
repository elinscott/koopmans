"""Read a pseudopotential's own header to tell norm-conserving from the rest.

Koopmans functionals are defined for norm-conserving pseudopotentials, and
``kcp.x`` and ``kcw.x`` accept no other kind. The label a family carries says
nothing about this once the family is one the user installed themselves, so
the answer comes from the UPF headers, which ``upf_tools`` reads for both the
v1 and v2 layouts.
"""

from __future__ import annotations

import warnings
from typing import Any, Protocol


class _ReadableFile(Protocol):
    """What this module needs of a pseudopotential node: its bytes as text."""

    def get_content(self) -> str:
        """Return the file's text."""
        ...


# UPF ``pseudo_type`` values that are not norm-conserving. "NC" and "SL"
# (semilocal) are; "1/r" is a bare Coulomb potential, which pw.x takes but the
# Koopmans codes do not. "US" is what a v1 header carries, where there are no
# flags to fall back on; "USPP" is PSlibrary's v2 spelling, and "1/r" is
# carried against a header that names a type without flagging itself.
_NOT_NORM_CONSERVING = {"US", "USPP", "PAW", "1/r"}


def non_norm_conserving_kinds(pseudos: dict[str, _ReadableFile]) -> dict[str, str]:
    """Return the kinds whose pseudopotential is demonstrably not norm-conserving.

    Maps kind name to the offending ``pseudo_type``. A header that states
    nothing about its type is left out: the check refuses on positive
    evidence, so an unreadable or minimal header never blocks a run.
    """
    offenders: dict[str, str] = {}
    for kind, pseudo in sorted(pseudos.items()):
        pseudo_type = _pseudo_type(pseudo)
        if pseudo_type is not None:
            offenders[kind] = pseudo_type
    return offenders


def _pseudo_type(pseudo: _ReadableFile) -> str | None:
    """Return the pseudopotential's type if its header says it is not norm-conserving.

    ``None`` when the header calls it norm-conserving, says nothing, or
    cannot be read at all. A PAW file sets ``is_ultrasoft`` as well as
    ``is_paw``, so PAW is decided first.
    """
    header = _read_header(pseudo)
    if header is None:
        return None

    pseudo_type = header.get("pseudo_type")
    if header.get("is_paw"):
        return pseudo_type or "PAW"
    if header.get("is_ultrasoft"):
        return pseudo_type or "US"
    if isinstance(pseudo_type, str) and pseudo_type.strip().upper() in _NOT_NORM_CONSERVING:
        return pseudo_type
    return None


def _read_header(pseudo: _ReadableFile) -> dict[str, Any] | None:
    """Return a pseudopotential's ``PP_HEADER`` as a dict, or ``None`` if it has none.

    Reading the file, parsing it and finding a header must all succeed;
    whichever of them fails means "cannot tell", which is not grounds to
    refuse. ``upf_tools`` warns when a file names no version, which every v1
    file does, so its warnings are dropped rather than shown.
    """
    from upf_tools import UPFDict

    try:
        content = pseudo.get_content()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            header = UPFDict.from_str(content).get("header")
    except Exception:
        return None
    return header if isinstance(header, dict) else None
