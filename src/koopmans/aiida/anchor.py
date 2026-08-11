"""Anchor files: the paper trail from a submitted input file to its process.

``koopmans submit`` hands a calculation to the daemon and returns
immediately; the only way to find that calculation again is to remember
its AiiDA node identity somewhere. An anchor file (``<seed>.run.yaml``,
next to the input file) is that memory: a YAML list of every submission
made from that input, newest last, so a resubmission never overwrites the
record of the run before it.

:func:`resolve_target` turns what a user types at ``koopmans status`` /
``koopmans attach`` — nothing, an anchor file, an input file, or an
explicit ``--uuid``/``--pk`` — into the identity of one process.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, TypedDict

import yaml

__all__ = [
    "AnchorEntry",
    "ResolvedTarget",
    "anchor_path_for_input",
    "append_anchor_entry",
    "newest_anchor_entry",
    "read_anchor_entries",
    "resolve_target",
]


class AnchorEntry(TypedDict):
    """One submission recorded in an anchor file."""

    uuid: str
    pk: int
    input: str
    profile: str
    submitted: str


class ResolvedTarget(NamedTuple):
    """A process identity to load: ``uuid`` takes priority over ``pk``.

    ``pk`` alone (``uuid`` unset) only happens for a bare ``--pk``, which
    names an AiiDA node directly with no anchor file involved.
    """

    uuid: str | None
    pk: int | None


def anchor_path_for_input(input_path: Path) -> Path:
    """Return the anchor file an input file's submissions are recorded under."""
    return input_path.with_name(f"{input_path.stem}.run.yaml")


def read_anchor_entries(anchor_path: Path) -> list[AnchorEntry]:
    """Return the submissions recorded in ``anchor_path``, oldest first.

    An anchor file that does not exist yet has no submissions.
    """
    if not anchor_path.exists():
        return []
    data = yaml.safe_load(anchor_path.read_text())
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"{anchor_path} does not hold a list of run entries.")
    return list(data)


def append_anchor_entry(anchor_path: Path, entry: AnchorEntry) -> None:
    """Record ``entry`` as the newest submission in ``anchor_path``.

    Earlier entries are kept, so resubmitting from the same input file
    never loses the record of a previous run.
    """
    entries = read_anchor_entries(anchor_path)
    entries.append(entry)
    anchor_path.write_text(yaml.safe_dump(entries, sort_keys=False))


def newest_anchor_entry(anchor_path: Path) -> AnchorEntry:
    """Return the most recent submission recorded in ``anchor_path``.

    Raises if the file holds no entries.
    """
    entries = read_anchor_entries(anchor_path)
    if not entries:
        raise ValueError(f"{anchor_path} has no recorded submissions.")
    return entries[-1]


def resolve_target(
    target: str | None,
    *,
    uuid: str | None,
    pk: int | None,
    cwd: Path | None = None,
) -> ResolvedTarget:
    """Resolve a ``koopmans status``/``koopmans attach`` target to a process identity.

    ``--uuid``/``--pk`` name a process directly and skip anchor files
    entirely. Otherwise ``target`` is read as an anchor file
    (``foo.run.yaml``) or an input file whose sibling anchor file
    (``foo.yaml`` -> ``foo.run.yaml``) is read instead; with no target,
    the single ``*.run.yaml`` in ``cwd`` is used. Every case but a direct
    ``--uuid``/``--pk`` resolves to the anchor's newest entry.

    Raises ``ValueError``, naming the ambiguity or the missing file, for
    anything the caller should turn into a user-facing error.
    """
    if uuid is not None and pk is not None:
        raise ValueError("Pass only one of --uuid or --pk, not both.")
    if uuid is not None:
        return ResolvedTarget(uuid=uuid, pk=None)
    if pk is not None:
        return ResolvedTarget(uuid=None, pk=pk)

    cwd = Path.cwd() if cwd is None else cwd

    if target is None:
        anchors = sorted(cwd.glob("*.run.yaml"))
        if not anchors:
            raise ValueError(
                "No *.run.yaml file found in the current directory. Run "
                "`koopmans submit <input file>` first, or pass --uuid/--pk."
            )
        if len(anchors) > 1:
            names = ", ".join(anchor.name for anchor in anchors)
            raise ValueError(
                f"Several run files found ({names}); pass one explicitly, e.g. "
                f"`koopmans status {anchors[0].name}`."
            )
        anchor_path = anchors[0]
    else:
        target_path = Path(target)
        if not target_path.is_absolute():
            target_path = cwd / target_path
        anchor_path = (
            target_path
            if target_path.name.endswith(".run.yaml")
            else anchor_path_for_input(target_path)
        )
        if not anchor_path.exists():
            raise ValueError(f"No run file found at {anchor_path}.")

    entry = newest_anchor_entry(anchor_path)
    return ResolvedTarget(uuid=str(entry["uuid"]), pk=int(entry["pk"]))
