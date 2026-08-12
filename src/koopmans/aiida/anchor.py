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

import os
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import yaml
from pydantic import Field, RootModel, ValidationError

from koopmans.base import BaseModel

__all__ = [
    "AnchorEntry",
    "ResolvedTarget",
    "anchor_path_for_input",
    "append_anchor_entry",
    "newest_anchor_entry",
    "read_anchor_entries",
    "resolve_target",
]


class AnchorEntry(BaseModel):
    """One submission recorded in an anchor file."""

    uuid: str = Field(description="the submitted process's AiiDA node UUID")
    pk: int = Field(description="the submitted process's AiiDA node pk")
    input: str = Field(description="the input file's name, relative to the anchor file")
    profile: str = Field(description="the AiiDA profile the process was submitted under")
    submitted: str = Field(description="an ISO-8601 timestamp of the submission")


class _AnchorFile(RootModel[list[AnchorEntry]]):
    """The list of submissions an anchor file holds, oldest first."""


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


def _lock_path_for(anchor_path: Path) -> Path:
    return anchor_path.with_name(anchor_path.name + ".lock")


def _acquire_lock(lock_path: Path, timeout: float = 10.0) -> None:
    """Block until ``lock_path`` can be created exclusively, or time out.

    Anchor entries are appended by whole-file read-modify-write; two
    writers racing on the same anchor file — concurrent `koopmans submit`
    processes, or threads within one process — would otherwise each read
    the same "before" list and each write it back with only their own
    entry added, silently losing the other's. ``os.O_EXCL`` makes lock
    creation atomic across processes, unlike an in-process
    ``threading.Lock``.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.close(os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Could not acquire the anchor lock {lock_path} within {timeout}s. "
                    "Delete it by hand if an earlier `koopmans submit` crashed while "
                    "holding it."
                ) from None
            time.sleep(0.01)


def _write_entries(anchor_path: Path, entries: list[AnchorEntry]) -> None:
    """Atomically replace ``anchor_path``'s contents with ``entries``.

    Writes to a temp file in the same directory and ``os.replace``s it
    over the target, so a reader never observes a half-written file — a
    crash mid-write leaves either the old file or the new one, never a
    truncated hybrid.
    """
    payload = yaml.safe_dump([entry.model_dump() for entry in entries], sort_keys=False)
    fd, tmp_name = tempfile.mkstemp(
        dir=anchor_path.parent, prefix=f".{anchor_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        os.replace(tmp_name, anchor_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_anchor_entries(anchor_path: Path) -> list[AnchorEntry]:
    """Return the submissions recorded in ``anchor_path``, oldest first.

    An anchor file that does not exist yet has no submissions. Anything
    else wrong with the file — a directory in its place, invalid YAML, a
    shape that is not a list, or an entry missing a field or holding the
    wrong type — raises ``ValueError`` naming the file, so every
    malformed anchor becomes one error the caller can show the user
    instead of a bare parser or OS traceback.
    """
    if not anchor_path.exists():
        return []
    try:
        text = anchor_path.read_text()
    except OSError as exc:
        raise ValueError(
            f"Could not read {anchor_path} ({exc}). Delete or repair the file."
        ) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"{anchor_path} is not valid YAML ({exc}). Delete or repair the file."
        ) from exc
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"{anchor_path} does not hold a list of run entries.")
    try:
        return _AnchorFile.model_validate(data).root
    except ValidationError as exc:
        raise ValueError(
            f"{anchor_path} holds a malformed run entry ({exc}). Delete or repair the file."
        ) from exc


def append_anchor_entry(anchor_path: Path, entry: AnchorEntry) -> None:
    """Record ``entry`` as the newest submission in ``anchor_path``.

    Earlier entries are kept, so resubmitting from the same input file
    never loses the record of a previous run. The read-modify-write is
    serialized by a lock file next to the anchor (see :func:`_acquire_lock`)
    so concurrent appends never race; each write itself is also atomic
    (see :func:`_write_entries`), so a crash mid-write cannot corrupt the
    file even if the lock is somehow bypassed.
    """
    validated = entry if isinstance(entry, AnchorEntry) else AnchorEntry.model_validate(entry)
    lock_path = _lock_path_for(anchor_path)
    _acquire_lock(lock_path)
    try:
        entries = read_anchor_entries(anchor_path)
        entries.append(validated)
        _write_entries(anchor_path, entries)
    finally:
        lock_path.unlink(missing_ok=True)


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

    Raises ``ValueError``, naming the ambiguity or the missing/malformed
    file, for anything the caller should turn into a user-facing error.
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
    return ResolvedTarget(uuid=entry.uuid, pk=entry.pk)
