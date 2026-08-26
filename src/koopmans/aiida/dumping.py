"""Utilities for dumping AiiDA calculations to local file structures."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from koopmans.aiida.utils import suppress_aiida_logging

if TYPE_CHECKING:
    from aiida import orm
    from aiida.common import LinkType

__all__ = [
    "INPUTS_JSON_FILE",
    "MODEL_FILENAME",
    "NODE_METADATA_FILE",
    "OUTPUTS_JSON_FILE",
    "dump_workgraph",
    "trained_model_output",
]


# AiiDA's dump names each child folder "<NN>-<link_label>", appends the
# process label when it differs from the link label, and ends with the pk.
# A sub-workgraph's process label is always "WorkGraph<graph_name>", which
# never equals the link label, so every sub-workgraph folder repeats its
# name inside the suffix. Both decorations go; "<NN>-<link_label>" stays.
_FOLDER_NAME_SUFFIXES = [
    re.compile(r"^(.+)-\d+$"),
    re.compile(r"^(.+)-WorkGraph<[^<>]+>$"),
]

# "<NN>-<link_label>", the shape every dumped step folder has once the pk
# and the process label are gone.
_STEP_FOLDER_NAME = re.compile(r"^(\d+)-(.+)$")

# The process label AiiDA appends to a step folder, e.g. the
# "-KcpCalculation" of "01-dft_init-KcpCalculation" and the
# "-PwBaseWorkChain" of "01-scf-PwBaseWorkChain". A link label is a python
# identifier and so holds no dash: the second dash is what marks the
# suffix as appended, which is why a folder whose link label is itself
# capitalised ("05-RunFinalKI") keeps its label.
_PROCESS_LABEL_SUFFIX = re.compile(r"^(\d+-.+)-[A-Z][A-Za-z0-9]*$")


# The dumped source of a python task: its code, not data the run made.
_TASK_SOURCE_FILE = "source_file"

# aiida-core's record of which node a folder came from: pk, uuid, node
# type and timestamps. It names the folder rather than adding to it.
NODE_METADATA_FILE = "aiida_node_metadata.yaml"

# A trained screening model, as `ml: {model_file: ...}` reads it back.
MODEL_FILENAME = "model.json"

# A step's Data inputs/outputs that have no repository and so never reach
# aiida-core's dump: Dict, List, Int, Float, Str, Bool.
INPUTS_JSON_FILE = "inputs.json"
OUTPUTS_JSON_FILE = "outputs.json"

# Marks a linked Data node as holding no JSON-native value.
_NOT_JSON_REPRESENTABLE = object()

# What a step folder can hold and still count as having produced nothing.
# INPUTS_JSON_FILE joins this set: a pyfunction's scalar arguments are
# Data inputs like any other, so without this every argument-taking
# bookkeeping task would gain a folder of its own just to echo them back.
# OUTPUTS_JSON_FILE stays out of it — a folder holding nothing else is
# exactly the case this dump exists to fix (issue #205).
_NON_CONTENT_FILES = frozenset({_TASK_SOURCE_FILE, NODE_METADATA_FILE, INPUTS_JSON_FILE})

# The dump's own bookkeeping, which says nothing about the run.
_DUMP_BOOKKEEPING_FILES = ("README.md", "aiida_dump_log.json", ".aiida_dump_safeguard")

_DIGEST_BLOCK = 1 << 20

_SYMLINK_README = """\
Some files here appear in more than one step: an input staged from an
earlier step's output, a pseudopotential every calculation reads. Only
one copy of each is a real file; the rest are relative symlinks to it.

Most ways of copying this directory keep those links — `cp -r`, `tar`
and `rsync -a` all do. The one to avoid is `rsync -r` without `-a` or
`-l`: it skips every link, mentions it only in passing, exits 0 all the
same, and leaves you a copy with those files missing. Use `rsync -a`.

For a copy with no links in it at all, dereference them: `cp -aL`,
`tar -h`, or `rsync -aL`.
"""

# Sibling ordering: siblings group into families — the label with its
# digit runs masked — and a family whose members already sit in
# consecutive positions is sorted naturally, while everything else keeps
# the order it ran in. The live progress table draws its rows by the same
# rule, against creation time rather than the number written on a folder,
# and carries its own copy of it; keep the two in step.
_DIGIT_RUN_RE = re.compile(r"(\d+)")


def _natural_key(label: str) -> tuple[tuple[str, int], ...]:
    """Return a sort key that compares digit runs in a label numerically.

    ``re.split`` on a capturing digit group always alternates text,
    digits, text, …, so pairing them up yields keys whose components line
    up across labels: ``"orb_2"`` sorts before ``"orb_10"``.
    """
    parts = _DIGIT_RUN_RE.split(label)
    return tuple(
        (parts[i], int(parts[i + 1]) if i + 1 < len(parts) else -1) for i in range(0, len(parts), 2)
    )


def _family_key(label: str) -> str:
    """Return the label with every digit run masked ("orb_10" → "orb_#")."""
    return _DIGIT_RUN_RE.sub("#", label)


def _is_calculation_folder(path: Path) -> bool:
    """Return whether the folder holds one dumped process, run and recorded."""
    return (path / "inputs").is_dir() and (path / "outputs").is_dir()


def _step_folders(path: Path) -> list[Path]:
    """Return the folder's "<NN>-<label>" subfolders, in number order."""
    numbered = []
    for child in path.iterdir():
        match = _STEP_FOLDER_NAME.match(child.name)
        if child.is_dir() and match is not None:
            numbered.append((int(match.group(1)), len(match.group(1)), child))
    numbered.sort()
    return [child for _, _, child in numbered]


def _file_digest(path: Path) -> str:
    """Return the SHA-256 of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_DIGEST_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_keys(root_path: Path) -> dict[Path, str]:
    """Return a key per real file that two share exactly when identical.

    A file whose size occurs once in the tree can have no twin, so it
    takes a key of its own and is never read — which keeps the pass off
    the large outputs a finished run retrieves.

    Symlinks are left out, so a tree that already holds some is not
    counted through them.
    """
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in root_path.rglob("*"):
        if path.is_file() and not path.is_symlink():
            by_size[path.stat().st_size].append(path)

    keys: dict[Path, str] = {}
    for size, paths in by_size.items():
        if len(paths) == 1:
            keys[paths[0]] = f"{size}:{paths[0]}"
        else:
            keys.update({path: f"{size}:{_file_digest(path)}" for path in paths})
    return keys


def _prune_source_only_step_folders(path: Path) -> None:
    """Delete step folders holding nothing but python tasks' own source.

    A bookkeeping task dumps its ``source_file`` — its code, which the
    installed package holds — and its ``aiida_node_metadata.yaml``, which
    names the node the folder came from. Any other file keeps the
    folder, whatever the process was: aiida-core writes ``node_outputs``
    only for ``SinglefileData`` and ``FolderData``, so an ``ArrayData`` a
    task produced reaches disk only as the ``function_inputs`` of
    whatever consumed it, and a calculation killed before retrieval has
    inputs and no outputs at all.

    Runs innermost first, so a folder left holding only such folders goes
    with them.
    """
    for child in _step_folders(path):
        _prune_source_only_step_folders(child)
        if all(item.name in _NON_CONTENT_FILES for item in child.rglob("*") if item.is_file()):
            shutil.rmtree(child)


def _canonical_rank(path: Path, root_path: Path) -> tuple[int, str]:
    """Return the sort key that picks which copy of a file stays real.

    A copy under a step's ``outputs`` ranks first: that step produced the
    file, and is where a reader following a link should land rather than
    in some later step that was handed it. Tree order settles the rest.
    """
    relative = path.relative_to(root_path)
    return (0 if "outputs" in relative.parts else 1, str(relative))


def _link_duplicate_files(root_path: Path) -> int:
    """Replace repeated files with relative symlinks to one real copy.

    Every step that ran keeps its folder and its own listing; only the
    repeated bytes go, so a file staged into a later step's ``inputs``
    now points at the ``outputs`` it came from. Links are relative, so
    the tree survives being moved.

    An empty file is left alone, and never serves as a target. A
    successful run leaves an empty ``_scheduler-stderr.txt`` under every
    calculation; linking them together saves nothing and asserts a
    relationship between unrelated steps that the reader then has to
    puzzle out.

    A ``source_file`` links only to another ``source_file``: one task run
    once per orbital dumps its code under each, and those copies collapse
    like any other, but no data file is ever made to depend on a folder
    that carries only code.

    Expects a freshly written tree that holds no symlinks of its own. Any
    it does find are left where they are and never chosen as the copy to
    keep: a link ranked ahead of the real file would leave the two
    pointing at each other and the bytes gone.

    :param root_path: Root of the tidied tree.
    :return: How many symlinks were made.
    """
    groups: dict[tuple[bool, str], list[Path]] = defaultdict(list)
    for path, key in _content_keys(root_path).items():
        if path.stat().st_size:
            groups[(path.name == _TASK_SOURCE_FILE, key)].append(path)

    created = 0
    for paths in groups.values():
        if len(paths) < 2:
            continue
        canonical, *duplicates = sorted(paths, key=lambda path: _canonical_rank(path, root_path))
        for duplicate in duplicates:
            duplicate.unlink()
            duplicate.symlink_to(os.path.relpath(canonical, duplicate.parent))
            created += 1
    return created


def _display_order(children: Sequence[Path]) -> list[tuple[Path, str]]:
    """Return the step folders as ``(folder, label)`` in reading order.

    The dump numbers a fan-out in creation order, which is lexicographic
    by map key, so ``orb_10`` lands between ``orb_1`` and ``orb_2``.
    Sorting a family naturally puts those indices back in counting order.

    Only a family whose members already occupy consecutive positions is
    sorted. A family split by another step never was a fan-out — the
    three-step spin initialization runs ``nspin1``, ``nspin2_dummy``,
    ``nspin2`` — and pulling it together would put a step before the one
    whose output it reads.
    """
    entries = []
    for child in children:
        match = _STEP_FOLDER_NAME.match(child.name)
        if match is not None:
            entries.append((child, match.group(2)))

    positions: dict[str, list[int]] = defaultdict(list)
    for index, (_, label) in enumerate(entries):
        positions[_family_key(label)].append(index)

    ordered = list(entries)
    for indices in positions.values():
        if len(indices) > 1 and indices == list(range(indices[0], indices[-1] + 1)):
            run = sorted((entries[index] for index in indices), key=lambda e: _natural_key(e[1]))
            ordered[indices[0] : indices[-1] + 1] = run
    return ordered


def _renumber_step_folders(path: Path) -> None:
    """Renumber the step folders under ``path`` contiguously from one.

    Numbers follow :func:`_display_order` rather than the order the dump
    wrote, and keep the zero padding it used.
    """
    children = _step_folders(path)
    widths = [len(m.group(1)) for c in children if (m := _STEP_FOLDER_NAME.match(c.name))]
    width = max(widths, default=2)

    ordered = _display_order(children)
    final_names = [f"{number:0{width}d}-{label}" for number, (_, label) in enumerate(ordered, 1)]

    # Reordering can send a folder to a number a sibling still holds, so
    # everything that moves is parked under a name no step folder can
    # have before anything takes its final one.
    parked = []
    for index, ((child, _), final_name) in enumerate(zip(ordered, final_names, strict=True)):
        if child.name == final_name:
            parked.append(child)
            continue
        staging = child.parent / f".tidy-{index}-{child.name}"
        shutil.move(str(child), str(staging))
        parked.append(staging)

    for staged, final_name in zip(parked, final_names, strict=True):
        renamed = staged.parent / final_name
        if staged != renamed:
            shutil.move(str(staged), str(renamed))
        _renumber_step_folders(renamed)


def _strip_process_label_suffixes(path: Path) -> None:
    """Drop the trailing "-<ProcessLabel>" from every step folder.

    The CalcJob class name on a calculation folder and the WorkChain
    class name on the step wrapping it both go. A folder whose stripped
    name is already taken keeps its suffix.
    """
    for child in _step_folders(path):
        renamed = child
        match = _PROCESS_LABEL_SUFFIX.match(child.name)
        if match is not None:
            stripped = child.parent / match.group(1)
            if not stripped.exists():
                shutil.move(str(child), str(stripped))
                renamed = stripped
        _strip_process_label_suffixes(renamed)


def _hoist_lone_calculations(path: Path) -> None:
    """Lift a lone calculation's contents into the step folder holding it.

    Hoists only when that calculation is everything the folder holds, so
    a folder that also keeps a metadata file of its own — the root does —
    keeps the calculation's folder and the step name on it.

    Descends top-down and stops at the folder it hoists into, so a chain
    of single-child steps collapses by one layer only and every step name
    on the way survives.
    """
    children = list(path.iterdir())
    if len(children) == 1 and children[0].is_dir() and _is_calculation_folder(children[0]):
        calculation = children[0]
        for item in calculation.iterdir():
            shutil.move(str(item), str(path / item.name))
        calculation.rmdir()
        return
    for child in _step_folders(path):
        _hoist_lone_calculations(child)


def _tidy_dumped_tree(root_path: Path) -> None:
    """Prune, renumber and flatten the step folders of a dumped tree.

    The passes run in a fixed order:

    - a step folder holding nothing but python source goes;
    - the surviving siblings are renumbered contiguously from one;
    - every step folder drops its trailing "-<ProcessLabel>";
    - a step folder holding nothing but one calculation takes over its
      contents;
    - files repeated across steps become relative symlinks to one copy,
      and the root gains a ``README`` saying so.

    Pruning has to precede flattening: a step is left holding a single
    calculation only once its bookkeeping siblings are gone. Stripping
    has to precede flattening too, so that a hoisted-into step folder is
    named for its own step rather than for the calculation it absorbed.
    Linking comes last because the passes before it move folders, and a
    relative link written earlier would no longer point anywhere.

    Runs once, on a freshly dumped tree: hoisting deliberately collapses
    one layer per pass, so a second pass over the same tree collapses
    another.

    :param root_path: Root of the dumped tree; it is never itself pruned.
    """
    _prune_source_only_step_folders(root_path)
    _renumber_step_folders(root_path)
    _strip_process_label_suffixes(root_path)
    _hoist_lone_calculations(root_path)
    if _link_duplicate_files(root_path):
        (root_path / "README").write_text(_SYMLINK_README)


def _simplify_folder_names(root_path: Path) -> None:
    """Strip pk numbers and WorkGraph process labels from folder names.

    Processes directories bottom-up to avoid path issues when renaming
    parents, and leaves a folder untouched if the simplified name is
    already taken.

    :param root_path: Root directory to process.
    """
    # Get all directories, sorted by depth (deepest first) for bottom-up processing
    all_dirs = sorted(root_path.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    all_dirs = [d for d in all_dirs if d.is_dir()]

    for dir_path in all_dirs:
        new_name = dir_path.name
        for pattern in _FOLDER_NAME_SUFFIXES:
            match = pattern.match(new_name)
            if match:
                new_name = match.group(1)
        new_path = dir_path.parent / new_name
        if dir_path != new_path and not new_path.exists():
            shutil.move(str(dir_path), str(new_path))


def _simplify_calcjob_dump(output_path: Path) -> None:
    """Simplify the structure of a dumped CalcJobNode.

    - Merges node_inputs into inputs
    - Merges node_outputs into outputs
    - Removes the dump's own bookkeeping files

    :param output_path: Path to the dumped calculation directory.
    """
    # Merge node_inputs into inputs
    node_inputs = output_path / "node_inputs"
    inputs = output_path / "inputs"
    if node_inputs.exists():
        for item in node_inputs.iterdir():
            shutil.move(str(item), str(inputs / item.name))
        node_inputs.rmdir()

    # Merge node_outputs into outputs
    node_outputs = output_path / "node_outputs"
    outputs = output_path / "outputs"
    if node_outputs.exists():
        for item in node_outputs.iterdir():
            shutil.move(str(item), str(outputs / item.name))
        node_outputs.rmdir()

    # Drop the dump's own bookkeeping; the node metadata stays
    for filename in _DUMP_BOOKKEEPING_FILES:
        filepath = output_path / filename
        if filepath.exists():
            filepath.unlink()


def _node_metadata(metadata_path: Path) -> dict[str, Any]:
    """Return a dumped metadata file's ``Node data`` mapping, or ``{}``.

    A file that cannot be read at all — unparseable, not text, unreadable,
    or truncated mid-write — answers ``{}`` rather than raising, so a
    reader keeps whatever it was deciding from the file's absence.
    """
    import yaml

    try:
        parsed = yaml.safe_load(metadata_path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    node_data = parsed.get("Node data", {}) if isinstance(parsed, dict) else {}
    return node_data if isinstance(node_data, dict) else {}


def _describes_a_workflow(metadata_path: Path) -> bool:
    """Return whether a dumped metadata file records a workflow node.

    Reads the node's own ``node_type``, which every dumped process
    records: ``process.workflow.…`` for a workgraph or a WorkChain,
    ``process.calculation.…`` for a CalcJob or a python task. Nothing
    else in the file decides, and a file that cannot be read at all
    answers no, so it is kept rather than deleted on a guess.
    """
    node_type = _node_metadata(metadata_path).get("node_type")
    return isinstance(node_type, str) and node_type.startswith("process.workflow.")


def _json_value(node: orm.Data) -> Any:
    """Return ``node``'s value as a JSON-native object.

    Answers :data:`_NOT_JSON_REPRESENTABLE` for anything else, including a
    repository-backed ``Data`` type (``ArrayData``, ``FolderData``,
    ``SinglefileData``, ``RemoteData``, …), which has no JSON form.
    """
    from aiida import orm

    if isinstance(node, orm.Dict):
        return node.get_dict()  # type: ignore[no-untyped-call]
    if isinstance(node, orm.List):
        return node.get_list()  # type: ignore[no-untyped-call]
    if isinstance(node, (orm.Int, orm.Float, orm.Str, orm.Bool)):
        return node.value
    return _NOT_JSON_REPRESENTABLE


def _nest_by_link_label(flat: dict[str, Any]) -> dict[str, Any]:
    """Turn ``{link_label: value}`` into nested dicts on the ``__`` separator.

    Mirrors aiida-core's own reading of a namespaced link label as a path
    (``NodeRepoIoDumper._dump_calculation_io_files`` splits on the same
    separator to build a directory), so a namespace socket like ``alphas``
    with sub-outputs ``filled``/``empty`` nests the same way here as its
    repository-backed siblings do under ``outputs/``.
    """
    nested: dict[str, Any] = {}
    for label, value in flat.items():
        cursor = nested
        parts = label.split("__")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


def _direct_calculation_created_pks(node: orm.WorkflowNode) -> frozenset[int]:
    """Return the pks a direct ``CALL_CALC`` child of ``node`` created.

    Only a one-hop calculation child counts: :func:`_hoist_lone_calculations`
    only ever merges a wrapping step's *own* calculation folder up into it,
    never a grandchild's, so a value that only becomes redundant two or
    more layers down (the root re-exporting ``RunFinalKI``'s own re-export
    of ``ki_final``'s output) is not caught here — dropping it there would
    delete the only copy visible at that level.
    """
    from aiida.common import LinkType

    created: set[int] = set()
    for call_link in node.base.links.get_outgoing(link_type=LinkType.CALL_CALC).all():
        for create_link in call_link.node.base.links.get_outgoing(link_type=LinkType.CREATE).all():
            if create_link.node.pk is not None:
                created.add(create_link.node.pk)
    return frozenset(created)


def _json_representable_links(
    node: orm.ProcessNode,
    link_type: LinkType,
    incoming: bool,
    exclude_pks: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Return ``{link_label: value}`` for the JSON-representable ``Data`` at ``link_type``.

    A linked node whose pk is in ``exclude_pks`` is skipped, whatever its
    link label — used to drop a workflow's re-export of a value its own
    hoisted calculation already carries under a different name.
    """
    from aiida import orm

    links = (
        node.base.links.get_incoming(link_type=link_type)
        if incoming
        else node.base.links.get_outgoing(link_type=link_type)
    ).all()
    flat = {}
    for link in links:
        if not isinstance(link.node, orm.Data) or link.node.pk in exclude_pks:
            continue
        value = _json_value(link.node)
        if value is not _NOT_JSON_REPRESENTABLE:
            flat[link.link_label] = value
    return flat


def _write_data_json(node: orm.ProcessNode, folder: Path) -> None:
    """Write ``folder``'s :data:`INPUTS_JSON_FILE`/:data:`OUTPUTS_JSON_FILE`.

    A calculation's own ``INPUT_CALC``/``CREATE`` links give its inputs and
    outputs. A workflow does not create data itself, so its ``RETURN``
    links — the same ones a reader would follow from ``process.outputs`` —
    stand in for :data:`OUTPUTS_JSON_FILE`. No :data:`INPUTS_JSON_FILE` is
    written for a workflow, since its ``INPUT_WORK`` links point at the
    same Data its own calculations already read.

    A workflow wrapping exactly one calculation is later hoisted into that
    calculation's own folder (:func:`_hoist_lone_calculations`), so a
    workflow's ``outputs.json`` never blocks that hoist: any RETURN value
    the workflow's own direct calculation child already created is dropped
    here (:func:`_direct_calculation_created_pks`) — the same node under a
    second name — leaving a value genuinely produced elsewhere in the
    workflow's own outputs.json (e.g. ``ComputeScreeningParameters``'s
    ``alphas``, assembled several layers inside a sub-workgraph child, not
    by a direct calculation child) untouched. A wrapper that drops down to
    nothing writes no file at all, so the pre-existing single-calculation
    check in :func:`_hoist_lone_calculations` runs exactly as it did before
    this file existed. Neither file is written when nothing linked has a
    JSON-representable value.

    :param node: The process the folder was dumped from.
    :param folder: The step's own dumped folder.
    """
    from aiida import orm
    from aiida.common import LinkType

    links_to_write: tuple[tuple[str, LinkType, bool, frozenset[int]], ...]
    if isinstance(node, orm.CalculationNode):
        links_to_write = (
            (INPUTS_JSON_FILE, LinkType.INPUT_CALC, True, frozenset()),
            (OUTPUTS_JSON_FILE, LinkType.CREATE, False, frozenset()),
        )
    elif isinstance(node, orm.WorkflowNode):
        exclude_pks = _direct_calculation_created_pks(node)
        links_to_write = ((OUTPUTS_JSON_FILE, LinkType.RETURN, False, exclude_pks),)
    else:
        return

    for filename, link_type, incoming, exclude_pks in links_to_write:
        flat = _json_representable_links(node, link_type, incoming, exclude_pks)
        if flat:
            (folder / filename).write_text(
                json.dumps(_nest_by_link_label(flat), indent=2, sort_keys=True) + "\n"
            )


def _dump_data_json(root_path: Path) -> None:
    """Write every step's :data:`INPUTS_JSON_FILE`/:data:`OUTPUTS_JSON_FILE`.

    Reads each step's pk from its own :data:`NODE_METADATA_FILE` — the file
    that already ties a folder back to its node — so this has to run before
    :func:`_prune_workflow_metadata` deletes the workflow ones, and before
    :func:`_tidy_dumped_tree` moves folders around.

    :param root_path: Root of the freshly dumped tree.
    """
    from aiida import orm

    for metadata_path in root_path.rglob(NODE_METADATA_FILE):
        pk = _node_metadata(metadata_path).get("pk")
        if pk is None:
            continue
        node = orm.load_node(pk)
        if isinstance(node, orm.ProcessNode):
            _write_data_json(node, metadata_path.parent)


def _prune_workflow_metadata(root_path: Path) -> None:
    """Delete every workflow node's metadata file below the root.

    A workflow folder runs nothing itself — the calculations under it do
    — so its metadata sits between the reader and the data, and its
    presence would stop a step folder wrapping a lone calculation from
    being flattened into it. The root keeps its own file: that pk is the
    handle to the whole run.

    :param root_path: Root of the dumped tree.
    """
    for path in root_path.rglob(NODE_METADATA_FILE):
        if path.parent != root_path and _describes_a_workflow(path):
            path.unlink()


def trained_model_output(process: orm.ProcessNode) -> orm.Dict | None:
    """Return the process's non-empty trained-model ``Dict`` output, if any."""
    from aiida import orm

    model_node = getattr(process.outputs, "model", None)
    if isinstance(model_node, orm.Dict) and model_node.get_dict():  # type: ignore[no-untyped-call]
        return model_node
    return None


def _dump_model_json(process: orm.ProcessNode, output_path: Path) -> None:
    """Write a trained screening model as a ``model.json`` copy.

    This copy is what a later run names through ``ml: {model_file: ...}``;
    the stored ``model`` ``orm.Dict`` output remains in the database, and
    ``ml: {model: ...}`` names it there. Processes without a non-empty
    ``model`` Dict output are left alone.
    """
    import json

    model_node = trained_model_output(process)
    if model_node is None:
        return
    model = model_node.get_dict()  # type: ignore[no-untyped-call]
    (output_path / MODEL_FILENAME).write_text(json.dumps(model, indent=2) + "\n")


def dump_workgraph(
    process: orm.ProcessNode,
    output_path: Path,
    overwrite: bool = True,
) -> Path:
    """Dump a workgraph to a local directory with simplified structure.

    Uses AiiDA's dump functionality, then:
    - Strips pk numbers and WorkGraph process labels from folder names
    - Simplifies each CalcJobNode folder structure
    - Removes the dump's own bookkeeping files
    - Writes each step's JSON-representable ``Data`` inputs/outputs (see
      :func:`_dump_data_json`)
    - Keeps each calculation's ``aiida_node_metadata.yaml``, and the
      root's (see :func:`_prune_workflow_metadata`)
    - Tidies the step folders (see :func:`_tidy_dumped_tree`)

    :param process: The workgraph ProcessNode.
    :param output_path: Output directory. Defaults to current working directory.
    :return: Path where the workgraph was dumped.
    """
    if overwrite and output_path.exists():
        shutil.rmtree(output_path)

    # Use AiiDA's dump to create the initial structure. ``dump_unsealed=True``
    # so a workgraph killed by the progress UI's fast-fail path (or otherwise
    # terminated without sealing) can still be inspected — the alternative is
    # a hard ``ExportValidationError`` and no on-disk artifact at all.
    # ``include_attributes=False`` keeps each node's metadata file to its
    # identity: a workgraph's attributes carry the serialized graph, which
    # grows with the graph and dwarfs everything else in the file.
    with suppress_aiida_logging():
        process.dump(
            output_path=output_path,
            include_inputs=True,
            include_outputs=True,
            include_attributes=False,
            overwrite=True,
            dump_unsealed=True,
        )

    # Strip pk numbers and WorkGraph process labels from all folder names
    _simplify_folder_names(output_path)

    # Simplify each CalcJobNode folder (merge node_inputs/outputs)
    for folder in output_path.rglob("*"):
        # CalcJob folders are identified by having an "inputs" subdirectory
        if folder.is_dir() and (folder / "inputs").exists():
            _simplify_calcjob_dump(folder)

    # Remove the dump's own bookkeeping throughout the tree
    for filename in _DUMP_BOOKKEEPING_FILES:
        for filepath in output_path.rglob(filename):
            filepath.unlink()

    # Every step's JSON-representable Data, while every folder still
    # carries the metadata file naming its node
    _dump_data_json(output_path)

    _prune_workflow_metadata(output_path)

    _tidy_dumped_tree(output_path)

    _dump_model_json(process, output_path)

    return output_path
