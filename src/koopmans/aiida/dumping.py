"""Utilities for dumping AiiDA calculations to local file structures."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from koopmans.aiida.utils import suppress_aiida_logging

if TYPE_CHECKING:
    from aiida import orm

__all__ = ["dump_workgraph", "trained_model_output"]


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

# The CalcJob class name AiiDA appends to a calculation folder, e.g. the
# "-KcpCalculation" of "01-dft_init-KcpCalculation". What is left has to
# be a step folder in its own right, so that a folder whose link label is
# itself capitalised ("05-RunFinalKI") keeps its label.
_CLASS_NAME_SUFFIX = re.compile(r"^(\d+-.+)-[A-Z][A-Za-z0-9]*$")


def _is_calculation_folder(path: Path) -> bool:
    """Return whether the folder holds one dumped process, run and recorded."""
    return (path / "inputs").is_dir() and (path / "outputs").is_dir()


def _produced_outputs(path: Path) -> bool:
    """Return whether any process under the folder recorded outputs."""
    return any(candidate.is_dir() for candidate in path.rglob("outputs"))


def _step_folders(path: Path) -> list[Path]:
    """Return the folder's "<NN>-<label>" subfolders, in number order."""
    numbered = []
    for child in path.iterdir():
        match = _STEP_FOLDER_NAME.match(child.name)
        if child.is_dir() and match is not None:
            numbered.append((int(match.group(1)), len(match.group(1)), child))
    numbered.sort()
    return [child for _, _, child in numbered]


def _prune_outputless_step_folders(path: Path) -> None:
    """Delete step folders under which no process recorded outputs.

    Such a folder holds at most the bookkeeping task's own source and a
    serialized copy of the inputs it was handed, both of which the step
    that consumed them already carries. A folder holding only such
    folders goes with them.
    """
    for child in _step_folders(path):
        if _produced_outputs(child):
            _prune_outputless_step_folders(child)
        else:
            shutil.rmtree(child)


def _renumber_step_folders(path: Path) -> None:
    """Renumber the step folders under ``path`` contiguously from one.

    Keeps the order the original numbers gave and the zero padding they
    were written with.
    """
    children = _step_folders(path)
    matches = [_STEP_FOLDER_NAME.match(child.name) for child in children]
    width = max((len(m.group(1)) for m in matches if m is not None), default=2)

    for number, (child, match) in enumerate(zip(children, matches, strict=True), start=1):
        if match is None:
            continue
        renamed = child.parent / f"{number:0{width}d}-{match.group(2)}"
        # Compacting never raises a number, so the target is always free.
        if renamed != child:
            shutil.move(str(child), str(renamed))
        _renumber_step_folders(renamed)


def _strip_class_name_suffixes(path: Path) -> None:
    """Drop the trailing "-<ClassName>" from every calculation folder.

    A folder whose stripped name is already taken keeps its suffix.
    """
    for child in _step_folders(path):
        if not _is_calculation_folder(child):
            _strip_class_name_suffixes(child)
            continue
        match = _CLASS_NAME_SUFFIX.match(child.name)
        if match is None:
            continue
        stripped = child.parent / match.group(1)
        if not stripped.exists():
            shutil.move(str(child), str(stripped))


def _hoist_lone_calculations(path: Path) -> None:
    """Lift a lone calculation's contents into the step folder holding it.

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

    - a step folder under which nothing recorded outputs goes;
    - the surviving siblings are renumbered contiguously from one;
    - a calculation folder drops its trailing "-<ClassName>";
    - a step folder holding nothing but one calculation takes over its
      contents.

    Pruning has to precede flattening: a step is left holding a single
    calculation only once its bookkeeping siblings are gone. Stripping
    has to precede flattening too, because a step that has taken over a
    calculation's contents then looks like a calculation folder itself.

    Runs once, on a freshly dumped tree.

    :param root_path: Root of the dumped tree; it is never itself pruned.
    """
    _prune_outputless_step_folders(root_path)
    _renumber_step_folders(root_path)
    _strip_class_name_suffixes(root_path)
    _hoist_lone_calculations(root_path)


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
    - Removes metadata files (README.md, aiida_node_metadata.yaml, etc.)

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

    # Remove metadata files
    for filename in [
        "README.md",
        "aiida_node_metadata.yaml",
        "aiida_dump_log.json",
        ".aiida_dump_safeguard",
    ]:
        filepath = output_path / filename
        if filepath.exists():
            filepath.unlink()


def trained_model_output(process: orm.ProcessNode) -> orm.Dict | None:
    """Return the process's non-empty trained-model ``Dict`` output, if any."""
    from aiida import orm

    model_node = getattr(process.outputs, "model", None)
    if isinstance(model_node, orm.Dict) and model_node.get_dict():  # type: ignore[no-untyped-call]
        return model_node
    return None


def _dump_model_json(process: orm.ProcessNode, output_path: Path) -> None:
    """Write a trained screening model as a ``model.json`` convenience copy.

    The stored ``model`` ``orm.Dict`` output stays the canonical artifact
    (a later run references it via ``ml: {model: <pk-or-uuid>}``); the JSON
    copy feeds ``ml: {model_file: ...}`` outside the training profile.
    Processes without a non-empty ``model`` Dict output are left alone.
    """
    import json

    model_node = trained_model_output(process)
    if model_node is None:
        return
    model = model_node.get_dict()  # type: ignore[no-untyped-call]
    (output_path / "model.json").write_text(json.dumps(model, indent=2) + "\n")


def dump_workgraph(
    process: orm.ProcessNode,
    output_path: Path,
    overwrite: bool = True,
) -> Path:
    """Dump a workgraph to a local directory with simplified structure.

    Uses AiiDA's dump functionality, then:
    - Strips pk numbers and WorkGraph process labels from folder names
    - Simplifies each CalcJobNode folder structure
    - Removes top-level metadata files
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
    with suppress_aiida_logging():
        process.dump(
            output_path=output_path,
            include_inputs=True,
            include_outputs=True,
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

    # Remove metadata files throughout the tree
    for filename in [
        "README.md",
        "aiida_node_metadata.yaml",
        "aiida_dump_log.json",
        ".aiida_dump_safeguard",
    ]:
        for filepath in output_path.rglob(filename):
            filepath.unlink()

    # Only now, with the metadata files gone, does a step that ran no
    # calculation look empty.
    _tidy_dumped_tree(output_path)

    _dump_model_json(process, output_path)

    return output_path
