"""Utilities for dumping AiiDA calculations to local file structures."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from koopmans.aiida.utils import suppress_aiida_logging

if TYPE_CHECKING:
    from aiida import orm

__all__ = ["dump_workgraph"]


def _strip_pk_from_folder_names(root_path: Path) -> None:
    """Remove trailing pk numbers from folder names throughout a directory tree.

    AiiDA's dump creates folders like "WorkGraphName-1234" or "01-task-5678".
    This function strips the trailing "-<number>" from all folder names.

    Processes directories bottom-up to avoid path issues when renaming parents.

    :param root_path: Root directory to process.
    """
    # Pattern to match folder names ending with -<pk_number>
    pk_pattern = re.compile(r"^(.+)-(\d+)$")

    # Get all directories, sorted by depth (deepest first) for bottom-up processing
    all_dirs = sorted(root_path.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    all_dirs = [d for d in all_dirs if d.is_dir()]

    for dir_path in all_dirs:
        match = pk_pattern.match(dir_path.name)
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


def dump_workgraph(
    process: orm.ProcessNode,
    output_path: Path,
    overwrite: bool = True,
) -> Path:
    """Dump a workgraph to a local directory with simplified structure.

    Uses AiiDA's dump functionality, then:
    - Renames CalcJobNode folders to have descriptive names (e.g., "01-pw-scf")
    - Simplifies each CalcJobNode folder structure
    - Removes top-level metadata files

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

    # Strip pk numbers from all folder names
    _strip_pk_from_folder_names(output_path)

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

    return output_path
