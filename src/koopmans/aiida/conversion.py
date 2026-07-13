"""Convert KoopmansInput to AiiDA data nodes.

This module provides utilities to convert parsed input files into
AiiDA-compatible data structures for use with workgraphs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiida import orm
from aiida.tools import get_kpoints_path
from qe_tools import CONSTANTS


def _convert_paths_to_strings(obj: Any) -> Any:
    """Recursively convert Path objects to strings in a nested structure."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths_to_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_paths_to_strings(item) for item in obj)
    return obj


if TYPE_CHECKING:
    from koopmans.input_file import AtomsInput, KoopmansInput, KpointsInput
    from koopmans.input_file.cell_parameters import (
        CellParametersViaAlat,
        CellParametersViaIbrav,
        CellParametersViaVectors,
    )

# Quantum ESPRESSO's own value, so that converted quantities match QE output
BOHR_TO_ANGSTROM: float = CONSTANTS.bohr_to_ang


def celldms_to_cell(ibrav: int, celldms: dict[int, float]) -> list[list[float]]:
    """Convert ibrav and celldms to cell vectors in Angstrom.

    This implements the Quantum ESPRESSO ibrav conventions.
    See: https://www.quantum-espresso.org/Doc/INPUT_PW.html#idm226

    Args:
        ibrav: Bravais lattice index.
        celldms: Dictionary of cell dimensions (celldm(1) to celldm(6)).

    Returns:
        3x3 list of cell vectors in Angstrom.
    """
    import math

    a = celldms[1] * BOHR_TO_ANGSTROM  # celldm(1) is in Bohr
    b = celldms.get(2, 1.0) * a if 2 in celldms else a
    c = celldms.get(3, 1.0) * a if 3 in celldms else a
    cos_alpha = celldms.get(4, 0.0)
    cos_beta = celldms.get(5, 0.0)
    cos_gamma = celldms.get(6, 0.0)

    if ibrav == 1:
        # Cubic P (sc)
        return [[a, 0, 0], [0, a, 0], [0, 0, a]]
    elif ibrav == 2:
        # Cubic F (fcc)
        return [[-a / 2, 0, a / 2], [0, a / 2, a / 2], [-a / 2, a / 2, 0]]
    elif ibrav == 3:
        # Cubic I (bcc)
        return [[a / 2, a / 2, a / 2], [-a / 2, a / 2, a / 2], [-a / 2, -a / 2, a / 2]]
    elif ibrav == 4:
        # Hexagonal
        return [[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, c]]
    elif ibrav == 5:
        # Trigonal R, 3-fold axis c
        tx = math.sqrt((1 - cos_alpha) / 2)
        ty = math.sqrt((1 - cos_alpha) / 6)
        tz = math.sqrt((1 + 2 * cos_alpha) / 3)
        return [[a * tx, -a * ty, a * tz], [0, 2 * a * ty, a * tz], [-a * tx, -a * ty, a * tz]]
    elif ibrav == 6:
        # Tetragonal P (st)
        return [[a, 0, 0], [0, a, 0], [0, 0, c]]
    elif ibrav == 7:
        # Tetragonal I (bct)
        return [[a / 2, -a / 2, c / 2], [a / 2, a / 2, c / 2], [-a / 2, -a / 2, c / 2]]
    elif ibrav == 8:
        # Orthorhombic P
        return [[a, 0, 0], [0, b, 0], [0, 0, c]]
    elif ibrav == 14:
        # Triclinic
        sin_gamma = math.sqrt(1 - cos_gamma**2)
        return [
            [a, 0, 0],
            [b * cos_gamma, b * sin_gamma, 0],
            [
                c * cos_beta,
                c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                c
                * math.sqrt(
                    1
                    - cos_alpha**2
                    - cos_beta**2
                    - cos_gamma**2
                    + 2 * cos_alpha * cos_beta * cos_gamma
                )
                / sin_gamma,
            ],
        ]
    else:
        raise NotImplementedError(f"ibrav={ibrav} is not yet implemented")


def cell_in_angstrom(
    cell_params: CellParametersViaIbrav | CellParametersViaVectors | CellParametersViaAlat,
) -> list[list[float]]:
    """Return the cell vectors in Angstrom for any of the specification variants.

    Args:
        cell_params: The cell parameters from the input file.

    Returns:
        The cell vectors in Angstrom.
    """
    from koopmans.input_file.cell_parameters import (
        CellParametersViaAlat,
        CellParametersViaIbrav,
        CellParametersViaVectors,
    )

    if isinstance(cell_params, CellParametersViaIbrav):
        return celldms_to_cell(cell_params.ibrav, cell_params.celldms)
    if isinstance(cell_params, CellParametersViaVectors):
        cell = [list(v) for v in cell_params.vectors]
        if cell_params.units == "bohr":
            cell = [[x * BOHR_TO_ANGSTROM for x in row] for row in cell]
        return cell
    if isinstance(cell_params, CellParametersViaAlat):
        alat = cell_params.celldms[1] * BOHR_TO_ANGSTROM
        return [[x * alat for x in v] for v in cell_params.vectors]
    raise TypeError(f"Unknown cell_parameters type: {type(cell_params)}")


def alat_in_angstrom(
    cell_params: CellParametersViaIbrav | CellParametersViaVectors | CellParametersViaAlat,
    cell: list[list[float]],
) -> float:
    """Return the lattice parameter ``alat`` in Angstrom.

    Follows Quantum ESPRESSO's convention: ``celldm(1)`` when given, otherwise
    the length of the first cell vector.

    Args:
        cell_params: The cell parameters from the input file.
        cell: The cell vectors in Angstrom.

    Returns:
        The lattice parameter in Angstrom.
    """
    from koopmans.input_file.cell_parameters import CellParametersViaVectors

    if isinstance(cell_params, CellParametersViaVectors):
        return float(sum(x**2 for x in cell[0]) ** 0.5)
    return cell_params.celldms[1] * BOHR_TO_ANGSTROM


def atoms_input_to_structure(atoms: AtomsInput) -> orm.StructureData:
    """Convert AtomsInput to AiiDA StructureData.

    Args:
        atoms: The atoms input from KoopmansInput.

    Returns:
        AiiDA StructureData node.
    """
    from koopmans.input_file.atomic_positions import SnapshotsInput

    cell_params = atoms.cell_parameters
    positions = atoms.atomic_positions
    if isinstance(positions, SnapshotsInput):
        raise ValueError(
            "atoms_input_to_structure needs explicit atomic positions; expand a "
            "snapshots trajectory into per-frame AtomsInputs first."
        )

    cell = cell_in_angstrom(cell_params)

    # Determine periodicity
    pbc = cell_params.periodic
    if isinstance(pbc, bool):
        pbc = (pbc, pbc, pbc)

    # Create structure
    structure = orm.StructureData(cell=cell, pbc=pbc)  # type: ignore[no-untyped-call]

    # Add atoms
    units = positions.units
    for pos in positions.positions:
        symbol = pos[0]
        coords = pos[1:4]

        if units == "crystal":
            # Convert fractional to Cartesian
            cart_coords = [sum(coords[j] * cell[j][i] for j in range(3)) for i in range(3)]
        elif units == "alat":
            cart_coords = [c * alat_in_angstrom(cell_params, cell) for c in coords]
        elif units == "bohr":
            cart_coords = [c * BOHR_TO_ANGSTROM for c in coords]
        elif units in ("ang", "angstrom"):
            cart_coords = list(coords)
        else:
            raise ValueError(f"Unknown atomic position units: {units}")

        structure.append_atom(position=cart_coords, symbols=symbol)  # type: ignore[no-untyped-call]

    return structure


def kpoints_input_to_kpoints_mesh(kpoints: KpointsInput) -> orm.KpointsData:
    """Convert KpointsInput to AiiDA KpointsData for SCF calculations.

    Args:
        kpoints: The kpoints input from KoopmansInput.

    Returns:
        AiiDA KpointsData node with k-point mesh.
    """
    kpts = orm.KpointsData()
    kpts.set_kpoints_mesh(list(kpoints.grid), offset=list(kpoints.offset))  # type: ignore[no-untyped-call]
    return kpts


def _parse_kpoints_path_string(
    path_string: str, point_coords: dict[str, list[float]]
) -> list[tuple[str, str]]:
    """Parse a user-specified k-path string into a list of segment tuples.

    Args:
        path_string: Path string like ``"GXMG"`` or ``"GXMG,YZ"`` where ``,`` indicates a break.
        point_coords: Dict mapping special point labels to their coordinates.

    Returns:
        List of (start_label, end_label) tuples defining path segments.

    Raises:
        ValueError: If an unknown special point is found in the path string.
    """
    path = []

    # Build set of available labels, adding "G" as alias for "GAMMA"
    available_labels = set(point_coords.keys())
    if "GAMMA" in available_labels:
        available_labels.add("G")
    sorted_labels = sorted(available_labels, key=len, reverse=True)

    # Split by comma to get continuous segments
    for segment in path_string.split(","):
        segment = segment.strip()
        if not segment:
            continue

        # Parse labels by matching against known point names (longest first)
        labels = []
        remaining = segment

        while remaining:
            matched = False
            for label in sorted_labels:
                if remaining.startswith(label):
                    actual_label = "GAMMA" if label == "G" else label
                    labels.append(actual_label)
                    remaining = remaining[len(label) :]
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"Unknown special point starting at '{remaining}' "
                    f"in k-path segment '{segment}'. "
                    f"Available points: {sorted(point_coords.keys())}"
                )

        # Build path tuples for this segment
        for i in range(len(labels) - 1):
            path.append((labels[i], labels[i + 1]))

    return path


def _calculate_kpoints_along_path(
    path: list[tuple[str, str]],
    point_coords: dict[str, list[float]],
    density: float,
) -> tuple[list[list[float]], list[tuple[int, str]]]:
    """Calculate k-points along a path with the specified density.

    Args:
        path: List of (start_label, end_label) tuples defining path segments.
        point_coords: Dict mapping special point labels to their coordinates.
        density: Number of k-points per reciprocal space unit.

    Returns:
        Tuple of (kpoint_list, label_list) where kpoint_list contains coordinates
        and label_list contains (index, label) tuples for special points.
    """
    import numpy as np

    kpoint_list: list[list[float]] = []
    label_list: list[tuple[int, str]] = []

    previous_end: str | None = None
    for start_label, end_label in path:
        start_coord = np.array(point_coords[start_label])
        end_coord = np.array(point_coords[end_label])

        segment_length = np.linalg.norm(end_coord - start_coord)
        n_points = max(2, int(np.ceil(segment_length * density)))

        for i in range(n_points):
            if i == 0 and start_label == previous_end:
                # The previous segment already emitted this point. (At a
                # discontinuity — a comma in the path string — the labels
                # differ and the new segment's start point must be kept.)
                continue

            t = i / (n_points - 1) if n_points > 1 else 0.0
            coord = start_coord + t * (end_coord - start_coord)
            kpoint_list.append(coord.tolist())

            if i == 0:
                label_list.append((len(kpoint_list) - 1, start_label))
            elif i == n_points - 1:
                label_list.append((len(kpoint_list) - 1, end_label))

        previous_end = end_label

    return kpoint_list, label_list


def kpoints_input_to_kpoints_path(
    kpoints: KpointsInput,
    structure: orm.StructureData,
) -> orm.KpointsData:
    """Convert KpointsInput to AiiDA KpointsData for bands calculations.

    Uses the k-point path specified in the input, or generates one automatically
    using seekpath if not specified.

    Args:
        kpoints: The kpoints input from KoopmansInput.
        structure: The structure to generate k-path for.

    Returns:
        AiiDA KpointsData node with k-point path.
    """
    result = get_kpoints_path(structure, method="seekpath")  # type: ignore[no-untyped-call]
    point_coords: dict[str, list[float]] = result["parameters"].dict["point_coords"]

    # The special-point coordinates are fractional in seekpath's standardized
    # primitive cell. If seekpath standardized the input cell (rotated or
    # re-chose the primitive vectors), those coordinates would be attached to
    # the wrong reciprocal basis — refuse rather than sample wrong directions.
    import numpy as np

    primitive_cell = np.array(result["primitive_structure"].cell)
    if not np.allclose(primitive_cell, np.array(structure.cell), atol=1e-5):
        raise NotImplementedError(
            "seekpath standardized the input cell, so the generated k-path would "
            "not match the input structure's reciprocal basis. Re-express the "
            "structure in seekpath's standardized primitive cell (see "
            "https://seekpath.readthedocs.io) or provide the cell in that form."
        )

    if kpoints.path is not None:
        path = _parse_kpoints_path_string(kpoints.path, point_coords)
    else:
        path = result["parameters"].dict["path"]

    kpoint_list, label_list = _calculate_kpoints_along_path(path, point_coords, kpoints.density)

    kpts = orm.KpointsData()
    kpts.set_kpoints(kpoint_list)  # type: ignore[no-untyped-call]
    kpts.labels = label_list

    return kpts


def input_to_pw_parameters(koopmans_input: KoopmansInput) -> dict[str, dict[str, Any]]:
    """Convert KoopmansInput to a PW input-parameter namelist dict.

    The dispatcher hands this straight into a builder ``overrides`` mapping;
    aiida-workgraph wraps it into ``orm.Dict`` at the CalcJob socket.
    """
    calc_params = koopmans_input.calculator_parameters
    pw_params = calc_params.pw

    # Build parameters dict
    parameters: dict[str, dict[str, Any]] = {
        "CONTROL": {
            "calculation": "scf",
        },
        "SYSTEM": {},
        "ELECTRONS": {},
    }

    # Add ecutwfc if specified
    if calc_params.ecutwfc is not None:
        parameters["SYSTEM"]["ecutwfc"] = calc_params.ecutwfc

    # Add nbnd if specified
    if calc_params.nbnd is not None:
        parameters["SYSTEM"]["nbnd"] = int(calc_params.nbnd)

    # Merge with explicit PW parameters from input
    if pw_params.control:
        parameters["CONTROL"].update(
            pw_params.control.model_dump(exclude_none=True, exclude_defaults=True)
        )
    if pw_params.system:
        parameters["SYSTEM"].update(
            pw_params.system.model_dump(exclude_none=True, exclude_defaults=True)
        )
    if pw_params.electrons:
        parameters["ELECTRONS"].update(
            pw_params.electrons.model_dump(exclude_none=True, exclude_defaults=True)
        )
    # Ensure all Path objects are converted to strings for JSON serialization
    parameters = _convert_paths_to_strings(parameters)

    return parameters


def get_pseudos_from_family(
    pseudo_family: str,
    structure: orm.StructureData,
) -> dict[str, orm.UpfData]:
    """Get pseudopotentials from a family for the elements in a structure.

    If the family is not installed, attempts to install it first.

    Args:
        pseudo_family: The label of the pseudopotential family.
        structure: The structure to get pseudopotentials for.

    Returns:
        Dictionary mapping element symbols to pseudopotential nodes.
    """
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    from koopmans.aiida.setup.pseudos import ensure_pseudo_family_installed

    ensure_pseudo_family_installed(pseudo_family)
    family = PseudoPotentialFamily.collection.get(label=pseudo_family)
    return family.get_pseudos(structure=structure)  # type: ignore[no-any-return]
