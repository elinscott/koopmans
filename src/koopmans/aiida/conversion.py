"""Convert KoopmansInput to AiiDA data nodes.

This module provides utilities to convert parsed input files into
AiiDA-compatible data structures for use with workgraphs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiida import orm
from aiida.tools import get_kpoints_path


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
    from koopmans.input_file import KoopmansInput
    from koopmans.input_file.cell_parameters import (
        CellParametersViaAlat,
        CellParametersViaIbrav,
        CellParametersViaVectors,
    )
    from koopmans.input_file.input_file import AtomsInput, KpointsInput

# Bohr to Angstrom conversion factor
BOHR_TO_ANGSTROM = 0.529177210903


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
                    1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma
                )
                / sin_gamma,
            ],
        ]
    else:
        raise NotImplementedError(f"ibrav={ibrav} is not yet implemented")


def atoms_input_to_structure(atoms: AtomsInput) -> orm.StructureData:
    """Convert AtomsInput to AiiDA StructureData.

    Args:
        atoms: The atoms input from KoopmansInput.

    Returns:
        AiiDA StructureData node.
    """
    from koopmans.input_file.cell_parameters import (
        CellParametersViaAlat,
        CellParametersViaIbrav,
        CellParametersViaVectors,
    )

    cell_params = atoms.cell_parameters
    positions = atoms.atomic_positions

    # Determine cell vectors
    if isinstance(cell_params, CellParametersViaIbrav):
        cell = celldms_to_cell(cell_params.ibrav, cell_params.celldms)
    elif isinstance(cell_params, CellParametersViaVectors):
        cell = [list(v) for v in cell_params.vectors]
        if cell_params.units == "bohr":
            cell = [[x * BOHR_TO_ANGSTROM for x in row] for row in cell]
    elif isinstance(cell_params, CellParametersViaAlat):
        alat = cell_params.celldms[1] * BOHR_TO_ANGSTROM
        cell = [[x * alat for x in v] for v in cell_params.vectors]
    else:
        raise TypeError(f"Unknown cell_parameters type: {type(cell_params)}")

    # Determine periodicity
    pbc = cell_params.periodic
    if isinstance(pbc, bool):
        pbc = (pbc, pbc, pbc)

    # Create structure
    structure = orm.StructureData(cell=cell, pbc=pbc)

    # Add atoms
    units = positions.units
    for pos in positions.positions:
        symbol = pos[0]
        coords = pos[1:4]

        if units == "crystal":
            # Convert fractional to Cartesian
            cart_coords = [
                sum(coords[j] * cell[j][i] for j in range(3)) for i in range(3)
            ]
        elif units == "bohr":
            cart_coords = [c * BOHR_TO_ANGSTROM for c in coords]
        elif units in ("ang", "angstrom"):
            cart_coords = list(coords)
        else:
            raise ValueError(f"Unknown atomic position units: {units}")

        structure.append_atom(position=cart_coords, symbols=symbol)

    return structure


def kpoints_input_to_kpoints_mesh(kpoints: KpointsInput) -> orm.KpointsData:
    """Convert KpointsInput to AiiDA KpointsData for SCF calculations.

    Args:
        kpoints: The kpoints input from KoopmansInput.

    Returns:
        AiiDA KpointsData node with k-point mesh.
    """
    kpts = orm.KpointsData()
    kpts.set_kpoints_mesh(list(kpoints.grid), offset=list(kpoints.offset))
    return kpts


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
    import numpy as np

    # Get seekpath result for special point coordinates and default path
    result = get_kpoints_path(structure, method="seekpath")
    point_coords: dict[str, list[float]] = result["parameters"].dict["point_coords"]

    if kpoints.path is not None:
        # Parse the user-specified path string (e.g., "GXMG" or "GXMG,YZ")
        # Labels are concatenated without separators; "," indicates a break in the path
        path = []

        # Split by comma to get continuous segments
        segments = kpoints.path.split(",")

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Parse labels from the segment by matching against known point names
            # Sort by length descending to match longer labels first (e.g., "GAMMA" before "G")
            # Also include "G" as an alias for "GAMMA"
            available_labels = set(point_coords.keys())
            if "GAMMA" in available_labels:
                available_labels.add("G")
            sorted_labels = sorted(available_labels, key=len, reverse=True)
            labels = []
            remaining = segment

            while remaining:
                matched = False
                for label in sorted_labels:
                    if remaining.startswith(label):
                        # Map "G" to "GAMMA"
                        actual_label = "GAMMA" if label == "G" else label
                        labels.append(actual_label)
                        remaining = remaining[len(label):]
                        matched = True
                        break
                if not matched:
                    raise ValueError(
                        f"Unknown special point starting at '{remaining}' in k-path segment '{segment}'. "
                        f"Available points: {sorted(point_coords.keys())}"
                    )

            # Build path tuples for this segment
            for i in range(len(labels) - 1):
                path.append((labels[i], labels[i + 1]))
    else:
        # Use the default path from seekpath
        path = result["parameters"].dict["path"]

    # Calculate k-points along the path with the specified density
    kpoint_list = []
    label_list = []

    for segment_idx, (start_label, end_label) in enumerate(path):
        start_coord = np.array(point_coords[start_label])
        end_coord = np.array(point_coords[end_label])

        # Calculate distance in reciprocal space (approximate)
        segment_length = np.linalg.norm(end_coord - start_coord)

        # Number of points based on density
        n_points = max(2, int(np.ceil(segment_length * kpoints.density)))

        # Generate points along this segment
        for i in range(n_points):
            if i == 0 and segment_idx > 0:
                # Skip first point of segments after the first to avoid duplicates
                continue

            t = i / (n_points - 1) if n_points > 1 else 0.0
            coord = start_coord + t * (end_coord - start_coord)
            kpoint_list.append(coord.tolist())

            # Add labels for first and last points of segments
            if i == 0:
                label_list.append((len(kpoint_list) - 1, start_label))
            elif i == n_points - 1:
                label_list.append((len(kpoint_list) - 1, end_label))

    # Create KpointsData with explicit k-points
    kpts = orm.KpointsData()
    kpts.set_kpoints(kpoint_list)
    kpts.labels = label_list

    return kpts


def input_to_pw_parameters(koopmans_input: KoopmansInput) -> orm.Dict:
    """Convert KoopmansInput to PW input parameters Dict.

    Args:
        koopmans_input: The parsed koopmans input.

    Returns:
        AiiDA Dict node with PW parameters.
    """
    calc_params = koopmans_input.calculator_parameters
    pw_params = calc_params.pw

    # Build parameters dict
    parameters: dict = {
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
        parameters["CONTROL"].update(pw_params.control.model_dump(exclude_none=True))
    if pw_params.system:
        parameters["SYSTEM"].update(pw_params.system.model_dump(exclude_none=True))
    if pw_params.electrons:
        parameters["ELECTRONS"].update(pw_params.electrons.model_dump(exclude_none=True))

    # Ensure all Path objects are converted to strings for JSON serialization
    parameters = _convert_paths_to_strings(parameters)

    return orm.Dict(parameters)


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

    from koopmans.aiida.setup import ensure_pseudo_family_installed

    ensure_pseudo_family_installed(pseudo_family)
    family = PseudoPotentialFamily.collection.get(label=pseudo_family)
    return family.get_pseudos(structure=structure)


def convert_koopmans_input(
    koopmans_input: KoopmansInput,
) -> dict:
    """Convert KoopmansInput to a dictionary of AiiDA data nodes.

    Args:
        koopmans_input: The parsed koopmans input.

    Returns:
        Dictionary containing:
        - structure: StructureData
        - kpoints_scf: KpointsData (mesh for SCF)
        - kpoints_bands: KpointsData (path for bands)
        - parameters: Dict (PW parameters)
        - pseudos: dict mapping element symbols to pseudopotential nodes
    """
    structure = atoms_input_to_structure(koopmans_input.atoms)
    kpoints_scf = kpoints_input_to_kpoints_mesh(koopmans_input.kpoints)
    kpoints_bands = kpoints_input_to_kpoints_path(koopmans_input.kpoints, structure)
    parameters = input_to_pw_parameters(koopmans_input)
    pseudos = get_pseudos_from_family(koopmans_input.workflow.pseudo_library, structure)

    return {
        "structure": structure,
        "kpoints_scf": kpoints_scf,
        "kpoints_bands": kpoints_bands,
        "parameters": parameters,
        "pseudos": pseudos,
    }
