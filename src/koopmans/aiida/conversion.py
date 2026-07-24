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
    from koopmans.input_file.parallelization import CodeParallelization

# Quantum ESPRESSO's own value, so that converted quantities match QE output
BOHR_TO_ANGSTROM: float = CONSTANTS.bohr_to_ang


def code_parallelization(
    config: CodeParallelization | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate a code's parallelization config into ``(options, settings)``.

    ``ntasks`` becomes the scheduler ``metadata.options.resources``
    (``tot_num_mpiprocs``); ``npool`` becomes ``-npool`` and ``pd`` becomes
    ``-pd true`` on the code's ``settings.cmdline`` (npool before pd). A
    missing field yields an empty dict for
    that half, so callers can merge selectively.

    ``omp`` is deliberately *not* translated here. This helper only seeds the
    shared scf/nscf/bands pw overrides, which ``aiida-koopmans``'s graph
    builders then re-merge from the same threaded mapping. The ntasks/npool/pd
    directives survive that double pass because re-merging rewrites them to the
    same value, but the omp export block is *appended* to any existing
    ``prepend_text``, so emitting it here as well would duplicate it. omp
    therefore rides the threaded mapping alone (via ``as_mapping``).

    Args:
        config: The per-code parallelization settings, or ``None``.

    Returns:
        A ``(options, settings)`` tuple of dicts, either of which may be empty.
    """
    options: dict[str, Any] = {}
    settings: dict[str, Any] = {}
    if config is None:
        return options, settings
    if config.ntasks is not None:
        options["resources"] = {"num_machines": 1, "tot_num_mpiprocs": config.ntasks}
    cmdline: list[str] = []
    if config.npool is not None:
        cmdline += ["-npool", str(config.npool)]
    if config.pd:
        cmdline += ["-pd", "true"]
    if cmdline:
        settings["cmdline"] = cmdline
    return options, settings


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
    cell_params = atoms.cell_parameters
    positions = atoms.atomic_positions

    cell = cell_in_angstrom(cell_params)

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


def _cell_special_points(structure: orm.StructureData) -> dict[str, list[float]]:
    """Return the cell's Bravais-lattice special k-points, in its own reciprocal basis.

    An explicit path string speaks the legacy (ASE) label vocabulary, which
    derives from the cell shape alone. Symmetry-detected (seekpath) labels are
    position-sensitive: nearly-symmetric positions (``0.3333`` vs ``1/3``)
    demote the lattice and rename every special point, so a legacy input's
    ``"ALMGAHK"`` would stop parsing on the very structure it was written for.
    """
    import numpy as np
    from ase.cell import Cell

    bandpath = Cell(np.array(structure.cell)).bandpath(npoints=0)  # type: ignore[no-untyped-call]
    # ASE keys the zone centre "G"; the path parser and the seekpath branch
    # both speak "GAMMA".
    return {
        ("GAMMA" if label == "G" else label): list(coords)
        for label, coords in bandpath.special_points.items()
    }


def kpoints_input_to_kpoints_path(
    kpoints: KpointsInput,
    structure: orm.StructureData,
) -> orm.KpointsData:
    """Convert KpointsInput to AiiDA KpointsData for bands calculations.

    An explicit path in the input is resolved against the cell's own
    Bravais-lattice special points (the legacy ASE vocabulary); when no path
    is given one is generated automatically with seekpath.

    Args:
        kpoints: The kpoints input from KoopmansInput.
        structure: The structure to generate k-path for.

    Returns:
        AiiDA KpointsData node with k-point path.
    """
    import numpy as np

    if kpoints.path is not None:
        point_coords = _cell_special_points(structure)
        path = _parse_kpoints_path_string(kpoints.path, point_coords)
    else:
        result = get_kpoints_path(structure, method="seekpath")  # type: ignore[no-untyped-call]
        primitive_cell = np.array(result["primitive_structure"].cell)
        input_cell = np.array(structure.cell)
        point_coords = result["parameters"].dict["point_coords"]
        path = result["parameters"].dict["path"]
        if not np.allclose(primitive_cell, input_cell, atol=1e-5):
            # seekpath re-chose the primitive lattice vectors (e.g. QE's ibrav=2
            # fcc convention vs the standardized choice), so its special-point
            # coordinates attach to a different reciprocal basis. When both cells
            # span the same lattice the mapping between them is an integer
            # unimodular matrix, and the special points can be re-expressed in the
            # input cell's own reciprocal basis via their Cartesian coordinates.
            transfer = input_cell @ np.linalg.inv(primitive_cell)
            if not (
                np.allclose(transfer, np.round(transfer), atol=1e-5)
                and np.isclose(abs(np.linalg.det(transfer)), 1.0, atol=1e-5)
            ):
                raise NotImplementedError(
                    "The input cell is not a primitive cell of the lattice seekpath "
                    "identified (it is a supercell or a rotated frame), so an automatic "
                    "k-point path cannot be expressed in the input cell's reciprocal "
                    "basis. Provide the structure as a primitive cell, or specify the "
                    "k-point path explicitly."
                )
            recip_primitive = np.linalg.inv(primitive_cell).T
            recip_input = np.linalg.inv(input_cell).T
            point_coords = {
                label: list(np.array(coords) @ recip_primitive @ np.linalg.inv(recip_input))
                for label, coords in point_coords.items()
            }

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

    # Build parameters dict. No CONTROL.calculation: this dict feeds every
    # step's overrides, so each step owner's protocol supplies its own
    # calculation type (scf, bands, nscf) and the override must not force one.
    parameters: dict[str, dict[str, Any]] = {
        "CONTROL": {},
        "SYSTEM": {},
        "ELECTRONS": {},
    }

    # Add ecutwfc if specified
    if calc_params.ecutwfc is not None:
        parameters["SYSTEM"]["ecutwfc"] = calc_params.ecutwfc
        # Pin ecutrho alongside it (kcp.x convention: 4x for norm-conserving,
        # same fallback as ``_extract_kcp_scalar_inputs``). Without this the
        # pseudo family's placeholder ecutrho wins in protocol-built PW runs,
        # putting them on a different grid from the CP supercell run that the
        # dft_init consistency checks compare against. An explicit
        # ``pw.system.ecutrho`` still overrides via the update below.
        kcp_system = calc_params.kcp.system
        parameters["SYSTEM"]["ecutrho"] = (
            kcp_system.ecutrho if kcp_system.ecutrho else 4.0 * calc_params.ecutwfc
        )

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
