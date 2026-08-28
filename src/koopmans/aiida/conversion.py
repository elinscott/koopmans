"""Convert KoopmansInput to AiiDA data nodes.

This module provides utilities to convert parsed input files into
AiiDA-compatible data structures for use with workgraphs.
"""

from __future__ import annotations

import math
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

# ``ecutrho / ecutwfc`` for a norm-conserving pseudopotential, which is what
# kcp.x and kcw.x accept — and so the only kind koopmans runs.
NORM_CONSERVING_DUAL: float = 4.0


def code_parallelization(
    config: CodeParallelization | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate a code's parallelization config into ``(options, settings)``.

    ``ntasks`` becomes the scheduler ``metadata.options.resources``
    (``num_machines`` + ``num_mpiprocs_per_machine``, the one shape both the
    hyperqueue and node-counting schedulers consume — hyperqueue silently
    ignores ``tot_num_mpiprocs``); ``npool`` becomes ``-npool`` and ``pd`` becomes
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
        options["resources"] = {"num_machines": 1, "num_mpiprocs_per_machine": config.ntasks}
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

    if positions is None:
        raise ValueError(
            "`atoms.snapshots` (a multi-frame xyz path) is only supported by the "
            "`trajectory` task; this task expects explicit `atomic_positions`."
        )

    cell = cell_in_angstrom(cell_params)

    # Create structure
    structure = orm.StructureData(cell=cell, pbc=cell_params.periodic)

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


def atoms_input_to_structures(atoms: AtomsInput) -> dict[str, orm.StructureData]:
    """Convert a snapshots-carrying AtomsInput into per-frame StructureData nodes.

    Reads every frame of the ``snapshots`` xyz file. The cell and periodicity
    always come from the input file's ``cell_parameters`` block and override
    whatever the xyz records, on every frame. Frame coordinates are Cartesian
    Angstrom (the xyz convention), so they bypass the units machinery.

    Args:
        atoms: The atoms input from KoopmansInput, with its ``snapshots``
            field set.

    Returns:
        Mapping of ``snapshot_1 .. snapshot_N`` to AiiDA StructureData nodes.
        The keys are valid AiiDA link labels.

    Raises:
        ValueError: If ``atoms`` carries explicit ``atomic_positions`` rather
            than ``snapshots``.
    """
    from ase.io import read as ase_read

    if atoms.snapshots is None:
        raise ValueError(
            "the `trajectory` task requires `atoms.snapshots` (a multi-frame xyz path); "
            "got explicit `atomic_positions`."
        )

    frames = ase_read(atoms.snapshots, index=":")

    cell = cell_in_angstrom(atoms.cell_parameters)
    pbc = atoms.cell_parameters.periodic

    structures: dict[str, orm.StructureData] = {}
    for index, frame in enumerate(frames, start=1):
        structure = orm.StructureData(cell=cell, pbc=pbc)
        for symbol, position in zip(
            frame.get_chemical_symbols(), frame.get_positions(), strict=True
        ):
            structure.append_atom(  # type: ignore[no-untyped-call]
                position=[float(x) for x in position], symbols=symbol
            )
        structures[f"snapshot_{index}"] = structure

    return structures


def kpoints_input_to_kpoints_mesh(kpoints: KpointsInput) -> orm.KpointsData:
    """Convert KpointsInput to AiiDA KpointsData for SCF calculations.

    The offset passes straight through: the input file states it the way
    ``KpointsData`` does, as a fraction of a grid step. The schema is what
    keeps it to a value a ``K_POINTS automatic`` card can carry, so nothing
    here has to know a second convention.

    Args:
        kpoints: The kpoints input from KoopmansInput.

    Returns:
        AiiDA KpointsData node with k-point mesh.
    """
    kpts = orm.KpointsData()
    kpts.set_kpoints_mesh(list(kpoints.grid), offset=list(kpoints.offset))  # type: ignore[no-untyped-call]
    return kpts


def step_kpoints_mesh(kpoints: KpointsInput, step: str) -> orm.KpointsData:
    """Convert the mesh the named step samples to AiiDA KpointsData.

    ``kpoints.overrides.<step>`` states the mesh for that step alone; every
    attribute it leaves unset comes from the top-level ``grid`` and
    ``offset``.

    Args:
        kpoints: The kpoints input from KoopmansInput.
        step: Name of a ``kpoints.overrides`` entry.

    Returns:
        AiiDA KpointsData node with k-point mesh.

    Raises:
        ValueError: If the step's entry states a ``grid_spacing`` instead of
            a mesh.
    """
    override = getattr(kpoints.overrides, step)
    if override is None:
        return kpoints_input_to_kpoints_mesh(kpoints)
    if override.grid_spacing is not None:
        raise ValueError(
            f"`kpoints.overrides.{step}.grid_spacing` leaves the mesh to the cell, so "
            f"the {step} mesh is not known before the calculation runs. Write "
            f"`kpoints.overrides.{step}.grid` instead."
        )
    kpts = orm.KpointsData()
    grid = kpoints.grid if override.grid is None else override.grid
    offset = kpoints.offset if override.offset is None else override.offset
    kpts.set_kpoints_mesh(list(grid), offset=list(offset))  # type: ignore[no-untyped-call]
    return kpts


def step_grid_spacing(kpoints: KpointsInput, step: str) -> float | None:
    """Return the largest k-point spacing the named step samples at, if it states one.

    ``None`` where the step states a mesh instead, which is every step the
    input file does not give a ``grid_spacing``.

    Args:
        kpoints: The kpoints input from KoopmansInput.
        step: Name of a ``kpoints.overrides`` entry.
    """
    override = getattr(kpoints.overrides, step)
    return None if override is None else override.grid_spacing


def wannier90_path_density(kpoints: KpointsInput) -> float:
    """Return the density wannier90 interpolates its band structure at.

    Falls back to :class:`~koopmans.input_file.WannierKpointsOverridesInput`'s
    own default where ``kpoints.overrides.wannier90`` is unset.

    Args:
        kpoints: The kpoints input from KoopmansInput.
    """
    from koopmans.input_file import WannierKpointsOverridesInput

    override = kpoints.overrides.wannier90
    return (override or WannierKpointsOverridesInput()).path_density


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
    reciprocal_cell: Any,
) -> tuple[list[list[float]], list[tuple[int, str]]]:
    """Calculate k-points along a path with the specified density.

    Args:
        path: List of (start_label, end_label) tuples defining path segments.
        point_coords: Dict mapping special point labels to their crystal
            (fractional reciprocal) coordinates.
        density: Number of k-points per inverse angstrom, in the same 2π
            convention as ``reciprocal_cell`` (and as ``kpoints.grid_spacing``).
        reciprocal_cell: The cell's reciprocal lattice vectors as rows, in
            1/angstrom, 2π convention (``aiida.orm.KpointsData.reciprocal_cell``).
            Segment lengths are measured in this Cartesian basis, so a
            converged density carries between structures — crystal
            coordinates alone say nothing about physical length.

    Returns:
        Tuple of (kpoint_list, label_list) where kpoint_list contains crystal
        coordinates and label_list contains (index, label) tuples for special
        points.
    """
    import numpy as np

    kpoint_list: list[list[float]] = []
    label_list: list[tuple[int, str]] = []

    previous_end: str | None = None
    for start_label, end_label in path:
        start_coord = np.array(point_coords[start_label])
        end_coord = np.array(point_coords[end_label])

        segment_length = np.linalg.norm((end_coord - start_coord) @ reciprocal_cell)
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
    density: float | None = None,
) -> orm.KpointsData:
    """Convert KpointsInput to AiiDA KpointsData for bands calculations.

    An explicit path in the input is resolved against the cell's own
    Bravais-lattice special points (the legacy ASE vocabulary); when no path
    is given one is generated automatically with seekpath.

    The returned node carries the structure's cell and pbc alongside the
    k-points and their labels.

    Args:
        kpoints: The kpoints input from KoopmansInput.
        structure: The structure to generate k-path for.
        density: Points per inverse angstrom to sample the path at. Defaults
            to ``kpoints.path_density``; a caller wiring a different step's
            interpolation density passes its own value instead.

    Returns:
        AiiDA KpointsData node with k-point path.
    """
    import numpy as np

    kpts = orm.KpointsData()
    # The cell fixes the reciprocal basis both the special points below and
    # the segment-length measurement are expressed in. Set before either, and
    # before the k-points, which are validated against it.
    kpts.set_cell_from_structure(structure)  # type: ignore[no-untyped-call]

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

    kpoint_list, label_list = _calculate_kpoints_along_path(
        path,
        point_coords,
        kpoints.path_density if density is None else density,
        kpts.reciprocal_cell,  # 2π convention, shared with grid_spacing by construction
    )

    kpts.set_kpoints(kpoint_list)  # type: ignore[no-untyped-call]
    kpts.labels = label_list

    return kpts


def kpoints_input_to_interpolation_path(
    kpoints: KpointsInput,
    structure: orm.StructureData,
    density: float | None = None,
) -> orm.KpointsData | None:
    """Return the input's k-path as a labelled explicit k-list, or ``None``.

    ``None`` unless the input names a path with a segment to interpolate
    along (:func:`koopmans.input_file.names_band_path`, the same predicate
    the input file's own band-path refusals read). Otherwise defers to
    :func:`kpoints_input_to_kpoints_path`. Callers use this to decide whether
    a step gets an explicit bands path or is left on its protocol default.

    Args:
        kpoints: The kpoints input from KoopmansInput.
        structure: The structure to generate the k-path for.
        density: Points per inverse angstrom to sample the path at. Defaults
            to ``kpoints.path_density``.

    Returns:
        AiiDA KpointsData node with the k-point path, or ``None``.
    """
    from koopmans.input_file import names_band_path

    if not names_band_path(kpoints):
        return None
    return kpoints_input_to_kpoints_path(kpoints, structure, density)


def _resolve_pw_cutoffs(system: dict[str, Any]) -> None:
    """Set ``ecutrho`` in place, at :data:`NORM_CONSERVING_DUAL` times ``ecutwfc``.

    With no ``ecutwfc`` stated the pair is left empty, for the
    pseudopotential family to recommend.
    """
    ecutwfc = system.get("ecutwfc")
    if ecutwfc is not None:
        system["ecutrho"] = NORM_CONSERVING_DUAL * ecutwfc


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

    # Add nbnd if specified
    if calc_params.nbnd is not None:
        parameters["SYSTEM"]["nbnd"] = int(calc_params.nbnd)

    # Merge with explicit PW parameters from input
    if pw_params.control:
        parameters["CONTROL"].update(
            pw_params.control.model_dump(exclude_none=True, exclude_unset=True)
        )
    if pw_params.system:
        parameters["SYSTEM"].update(
            pw_params.system.model_dump(exclude_none=True, exclude_unset=True)
        )
    if pw_params.electrons:
        parameters["ELECTRONS"].update(
            pw_params.electrons.model_dump(exclude_none=True, exclude_unset=True)
        )

    if (
        parameters["SYSTEM"].get("occupations") == "smearing"
        and "degauss" not in parameters["SYSTEM"]
    ):
        raise ValueError(
            "`calculator_parameters.pw.system.occupations = 'smearing'` needs "
            "`calculator_parameters.pw.system.degauss` set explicitly too. Every "
            "koopmans route runs pw.x with fixed occupations by default, which "
            "clears the protocol's own smearing keywords before this override "
            "lands, so pw.x would abort asking for a broadening value. Set "
            "`calculator_parameters.pw.system.degauss` (Ry)."
        )

    _resolve_pw_cutoffs(parameters["SYSTEM"])

    # Ensure all Path objects are converted to strings for JSON serialization
    parameters = _convert_paths_to_strings(parameters)

    return parameters


def input_to_kcw_overrides(koopmans_input: KoopmansInput) -> dict[str, dict[str, Any]]:
    """Convert ``calculator_parameters.kcw`` into a kcw.x namelist-overrides dict.

    One entry per namelist (``control``, ``wannier``, ``screen``, ``ham``),
    present only when the user set at least one of its keywords: the DFPT
    route's own values stand wherever the user is silent. A keyword written
    as ``null`` means the same as one left out — kcw.x has no third state —
    so it is dropped too rather than blanking the route's value.
    Route-owned keys are refused at parse time
    (``koopmans.input_file.kcw``) and never reach here.
    """
    kcw = koopmans_input.calculator_parameters.kcw
    overrides: dict[str, dict[str, Any]] = {}
    for name, namelist in (
        ("control", kcw.control),
        ("wannier", kcw.wannier),
        ("screen", kcw.screen),
        ("ham", kcw.ham),
    ):
        dumped = namelist.model_dump(exclude_unset=True, exclude_none=True)
        if dumped:
            overrides[name] = _convert_paths_to_strings(dumped)
    return overrides


def input_to_ph_parameters(koopmans_input: KoopmansInput) -> dict[str, dict[str, Any]]:
    """Convert ``calculator_parameters.ph`` into a ph.x ``INPUTPH`` namelist dict.

    The dielectric-constant (dft_eps) route merges this underneath its own
    ``epsil``/``trans``/q-mesh keys, so every key present here is a plain
    user override (route-owned keys are rejected at parse time; see
    ``koopmans.input_file.ph``).
    """
    parameters: dict[str, dict[str, Any]] = {
        "INPUTPH": koopmans_input.calculator_parameters.ph.model_dump(
            exclude_none=True, exclude_defaults=True
        ),
    }
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
