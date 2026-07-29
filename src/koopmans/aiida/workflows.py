"""Workflow building logic for koopmans AiiDA integration.

This module handles selecting and constructing the appropriate AiiDA workgraph
based on the task specified in a KoopmansInput.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

from aiida import orm
from aiida_koopmans.workgraphs import Codes
from aiida_quantumespresso.common.types import SpinType

from koopmans.aiida.conversion import (
    atoms_input_to_structure,
    atoms_input_to_structures,
    code_parallelization,
    input_to_pw_parameters,
)
from koopmans.input_file.workflow import (
    CalculateScreeningMethod,
    Correction,
    GroupOrbitalsBy,
    Task,
    VariationalOrbitalType,
)

if TYPE_CHECKING:
    from aiida_koopmans.types import AutomaticProjectionBlock
    from aiida_koopmans.workgraphs.block_wannierize import WannierizeOverrides
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput
    from koopmans.input_file.workflow import WorkflowConfig


def _load_code(name: str, executable: str) -> orm.AbstractCode:
    """Load the code labelled ``<name>@localhost``, with a setup hint on failure."""
    try:
        return orm.load_code(f"{name}@localhost")
    except Exception as exc:
        raise ValueError(
            f"Could not load {executable} code: {exc}\n"
            "Please run 'koopmans install' first to set up the AiiDA backend."
        ) from exc


def load_codes_for_task(workflow: WorkflowConfig) -> Codes:
    """Load the AiiDA codes required by the workflow described in ``workflow``.

    Which codes are needed depends not only on ``task`` but also on the
    Koopmans correction (``ki`` vs ``none`` vs …) and the screening method
    (``dscf`` needs kcp.x, ``dfpt`` would need kcw.x, etc.).

    Args:
        workflow: The ``WorkflowConfig`` block from a parsed ``KoopmansInput``.

    Returns:
        Dictionary mapping code names to Code instances.

    Raises:
        ValueError: If a required code is not found in the AiiDA profile.
        NotImplementedError: If the requested code combination is not supported yet.
    """
    task = workflow.task
    codes: Codes = {}

    # All tasks need pw.x
    codes["pw"] = _load_code("pw", "pw.x")

    # A corrected singlepoint — or a trajectory, which runs one DSCF
    # singlepoint per snapshot — needs a screening-method-specific code
    # regardless of ``calculate_alpha``: when alphas are guessed instead
    # of computed, kcp.x/kcw.x still evaluate the corrected functional — only
    # the screening step itself is skipped.
    if task in (Task.SINGLEPOINT, Task.TRAJECTORY) and workflow.correction != Correction.NONE:
        if workflow.screening_method == CalculateScreeningMethod.DSCF:
            codes["kcp"] = _load_code("kcp", "kcp.x")
        elif workflow.screening_method == CalculateScreeningMethod.DFPT:
            # kcw.x runs all three DFPT steps (wann2kc, screen, ham) selected
            # via its ``control.calculation`` flag, so a single code suffices.
            codes["kcw"] = _load_code("kcw", "kcw.x")

    # The dielectric-constant task runs ph.x on top of the scf
    if task == Task.DFT_EPS:
        codes["ph"] = _load_code("ph", "ph.x")

    # Wannierize task needs additional codes
    if task == Task.WANNIERIZE:
        codes["pw2wannier90"] = _load_code("pw2wannier90", "pw2wannier90.x")
        codes["wannier90"] = _load_code("wannier90", "wannier90.x")

        # Automated block splitting runs the Wannier.jl CalcJobs (the julia
        # binary registered via aiida_wannierjl.helpers.get_wannierjl_code).
        if workflow.block_wannierization_threshold is not None:
            codes["wannierjl"] = _load_code("wannierjl", "julia (Wannier.jl)")

        # projwfc is only needed when the Wannierize flow computes a projected
        # DOS / bandstructure, so treat it as optional rather than required.
        try:
            codes["projwfc"] = orm.load_code("projwfc@localhost")
        except Exception:  # noqa: S110
            pass

    return codes


def _prepare_common_inputs(
    koopmans_input: KoopmansInput,
    override_keys: list[str],
) -> tuple[orm.StructureData, str, dict[str, Any]]:
    """Prepare the common inputs shared by all workgraph builders.

    Converts the koopmans input into a structure, ensures the pseudo family is
    installed, and builds an overrides dict with a PW parameters entry for each
    of the requested sub-workflow keys.

    Args:
        koopmans_input: The parsed koopmans input.
        override_keys: Sub-workflow keys to include in overrides (e.g. ["scf", "bands"]).

    Returns:
        Tuple of (structure, pseudo_family, overrides).
    """
    from koopmans.aiida.setup.pseudos import ensure_pseudo_family_installed

    structure = atoms_input_to_structure(koopmans_input.atoms)
    parameters = input_to_pw_parameters(koopmans_input)
    pseudo_family = koopmans_input.workflow.pseudo_library

    ensure_pseudo_family_installed(pseudo_family)

    pw_overrides: dict[str, Any] = {"parameters": parameters}
    # The pw entry carries the pw.x parallelization directive: -npool rides
    # settings.cmdline; ntasks rides metadata.options.resources — both survive
    # get_builder_from_protocol's override merge (verified by eager build).
    # Seeding the shared scf/nscf/bands overrides here covers the primary pw.x
    # steps; the full per-code mapping is threaded to every graph builder too,
    # so pw.x steps assembled inside the graphs (e.g. the dielectric scf) pick
    # up the same directive.
    options, settings = code_parallelization(koopmans_input.parallelization.pw)
    if settings:
        pw_overrides["settings"] = settings
    if options:
        pw_overrides["metadata"] = {"options": options}

    overrides: dict[str, Any] = {
        key: {
            "pseudo_family": pseudo_family,
            "pw": dict(pw_overrides),
        }
        for key in override_keys
    }

    return structure, pseudo_family, overrides


def build_workgraph(koopmans_input: KoopmansInput) -> WorkGraph:
    """Build the appropriate workgraph for a KoopmansInput.

    Args:
        koopmans_input: The parsed koopmans input.

    Returns:
        A WorkGraph instance ready to be submitted.

    Raises:
        ValueError: If the task is not supported or required codes are missing.
    """
    task = koopmans_input.workflow.task

    # Load required codes
    codes = load_codes_for_task(koopmans_input.workflow)

    # Build the workgraph based on task
    if task == Task.DFT_BANDS:
        return _build_dft_bands_workgraph(koopmans_input, codes)
    elif task == Task.WANNIERIZE:
        return _build_wannierize_workgraph(koopmans_input, codes)
    elif task == Task.SINGLEPOINT:
        return _build_singlepoint_workgraph(koopmans_input, codes)
    elif task == Task.TRAJECTORY:
        return _build_trajectory_workgraph(koopmans_input, codes)
    elif task == Task.DFT_EPS:
        return _build_dft_eps_workgraph(koopmans_input, codes)
    else:
        raise ValueError(
            f"Task '{task.value}' is not yet implemented. "
            f"Supported tasks: {Task.DFT_BANDS.value}, {Task.WANNIERIZE.value}, "
            f"{Task.SINGLEPOINT.value}, {Task.TRAJECTORY.value}, {Task.DFT_EPS.value}"
        )


def _build_dft_bands_workgraph(
    koopmans_input: KoopmansInput,
    codes: Codes,
) -> WorkGraph:
    """Build a workgraph for DFT bands calculation.

    Args:
        koopmans_input: The parsed koopmans input.
        codes: Dictionary of loaded codes.

    Returns:
        A WorkGraph for PwBandsWorkChain.
    """
    from aiida_koopmans.workgraphs.pw import RunPwBands

    structure, _pseudo_family, overrides = _prepare_common_inputs(koopmans_input, ["scf", "bands"])

    return RunPwBands.build(
        code=codes["pw"],
        structure=structure,
        overrides=overrides,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
    )


def _build_dft_eps_workgraph(
    koopmans_input: KoopmansInput,
    codes: dict[str, orm.AbstractCode],
) -> WorkGraph:
    """Build a workgraph for the dielectric-constant (ph.x) task.

    Port of the legacy ``DFTPhWorkflow`` (``workflows/_dft.py``): one scf,
    then ph.x with ``epsil = .true.`` / ``trans = .false.`` at q = Gamma,
    exposing the isotropic average of the dielectric tensor as ``eps_inf``.
    The legacy scf passes ``nbnd=None`` (no empty bands are needed for a
    ground-state response), so ``nbnd`` is stripped from the PW overrides.

    Args:
        koopmans_input: The parsed koopmans input.
        codes: Dictionary of loaded codes.

    Returns:
        A WorkGraph chaining PwBaseWorkChain into PhBaseWorkChain.
    """
    from aiida_koopmans.workgraphs.ph import DielectricTask

    structure, pseudo_family, overrides = _prepare_common_inputs(koopmans_input, ["scf"])
    overrides["scf"]["pw"]["parameters"].get("SYSTEM", {}).pop("nbnd", None)

    return DielectricTask.build(
        pw_code=codes["pw"],
        ph_code=codes["ph"],
        structure=structure,
        pseudo_family=pseudo_family,
        overrides=overrides,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
    )


def _build_wannierize_workgraph(
    koopmans_input: KoopmansInput,
    codes: Codes,
) -> WorkGraph:
    """Build a workgraph for Wannierization.

    Args:
        koopmans_input: The parsed koopmans input.
        codes: Dictionary of loaded codes.

    Returns:
        A WorkGraph for Wannier90WorkChain, or the automated block-splitting
        flow when ``block_wannierization_threshold`` is set.
    """
    from aiida_koopmans.workgraphs.wannier90 import Wannierize
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    if koopmans_input.workflow.block_wannierization_threshold is not None:
        return _build_wannierize_split_workgraph(koopmans_input, codes)

    structure, pseudo_family, overrides = _prepare_common_inputs(koopmans_input, ["scf", "nscf"])

    # Check if external projectors are requested
    pw2w_params = koopmans_input.calculator_parameters.pw2wannier90
    extra_kwargs: dict[str, Any] = {}
    if pw2w_params.atom_proj_ext:
        external_projectors, projector_path = _load_external_projectors(
            structure, pw2w_params.atom_proj_dir
        )
        extra_kwargs["projection_type"] = WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL
        extra_kwargs["external_projectors_path"] = projector_path
        extra_kwargs["external_projectors"] = external_projectors

    return Wannierize.build(
        codes=codes,
        structure=structure,
        overrides=overrides,
        pseudo_family=pseudo_family,
        print_summary=False,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        **extra_kwargs,
    )


def _derive_wannierize_blocks(
    structure: orm.StructureData,
    projection_blocks: list[list[Any]],
    nbnd: int,
) -> list[Any]:
    """Turn user projection blocks into wannierization blocks for the split flow.

    Unlike ``_derive_dscf_blocks`` there are no straddle or occupied-coverage
    constraints: a block that mixes occupied and empty bands — or spans an
    internal gap — is exactly what the automated splitting handles. Blocks
    cover consecutive bands in input order; the last block absorbs the
    remaining ``nbnd - cursor`` bands as its disentanglement pool.
    """
    from aiida_koopmans.projections import (
        band_range_complement,
        projection_num_wann,
        projection_win_string,
    )
    from aiida_koopmans.types import ExplicitProjectionBlock, SpinChannel
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    blocks: list[Any] = []
    cursor = 0
    for i, block in enumerate(projection_blocks):
        num_wann = sum(projection_num_wann(structure, p) for p in block)
        start, end = cursor + 1, cursor + num_wann
        if end > nbnd:
            raise ValueError(f"The projection blocks span {end} bands but nbnd = {nbnd}.")
        blocks.append(
            ExplicitProjectionBlock(
                label=f"block_{i + 1}",
                spin=SpinChannel.NONE,
                num_wann=num_wann,
                num_bands=num_wann,
                include_bands=list(range(start, end + 1)),
                exclude_bands=band_range_complement(start, end, nbnd),
                projection_type=WannierProjectionType.ANALYTIC,
                projections=[projection_win_string(p) for p in block],
            )
        )
        cursor = end

    if blocks and cursor < nbnd:
        last = blocks[-1]
        last["num_bands"] = last["num_wann"] + (nbnd - cursor)
        start = last["include_bands"][0]
        last["include_bands"] = list(range(start, nbnd + 1))
        last["exclude_bands"] = list(range(1, start)) or None

    return blocks


def _pseudo_is_fully_relativistic(kind: str, upf: orm.UpfData) -> bool:
    """Sniff the ``has_so`` flag of a pseudo's UPF header.

    Upstream's ``is_soc_pseudo`` trips over a UPF v2 header that omits
    ``has_so`` (a bare ``TypeError``); real generators always write the
    flag, so convert that case into an error naming the pseudo. Old
    attribute-less v1 headers parse as scalar-relativistic upstream.
    """
    from aiida_wannier90_workflows.utils.pseudo.upf import get_upf_content, is_soc_pseudo

    try:
        return bool(is_soc_pseudo(get_upf_content(upf)))
    except TypeError as exc:
        raise ValueError(
            f"The pseudopotential for {kind} does not declare `has_so` in its UPF header, "
            "so whether it is fully relativistic cannot be determined (scalar-relativistic "
            'UPF files normally carry `has_so="F"`).'
        ) from exc


def _derive_automatic_wannierize_blocks(
    structure: orm.StructureData,
    pseudos: dict[str, orm.UpfData],
    nbnd: int | None,
    num_occ_bands: int,
) -> tuple[list[AutomaticProjectionBlock], int]:
    """Derive the wannierization blocks when no explicit projections are given.

    The whole manifold becomes a single automatic block seeded from the
    pseudopotentials' atomic projectors (pw2wannier90 ``atom_proj``); the
    runtime band-group detection decides how it splits. ``num_wann`` is
    fixed by the projector count of the pseudos — the width of the amn
    matrix pw2wannier90 writes — and the block carries no disentanglement
    pool: the detected groups cover only the Wannierised manifold, so a
    block with bands above it cannot be split. Returns the single-block
    list and the band count the nscf must cover.
    """
    from aiida_wannier90_workflows.common.types import WannierProjectionType
    from aiida_wannier90_workflows.utils.pseudo import get_number_of_projections

    fully_relativistic = sorted(
        kind for kind, upf in pseudos.items() if _pseudo_is_fully_relativistic(kind, upf)
    )
    if fully_relativistic:
        raise NotImplementedError(
            f"The pseudopotentials for {', '.join(fully_relativistic)} are fully relativistic; "
            "automatic projections support scalar-relativistic pseudopotentials only (the "
            "split route runs spin='none'). Provide explicit projections in "
            "`calculator_parameters.w90.projections` or use a scalar-relativistic family."
        )
    # Scalar-relativistic guaranteed by the guard above, so the projector count
    # is exact with the SOC flag pinned off.
    num_wann = get_number_of_projections(
        structure=structure, pseudos=pseudos, spin_non_collinear=False, spin_orbit_coupling=False
    )
    return _validated_single_automatic_block(
        num_wann,
        nbnd,
        num_occ_bands,
        WannierProjectionType.ATOMIC_PROJECTORS_QE,
        "the pseudopotentials",
    )


def _validated_single_automatic_block(
    num_wann: int,
    nbnd: int | None,
    num_occ_bands: int,
    projection_type: Any,
    source: str,
) -> tuple[list[AutomaticProjectionBlock], int]:
    """Build the single whole-manifold automatic block, validating its size.

    Shared by the projector sources that span the manifold with one
    automatic block (pseudopotential projectors, external projector files);
    ``source`` names the projector origin in the error messages. The block
    carries no disentanglement pool: the detected groups cover only the
    Wannierised manifold, so a block with bands above it cannot be split.
    """
    from aiida_koopmans.types import AutomaticProjectionBlock, SpinChannel

    if num_wann < num_occ_bands:
        raise ValueError(
            f"{source[0].upper()}{source[1:]} provide {num_wann} atomic projectors but the "
            f"system has {num_occ_bands} occupied bands, so automatic projections cannot "
            "span the occupied manifold. Provide explicit projections in "
            "`calculator_parameters.w90.projections`."
        )
    if nbnd is not None and nbnd < num_wann:
        raise ValueError(
            f"nbnd = {nbnd} is smaller than the {num_wann} atomic projectors of "
            f"{source}; automatic projections need one band per projector."
        )
    if nbnd is not None and nbnd > num_wann:
        raise NotImplementedError(
            f"nbnd = {nbnd} exceeds the {num_wann} atomic projectors of "
            f"{source}, which would disentangle the automatic block; splitting a "
            "disentangled block is not supported. Drop nbnd or provide explicit "
            "projections in `calculator_parameters.w90.projections`."
        )
    block = AutomaticProjectionBlock(
        label="block_1",
        spin=SpinChannel.NONE,
        num_wann=num_wann,
        num_bands=num_wann,
        include_bands=list(range(1, num_wann + 1)),
        exclude_bands=None,
        projection_type=projection_type,
    )
    return [block], num_wann


def _load_external_projectors(
    structure: orm.StructureData, proj_dir: Path | None
) -> tuple[dict[str, Any], str]:
    """Load the per-element orbital tables of an external projector directory.

    The directory follows aiida-wannier90-workflows' layout: one
    ``<element>.dat`` radial-projector file per element (the filename
    pw2wannier90 stages and reads) plus a ``projectors.json`` holding each
    element's orbital entries (``label`` / ``l`` / ``alpha`` per
    projector), which size the Wannier manifold and select the
    Lowdin-frozen projectors. ``projectors.json`` is part of upstream's
    external-projector contract and is required here: the projector counts
    could be rebuilt from the ``.dat`` files, but the ``alpha``
    (frozen-projector selection) and ``j`` (spin-orbit) metadata could
    not.

    The directory is validated on the local filesystem, so it must live on
    the machine building the workflow; projector directories that exist
    only on a remote computer are not supported yet.

    Returns the tables and the resolved directory path.
    """
    import json

    if proj_dir is None:
        raise ValueError(
            "`pw2wannier90.atom_proj_dir` must be set when `pw2wannier90.atom_proj_ext` "
            "is true: it locates the external projector files."
        )
    directory = Path(proj_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(
            f"`pw2wannier90.atom_proj_dir` does not exist on this machine: {directory}. "
            "The projector directory must be present on the machine building the "
            "workflow; projector directories that exist only on a remote computer "
            "are not supported yet."
        )
    table_file = directory / "projectors.json"
    if not table_file.is_file():
        raise ValueError(
            f"{directory} contains no `projectors.json`; an external projector "
            "directory follows aiida-wannier90-workflows' layout — one "
            "`<element>.dat` per element plus a `projectors.json` mapping each "
            "element to its orbital entries (`label` / `l` / `alpha`), as written "
            "by that package's `dev/projectors` generation script."
        )
    external_projectors = json.loads(table_file.read_text())
    elements = sorted(
        {structure.get_kind(site.kind_name).symbol for site in structure.sites}  # type: ignore[no-untyped-call]
    )
    missing_files = [
        f"{element}.dat" for element in elements if not (directory / f"{element}.dat").is_file()
    ]
    if missing_files:
        raise ValueError(
            f"{directory} is missing the projector files {missing_files}; "
            "pw2wannier90 reads one `<element>.dat` per element of the structure."
        )
    missing_tables = [element for element in elements if element not in external_projectors]
    if missing_tables:
        raise ValueError(f"`projectors.json` in {directory} has no entry for {missing_tables}.")
    return external_projectors, str(directory)


def _derive_external_wannierize_blocks(
    structure: orm.StructureData,
    external_projectors: dict[str, Any],
    nbnd: int | None,
    num_occ_bands: int,
) -> tuple[list[AutomaticProjectionBlock], int]:
    """Derive the wannierization blocks from external projector tables.

    The external-projector analogue of
    :func:`_derive_automatic_wannierize_blocks`: the whole manifold becomes
    a single automatic block (pw2wannier90 ``atom_proj_ext``) whose
    ``num_wann`` is the projector count of the orbital tables, under the
    same no-pool constraints. The upstream counter also rejects
    spin-orbit-coupled (``j``-resolved) projector tables, which the
    spin='none' split route cannot consume.
    """
    from aiida_wannier90_workflows.common.types import WannierProjectionType
    from aiida_wannier90_workflows.utils.pseudo import get_number_of_projections_ext

    num_wann = get_number_of_projections_ext(
        structure=structure,
        external_projectors=external_projectors,
        spin_non_collinear=False,
        spin_orbit_coupling=False,
    )
    return _validated_single_automatic_block(
        num_wann,
        nbnd,
        num_occ_bands,
        WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
        "the external projector files",
    )


def _reject_unwired_external_projectors(koopmans_input: KoopmansInput, route: str) -> None:
    """Reject ``atom_proj_ext`` on a route that does not consume it.

    The singlepoint and trajectory routes build their Wannierizations
    without consulting the external projector keywords, so accepting the
    switch there would silently drop it.
    """
    if koopmans_input.calculator_parameters.pw2wannier90.atom_proj_ext:
        raise NotImplementedError(
            f"`pw2wannier90.atom_proj_ext` is not wired into the {route} route; "
            "external projectors are currently supported by the `wannierize` task only."
        )


def _build_wannierize_split_workgraph(
    koopmans_input: KoopmansInput,
    codes: Codes,
) -> WorkGraph:
    """Build the Wannierization workgraph with automated block splitting.

    A pw.x bands run along the k-path feeds a runtime band-group detection
    (splitting at every gap wider than ``block_wannierization_threshold`` eV
    and at the occupied/empty boundary), and each projection block whose
    bands fall into several groups is Wannierised once, split with Wannier.jl
    parallel transport, re-Wannierised group by group and its products merged
    back together.

    With explicit projections in ``calculator_parameters.w90.projections``
    each user block becomes a wannierization block. Without any, a single
    atomic-projector block spans the whole manifold and the runtime
    detection decides how it splits; the projectors come from the
    pseudopotentials (:func:`_derive_automatic_wannierize_blocks`) or, with
    ``pw2wannier90.atom_proj_ext``, from the external projector directory
    ``pw2wannier90.atom_proj_dir``
    (:func:`_derive_external_wannierize_blocks`).

    Current scope: ``spin = 'none'``.
    """
    from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlocks
    from aiida_wannier90_workflows.utils.kpoints import get_explicit_kpoints

    from koopmans.aiida.conversion import (
        get_pseudos_from_family,
        kpoints_input_to_kpoints_mesh,
        kpoints_input_to_kpoints_path,
    )
    from koopmans.aiida.setup.pseudos import ensure_pseudo_family_installed

    workflow = koopmans_input.workflow
    calc_params = koopmans_input.calculator_parameters

    threshold = workflow.block_wannierization_threshold
    if threshold is None:
        raise ValueError(
            "The block-splitting Wannierize builder requires "
            "`block_wannierization_threshold` to be set."
        )
    if workflow.spin != SpinType.NONE:
        raise NotImplementedError(
            "block_wannierization_threshold currently supports spin='none' only "
            "(the group detection and per-block split are single-channel)."
        )
    projections = calc_params.wannier90.projections
    if koopmans_input.kpoints.path is None:
        raise ValueError(
            "block_wannierization_threshold needs a k-point path: the band-group "
            "detection reads the eigenvalues of a bands run along it."
        )

    structure = atoms_input_to_structure(koopmans_input.atoms)
    pseudo_family = workflow.pseudo_library
    ensure_pseudo_family_installed(pseudo_family)

    pseudos = get_pseudos_from_family(pseudo_family, structure)
    nelec = round(sum(pseudos[site.kind_name].z_valence for site in structure.sites))
    if nelec % 2:
        raise NotImplementedError(
            f"Odd electron count ({nelec}) requires spin='collinear', which the "
            "block-splitting flow does not support yet."
        )
    num_occ_bands = nelec // 2

    nbnd = calc_params.nbnd if calc_params.nbnd is not None else calc_params.pw.system.nbnd
    nbnd = int(nbnd) if nbnd is not None else None
    external_kwargs: dict[str, Any] = {}
    if projections:
        if calc_params.pw2wannier90.atom_proj_ext:
            raise ValueError(
                "Explicit projections in `calculator_parameters.w90.projections` and "
                "`pw2wannier90.atom_proj_ext` were both given; the explicit projections "
                "define every block, so the external projectors would be silently "
                "ignored. Drop one of the two."
            )
        if nbnd is None:
            nbnd = _num_wann_total(structure, projections)
        blocks = _derive_wannierize_blocks(structure, projections, nbnd)
    elif calc_params.pw2wannier90.atom_proj_ext:
        external_projectors, projector_path = _load_external_projectors(
            structure, calc_params.pw2wannier90.atom_proj_dir
        )
        blocks, nbnd = _derive_external_wannierize_blocks(
            structure, external_projectors, nbnd, num_occ_bands
        )
        external_kwargs = {
            "external_projectors_path": projector_path,
            "external_projectors": external_projectors,
        }
    else:
        blocks, nbnd = _derive_automatic_wannierize_blocks(structure, pseudos, nbnd, num_occ_bands)

    # The scf needs only the occupied bands, so nbnd is dropped from its
    # override; the nscf — and the bands run seeded from its overrides —
    # must cover every Wannierised band.
    parameters = input_to_pw_parameters(koopmans_input)
    scf_parameters = copy.deepcopy(parameters)
    scf_parameters.get("SYSTEM", {}).pop("nbnd", None)
    nscf_parameters = copy.deepcopy(parameters)
    nscf_parameters.setdefault("SYSTEM", {})["nbnd"] = nbnd
    wannier_overrides: WannierizeOverrides = {
        "scf": {"pseudo_family": pseudo_family, "pw": {"parameters": scf_parameters}},
        "nscf": {"pseudo_family": pseudo_family, "pw": {"parameters": nscf_parameters}},
    }

    # User wannier90 keywords (disentanglement windows, iteration counts, ...)
    # feed every per-block wannierisation; flat by design (see
    # ``WannierizeOverrides``).
    w90_user = calc_params.wannier90.model_dump(
        exclude_unset=True, exclude={"projections", "up", "down"}
    )
    if w90_user:
        wannier_overrides["wannier90"] = w90_user

    # wannier90 / pw2wannier90 need eigenstates on the full explicit k-list
    # (wannier90 kmesh.pl ordering, no symmetry reduction) and cannot
    # re-derive the Monkhorst-Pack dimensions from it, so expand the mesh
    # here and carry the grid separately.
    kmesh = kpoints_input_to_kpoints_mesh(koopmans_input.kpoints)
    mp_grid = [int(x) for x in kmesh.get_kpoints_mesh()[0]]  # type: ignore[no-untyped-call]

    return WannierizeBlocks.build(
        codes=codes,
        structure=structure,
        blocks=blocks,
        kpoints=get_explicit_kpoints(kmesh),
        mp_grid=mp_grid,
        bands_kpoints=kpoints_input_to_kpoints_path(koopmans_input.kpoints, structure),
        num_occ_bands=num_occ_bands,
        split_threshold=float(threshold),
        pseudo_family=pseudo_family,
        overrides=wannier_overrides,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        **external_kwargs,
    )


def _num_wann_total(structure: orm.StructureData, projection_blocks: list[list[Any]]) -> int:
    """Total Wannier-function count of a set of user projection blocks."""
    from aiida_koopmans.projections import projection_num_wann

    return sum(projection_num_wann(structure, p) for block in projection_blocks for p in block)


def _build_singlepoint_workgraph(
    koopmans_input: KoopmansInput,
    codes: Codes,
) -> WorkGraph:
    """Build a workgraph for a singlepoint Koopmans calculation.

    Dispatches on ``workflow.screening_method`` first (DSCF vs DFPT), then on
    ``workflow.correction``:

    - DSCF + ``KI``/``KIPZ`` → ``KoopmansDSCFWorkflow`` (kcp.x)
    - DFPT + ``KI`` → ``_build_singlepoint_dfpt_workgraph`` (kcw.x; KI only)
    - anything else → ``NotImplementedError``
    """
    from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow

    from koopmans.aiida.setup.pseudos import ensure_pseudo_family_installed

    workflow = koopmans_input.workflow

    _reject_unwired_external_projectors(koopmans_input, "singlepoint")

    # DFPT routes on the screening method alone: calculate_alpha = False is
    # the alpha_guess path inside the DFPT builder (screen step skipped),
    # not a reason to fall through to the kcp.x/DSCF branch.
    if workflow.screening_method == CalculateScreeningMethod.DFPT:
        return _build_singlepoint_dfpt_workgraph(koopmans_input, codes)

    _require_supported_correction(workflow.correction)

    if workflow.spin in (SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT):
        raise NotImplementedError(
            f"spin={workflow.spin.value!r} is not supported by the DSCF (kcp.x) stream: "
            "kcp.x has no noncollinear mode. Use screening_method='dfpt'."
        )

    structure = atoms_input_to_structure(koopmans_input.atoms)
    ensure_pseudo_family_installed(workflow.pseudo_library)

    inputs = _kcp_dscf_inputs(koopmans_input)

    extra_kwargs: dict[str, Any] = {}
    if workflow.init_orbitals in (
        VariationalOrbitalType.MLWFS,
        VariationalOrbitalType.PROJWFS,
    ):
        extra_kwargs = _dscf_wannier_init_inputs(koopmans_input, structure, codes, inputs["nbnd"])

    return KoopmansDSCFWorkflow.build(
        code=codes["kcp"],
        structure=structure,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        **inputs,
        **extra_kwargs,
    )


def _derive_dscf_blocks(
    structure: orm.StructureData,
    projection_blocks: list[list[Any]],
    nocc: int,
    nbnd: int,
    spin_channel: Any,
) -> list[Any]:
    """Turn user projection blocks into DSCF wannierization blocks.

    The DSCF route wannierises every user block separately and merges them
    per (filling, spin) via merge_evc.x, so any number of blocks is allowed.
    Each block covers ``num_wann`` consecutive bands; a block straddling the
    occupied/empty boundary is an input error, and the occupied blocks must
    cover every occupied band (the folded ``evc_occupied`` files seed the
    complete occupied manifold of the supercell kcp.x run).
    """
    from aiida_koopmans.projections import (
        band_range_complement,
        projection_num_wann,
        projection_win_string,
    )
    from aiida_koopmans.types import ExplicitProjectionBlock, SpinChannel
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    if not projection_blocks:
        raise ValueError(
            "Wannier-function initialisation requires explicit projections in "
            "``calculator_parameters.w90.projections``."
        )

    suffix = f"_{spin_channel.value}" if spin_channel in (SpinChannel.UP, SpinChannel.DOWN) else ""
    blocks: list[Any] = []
    cursor = 0
    n_occ = n_emp = 0
    for block in projection_blocks:
        num_wann = sum(projection_num_wann(structure, p) for p in block)
        start, end = cursor + 1, cursor + num_wann
        if end <= nocc:
            n_occ += 1
            label = f"occ{suffix}_{n_occ}"
        elif cursor >= nocc:
            n_emp += 1
            label = f"emp{suffix}_{n_emp}"
        else:
            raise ValueError(
                f"A projection block (bands {start}-{end}) straddles the occupied/empty "
                f"boundary at band {nocc}."
            )
        if end > nbnd:
            raise ValueError(f"The projection blocks span {end} bands but nbnd = {nbnd}.")
        blocks.append(
            ExplicitProjectionBlock(
                label=label,
                spin=spin_channel,
                num_wann=num_wann,
                num_bands=num_wann,
                include_bands=list(range(start, end + 1)),
                exclude_bands=band_range_complement(start, end, nbnd),
                projection_type=WannierProjectionType.ANALYTIC,
                projections=[projection_win_string(p) for p in block],
            )
        )
        cursor = end

    # The uppermost block per spin channel absorbs the remaining
    # ``nbnd - cursor`` bands as its disentanglement pool (``num_bands =
    # num_wann + num_extra_bands``) and excludes nothing above itself —
    # without this an entangled empty manifold (e.g. Si conduction bands)
    # has no window to disentangle from and the folded empty states are
    # garbage.
    if blocks and cursor < nbnd:
        last = blocks[-1]
        last["num_bands"] = last["num_wann"] + (nbnd - cursor)
        start = last["include_bands"][0]
        last["exclude_bands"] = list(range(1, start)) or None

    covered_occ = sum(b["num_wann"] for b in blocks if b["include_bands"][0] <= nocc)
    if covered_occ != nocc:
        raise ValueError(
            f"The occupied projection blocks span {covered_occ} Wannier functions but "
            f"the system has {nocc} occupied bands per primitive cell; every occupied "
            "band must be covered for the Wannier-seeded kcp.x initialisation."
        )
    return blocks


def _dscf_wannier_init_inputs(
    koopmans_input: KoopmansInput,
    structure: orm.StructureData,
    codes: dict[str, orm.AbstractCode],
    nbnd: int,
) -> dict[str, Any]:
    """Assemble the extra ``KoopmansDSCFWorkflow`` inputs for the Wannier route.

    Covers the periodic mlwfs/projwfs initialisation: the wannierize +
    fold-to-supercell codes, the projection blocks (primitive band indices;
    per spin channel when ``spin='collinear'``), the k-mesh, and the
    Makov-Payne knobs. The molecular/kohn-sham route needs none of this.
    """
    from aiida_koopmans.types import SpinChannel

    from koopmans.aiida.conversion import (
        get_pseudos_from_family,
        kpoints_input_to_kpoints_mesh,
    )

    workflow = koopmans_input.workflow
    calc_params = koopmans_input.calculator_parameters
    kpoints_input = koopmans_input.kpoints

    if isinstance(workflow.eps_inf, str):
        raise NotImplementedError(
            "eps_inf='auto' is not wired for the DSCF stream yet (the DielectricTask "
            "exists — hook it up like the DFPT dispatcher); provide a numeric value."
        )

    pseudo_family = workflow.pseudo_library
    pseudos = get_pseudos_from_family(pseudo_family, structure)
    nelec = round(sum(pseudos[site.kind_name].z_valence for site in structure.sites))

    if workflow.spin == SpinType.COLLINEAR:
        w90 = calc_params.wannier90
        if w90.up is None or w90.down is None:
            raise ValueError(
                "spin='collinear' Wannier initialisation needs per-spin projections: set "
                "``calculator_parameters.w90.up.projections`` and "
                "``calculator_parameters.w90.down.projections``."
            )
        magnetization = _coerce_optional_int(calc_params.tot_magnetization)
        if magnetization is None:
            raise ValueError(
                "spin='collinear' Wannier initialisation needs "
                "``calculator_parameters.tot_magnetization``."
            )
        if (nelec + magnetization) % 2:
            raise ValueError(
                f"nelec = {nelec} and tot_magnetization = {magnetization} do not give "
                "integer per-channel occupations."
            )
        blocks = _derive_dscf_blocks(
            structure, w90.up.projections, (nelec + magnetization) // 2, nbnd, SpinChannel.UP
        ) + _derive_dscf_blocks(
            structure, w90.down.projections, (nelec - magnetization) // 2, nbnd, SpinChannel.DOWN
        )
    else:
        if nelec % 2:
            raise ValueError(
                f"Odd electron count ({nelec}) requires spin='collinear' for the "
                "Wannier-initialised DSCF route."
            )
        blocks = _derive_dscf_blocks(
            structure, calc_params.wannier90.projections, nelec // 2, nbnd, SpinChannel.NONE
        )

    parameters = input_to_pw_parameters(koopmans_input)
    wannier_overrides: WannierizeOverrides = {
        "scf": {"pseudo_family": pseudo_family, "pw": {"parameters": parameters}},
        "nscf": {"pseudo_family": pseudo_family, "pw": {"parameters": parameters}},
    }

    # User wannier90 keywords (disentanglement windows, iteration counts, ...)
    # feed every per-block wannierisation. Flat by design (see
    # ``WannierizeOverrides``): the upstream namespace-nested override shape
    # is produced only inside the block wannierization builder.
    w90_user = calc_params.wannier90.model_dump(
        exclude_unset=True, exclude={"projections", "up", "down"}
    )
    if w90_user:
        wannier_overrides["wannier90"] = w90_user

    wannier_codes = dict(codes)
    wannier_codes.setdefault("wannier90", _load_code("wannier90", "wannier90.x"))
    wannier_codes.setdefault("pw2wannier90", _load_code("pw2wannier90", "pw2wannier90.x"))
    wannier_codes.setdefault("wann2kcp", _load_code("wann2kcp", "wann2kcp.x"))
    wannier_codes.setdefault("merge_evc", _load_code("merge_evc", "merge_evc.x"))

    return {
        "codes": wannier_codes,
        "blocks": blocks,
        "kgrid": list(kpoints_input.grid),
        "kpoints": kpoints_input_to_kpoints_mesh(kpoints_input),
        "gamma_only": bool(getattr(kpoints_input, "gamma_only", False)),
        "wannier_overrides": wannier_overrides,
        "mp_correction": workflow.mp_correction,
        "eps_inf": workflow.eps_inf,
    }


def _build_singlepoint_dfpt_workgraph(
    koopmans_input: KoopmansInput,
    codes: Codes,
) -> WorkGraph:
    """Build a workgraph for a singlepoint Koopmans calculation with DFPT screening.

    Assembles the full chain (scf + nscf → per-manifold wannierization →
    wann2kc → screen → ham) via ``aiida_koopmans.workgraphs.dfpt.SinglepointDFPTWorkflow``.

    Spin regimes (``workflow.spin``): ``none`` runs the closed-shell chain;
    ``collinear`` fans the wannierization and the kcw.x chain out per spin
    channel (needs per-spin projections in ``w90.up`` / ``w90.down`` and a
    ``tot_magnetization``); ``non_collinear`` / ``spin_orbit`` run the spinor
    chain (all bands singly occupied, ``num_wann`` doubled).

    Remaining restrictions (mirroring the ``SinglepointDFPTWorkflow`` scope):
    periodic, MLWF/projwf variational orbitals, and explicit projections.
    A manifold may span several projection blocks; their Wannier products
    are merged back into one file set before kcw.x consumes them.
    """
    from aiida_koopmans.workgraphs.dfpt import SinglepointDFPTWorkflow

    from koopmans.aiida.conversion import (
        get_pseudos_from_family,
        kpoints_input_to_kpoints_mesh,
        kpoints_input_to_kpoints_path,
    )

    workflow = koopmans_input.workflow

    group_orbitals_tol = _dfpt_grouping_tol(workflow)
    if workflow.correction != Correction.KI:
        raise NotImplementedError(
            "The DFPT route (kcw.x) only implements the KI correction; "
            f"correction={workflow.correction.value!r} is not supported. Use "
            "screening_method = 'dscf' for KIPZ."
        )
    if workflow.init_orbitals not in (
        VariationalOrbitalType.MLWFS,
        VariationalOrbitalType.PROJWFS,
    ):
        raise NotImplementedError(
            "DFPT screening only supports Wannier-function variational orbitals "
            "(init_orbitals = 'mlwfs' or 'projwfs'). The molecular kcw_at_ks path is "
            "not yet wired."
        )
    if getattr(koopmans_input.kpoints, "gamma_only", False):
        raise NotImplementedError(
            "Gamma-only DFPT (isolated systems) is not yet supported; provide a k-point grid."
        )
    eps_inf = _validated_eps_inf(workflow.eps_inf)

    calc_params = koopmans_input.calculator_parameters
    spin = workflow.spin

    if spin == SpinType.COLLINEAR:
        if calc_params.wannier90.up is None or calc_params.wannier90.down is None:
            raise ValueError(
                "spin='collinear' DFPT screening needs per-spin projections: set "
                "``calculator_parameters.w90.up.projections`` and "
                "``calculator_parameters.w90.down.projections``."
            )
        if calc_params.tot_magnetization is None:
            raise ValueError(
                "spin='collinear' DFPT screening needs "
                "``calculator_parameters.tot_magnetization`` to fix the per-channel "
                "occupations."
            )

    structure, pseudo_family, overrides = _prepare_common_inputs(koopmans_input, ["scf", "nscf"])

    # User wannier90 keywords (disentanglement windows, iteration counts, ...)
    # feed every per-block wannierisation. Flat by design (see
    # ``WannierizeOverrides``): the upstream namespace-nested override shape
    # is produced only inside the block wannierization builder. Projections
    # and per-spin blocks are consumed separately by the manifold derivation.
    w90_user = calc_params.wannier90.model_dump(
        exclude_unset=True, exclude={"projections", "up", "down"}
    )
    if w90_user:
        w90_overrides: WannierizeOverrides = {"wannier90": w90_user}
        overrides.update(w90_overrides)

    # Electron count from the pseudopotential valences: fixes the size of the
    # occupied manifold.
    pseudos = get_pseudos_from_family(pseudo_family, structure)
    nelec = round(sum(pseudos[site.kind_name].z_valence for site in structure.sites))

    nbnd = calc_params.nbnd if calc_params.nbnd is not None else calc_params.pw.system.nbnd
    nbnd = int(nbnd) if nbnd is not None else None

    if spin == SpinType.COLLINEAR:
        manifolds = _collinear_dfpt_manifolds(koopmans_input, structure, overrides, nelec, nbnd)
    else:
        manifolds = _single_channel_dfpt_manifolds(koopmans_input, structure, nelec, nbnd, spin)

    bands_kpoints = (
        kpoints_input_to_kpoints_path(koopmans_input.kpoints, structure)
        if koopmans_input.kpoints.path is not None
        else None
    )

    # The wannierization steps need codes that load_codes_for_task only wires
    # for the WANNIERIZE task; load them here until it grows a DFPT branch.
    codes = dict(codes)
    codes.setdefault("wannier90", _load_code("wannier90", "wannier90.x"))
    codes.setdefault("pw2wannier90", _load_code("pw2wannier90", "pw2wannier90.x"))
    if eps_inf == "auto":
        codes.setdefault("ph", _load_code("ph", "ph.x"))

    return SinglepointDFPTWorkflow.build(
        codes=codes,
        structure=structure,
        kpoints=kpoints_input_to_kpoints_mesh(koopmans_input.kpoints),
        kgrid=list(koopmans_input.kpoints.grid),
        bands_kpoints=bands_kpoints,
        pseudo_family=pseudo_family,
        overrides=overrides,
        # 'auto' prepends the scf + ph.x dielectric chain inside
        # SinglepointDFPT; l_vcut is the Gygi-Baldereschi flag (None -> the
        # periodic default, on).
        eps_inf=eps_inf,
        l_vcut=workflow.gb_correction,
        spin=spin,
        manifolds=manifolds,
        group_orbitals_tol=group_orbitals_tol,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
    )


def _reject_explicit_orbital_groups(workflow: WorkflowConfig) -> None:
    """Reject an explicit ``orbital_groups`` list until the fan-out threads it.

    The field parses and validates but is never carried into the per-orbital
    screening fan-out, so an explicit grouping would be honoured nowhere and
    orbitals would silently fall back to the criterion-based grouping. Fail
    loudly instead and point at the criterion that is wired up.
    """
    if workflow.orbital_groups is not None:
        raise NotImplementedError(
            "explicit orbital_groups are not yet threaded into the screening "
            "fan-out; use group_orbitals_by / group_orbitals_tol to group "
            "orbitals by self-Hartree energy (DSCF) or wannier90 spread (DFPT)."
        )


def _dfpt_grouping_tol(workflow: WorkflowConfig) -> float | None:
    """Resolve the workflow-level orbital-grouping tolerance for the DFPT route.

    Returns the tolerance for ``'spread'`` (grouping on), ``None`` for
    ``'none'`` / unset (no workflow-level grouping), and raises for
    ``'self_hartree'``, which the DFPT route has no metric for.
    """
    _reject_explicit_orbital_groups(workflow)
    criterion = workflow.group_orbitals_by
    if criterion is None or criterion == GroupOrbitalsBy.NONE:
        return None
    if criterion == GroupOrbitalsBy.SPREAD:
        return workflow.group_orbitals_tol
    raise NotImplementedError(
        f"group_orbitals_by={criterion.value!r} is not implemented for DFPT "
        "screening: the DFPT route clusters orbitals by their wannier90 spread. "
        "Use group_orbitals_by = 'spread' (or 'none')."
    )


def _validated_eps_inf(eps_inf: float | str | None) -> float | str | None:
    """Check that ``eps_inf`` is a numeric value, ``'auto'``, or unset."""
    if isinstance(eps_inf, str) and eps_inf != "auto":
        raise ValueError(
            f"eps_inf={eps_inf!r} is not understood: provide a numeric value "
            "or 'auto' (compute the dielectric constant with ph.x)."
        )
    return eps_inf


def _single_channel_dfpt_manifolds(
    koopmans_input: KoopmansInput,
    structure: orm.StructureData,
    nelec: int,
    nbnd: int | None,
    spin: SpinType,
) -> dict[str, Any]:
    """Derive the single-channel ``manifolds`` input for an unpolarized or spinor DFPT run.

    Both regimes run one kcw.x chain keyed ``"none"``; the spinor case
    differs only in the manifold derivation (all bands singly occupied,
    ``num_wann`` doubled).
    """
    from aiida_koopmans.types import ProjectionBlock, SpinChannel
    from aiida_koopmans.workgraphs.dfpt import (
        ManifoldBlocks,
        derive_dfpt_manifolds,
        normalize_alpha_guess,
    )

    workflow = koopmans_input.workflow
    spin_channel = SpinChannel.NONE if spin == SpinType.NONE else SpinChannel.SPINOR
    occ_blocks, emp_blocks, _has_disentangle, n_orbitals = derive_dfpt_manifolds(
        structure=structure,
        projection_blocks=koopmans_input.calculator_parameters.wannier90.projections,
        nelec=nelec,
        nbnd=nbnd,
        spin_channel=spin_channel,
    )
    # derive_dfpt_manifolds narrows to ExplicitProjectionBlock; the manifold
    # field holds the ProjectionBlock union (list invariance needs the cast).
    manifold = ManifoldBlocks(occ=cast(list[ProjectionBlock], occ_blocks))
    if emp_blocks:
        manifold["emp"] = cast(list[ProjectionBlock], emp_blocks)
    if not workflow.calculate_alpha:
        manifold["alpha_guess"] = normalize_alpha_guess(workflow.alpha_guess, n_orbitals)
    return {SpinChannel.NONE.value: manifold}


def _collinear_dfpt_manifolds(
    koopmans_input: KoopmansInput,
    structure: orm.StructureData,
    overrides: dict[str, Any],
    nelec: int,
    nbnd: int | None,
) -> dict[str, Any]:
    """Derive the per-spin-channel ``manifolds`` input for a collinear DFPT run.

    Returns the ``SinglepointDFPTWorkflow`` ``manifolds`` dict — one
    ``ManifoldBlocks`` per spin channel, keyed ``"up"`` / ``"down"`` — from
    the per-spin projections in ``w90.up`` / ``w90.down`` and the
    per-channel occupations fixed by ``tot_magnetization``. Also forwards
    the magnetization into the scf / nscf PW SYSTEM overrides (mutated in
    place): the PW runs must see the physical magnetization —
    ``SinglepointDFPTWorkflow`` only forces ``nspin=2`` in this regime.
    """
    from aiida_koopmans.types import ProjectionBlock, SpinChannel
    from aiida_koopmans.workgraphs.dfpt import (
        ManifoldBlocks,
        derive_dfpt_manifolds,
        normalize_alpha_guess,
    )

    workflow = koopmans_input.workflow
    w90 = koopmans_input.calculator_parameters.wannier90
    tot_magnetization = koopmans_input.calculator_parameters.tot_magnetization
    if w90.up is None or w90.down is None or tot_magnetization is None:
        # Already validated by _build_singlepoint_dfpt_workgraph; re-checked
        # here so the collinear helper narrows its own inputs.
        raise ValueError(
            "spin='collinear' DFPT screening needs per-spin projections "
            "(``w90.up`` / ``w90.down``) and ``tot_magnetization``."
        )
    magnetization = int(tot_magnetization)
    if (nelec + magnetization) % 2:
        raise ValueError(
            f"nelec = {nelec} and tot_magnetization = {magnetization} do not give "
            "integer per-channel occupations."
        )
    for key in ("scf", "nscf"):
        overrides[key]["pw"]["parameters"].setdefault("SYSTEM", {})["tot_magnetization"] = (
            magnetization
        )

    manifolds: dict[str, Any] = {}
    for channel, w90_channel in ((SpinChannel.UP, w90.up), (SpinChannel.DOWN, w90.down)):
        sign = 1 if channel == SpinChannel.UP else -1
        occ_blocks, emp_blocks, _has_disentangle, n_orbitals = derive_dfpt_manifolds(
            structure=structure,
            projection_blocks=w90_channel.projections,
            nelec=nelec,
            nbnd=nbnd,
            spin_channel=channel,
            nocc=(nelec + sign * magnetization) // 2,
        )
        manifold = ManifoldBlocks(occ=cast(list[ProjectionBlock], occ_blocks))
        if emp_blocks:
            manifold["emp"] = cast(list[ProjectionBlock], emp_blocks)
        if not workflow.calculate_alpha:
            manifold["alpha_guess"] = normalize_alpha_guess(
                workflow.alpha_guess, n_orbitals, channel
            )
        manifolds[channel.value] = manifold
    return manifolds


def _build_trajectory_workgraph(
    koopmans_input: KoopmansInput,
    codes: Codes,
) -> WorkGraph:
    """Build a workgraph for a trajectory (machine-learning train/test) task.

    Fans the snapshots out over per-snapshot ``KoopmansDSCFWorkflow`` runs via
    ``aiida_koopmans.workgraphs.ml.TrajectoryWorkflow`` and, depending on the
    ``ml`` configuration, trains a screening-parameter model on the computed
    alphas (``ml:train``) or scores an existing model against them
    (``ml:test``).

    Current limitations (raise ``NotImplementedError`` / ``ValueError``):

    - ``ml:predict`` needs per-orbital alpha injection, which the frozen
      ``KoopmansDSCFWorkflow`` interface does not support.
    - Only the ``self_hartree`` descriptor is exposed; the
      ``orbital_density`` power-spectrum descriptor has its full
      pw2wannier90 ``decompose`` route built and unit-tested in
      ``aiida-koopmans`` (``OrbitalDensityDatasetWorkflow``), but stays
      gated pending a live daemon regression that confirms the per-block
      Wannier-function-to-alpha ordering against the legacy reference.

    Each frame of the ``atoms.snapshots`` xyz becomes one ``snapshot_N``
    structure fed to the dynamic snapshots namespace.
    """
    from json import load as json_load

    from aiida_koopmans.workgraphs.ml import TrajectoryWorkflow

    from koopmans.aiida.setup.pseudos import ensure_pseudo_family_installed

    workflow = koopmans_input.workflow

    _reject_unwired_external_projectors(koopmans_input, "trajectory")

    if workflow.calculate_alpha and workflow.screening_method == CalculateScreeningMethod.DFPT:
        raise NotImplementedError(
            "The trajectory task only supports DSCF screening (kcp.x); DFPT screening "
            "is not yet implemented for trajectories."
        )

    _require_supported_correction(workflow.correction)

    if workflow.spin in (SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT):
        raise NotImplementedError(
            f"spin={workflow.spin.value!r} is not supported by the trajectory (kcp.x) "
            "stream: kcp.x has no noncollinear mode."
        )

    ml_config = koopmans_input.ml

    if ml_config.predict:
        raise NotImplementedError(
            "ml:predict is not yet supported: injecting per-orbital predicted alphas "
            "(and skipping the Delta-SCF refinement) requires an extension of the "
            "KoopmansDSCFWorkflow interface, which currently accepts only a scalar "
            "initial_alpha."
        )
    if (ml_config.train or ml_config.test) and ml_config.descriptor != "self_hartree":
        raise NotImplementedError(
            f"ml:descriptor={ml_config.descriptor!r} is implemented but gated "
            "pending live alignment validation. The full pw2wannier90 "
            "wan_mode='decompose' route is built and unit-tested "
            "(aiida_koopmans.workgraphs.ml.OrbitalDensityDatasetWorkflow, fed by "
            "the nscf scratch and per-block wannierizations now on "
            "KoopmansDSCFOutputs); the decompose math is reproduced to machine "
            "precision, but the per-block Wannier-function-to-alpha ordering "
            "awaits a live daemon regression against the legacy reference before "
            "the descriptor is exposed. Use ml:descriptor='self_hartree'."
        )
    ml_mode = "train" if ml_config.train else "test" if ml_config.test else "none"

    ml_model = None
    if ml_mode == "test":
        if ml_config.model_file is None:
            raise ValueError(
                "ml:test requires ml:model_file (the JSON model produced by an ml:train run)."
            )
        with open(ml_config.model_file) as handle:
            ml_model = json_load(handle)

    snapshots = atoms_input_to_structures(koopmans_input.atoms)
    ensure_pseudo_family_installed(workflow.pseudo_library)

    return TrajectoryWorkflow.build(
        code=codes["kcp"],
        snapshots=snapshots,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        **_kcp_dscf_inputs(koopmans_input),
        ml_mode=ml_mode,
        ml_model=ml_model,
        estimator=ml_config.estimator,
        descriptor=ml_config.descriptor,
        occ_and_emp_together=ml_config.occ_and_emp_together,
    )


def _require_supported_correction(correction: Correction) -> None:
    """Raise for corrections the kcp.x (DSCF) route does not support yet."""
    supported = {Correction.KI, Correction.KIPZ}
    if correction not in supported:
        raise NotImplementedError(
            f"correction={correction.value!r} is not yet supported. "
            f"Supported: {sorted(c.value for c in supported)}. "
            "PKIPZ requires a perturbative post-processing step; "
            "NONE / ALL are workflow-control flags."
        )


class _KcpDscfInputs(TypedDict):
    """Scalar inputs shared by the kcp.x DSCF builders (singlepoint and trajectory)."""

    pseudo_family: str
    ecutwfc: float
    ecutrho: float
    nbnd: int
    nspin: int
    tot_magnetization: int | None
    correction: Correction
    init_orbitals: VariationalOrbitalType
    alpha_numsteps: int
    fix_spin_contamination: bool
    initial_alpha: float
    spin_polarized: bool
    orbital_groups_self_hartree_tol: float | None


def _initial_alpha_from_guess(alpha_guess: float | list[float]) -> float:
    """Collapse the user ``alpha_guess`` to the scalar the kcp.x DSCF route accepts.

    ``KoopmansDSCFWorkflow`` seeds every orbital with the same starting alpha,
    so a list is only accepted when all its entries agree.

    Raises:
        NotImplementedError: If ``alpha_guess`` lists distinct per-orbital values.
    """
    if isinstance(alpha_guess, float):
        return alpha_guess
    if len(set(alpha_guess)) > 1:
        raise NotImplementedError(
            "Distinct per-orbital alpha_guess values are not yet supported on the "
            "DSCF route; provide a single starting alpha."
        )
    return float(alpha_guess[0])


def _kcp_dscf_inputs(koopmans_input: KoopmansInput) -> _KcpDscfInputs:
    """Assemble the scalar kwargs shared by the kcp.x DSCF builders.

    ``ecutwfc``/``nbnd`` prefer the top-level ``calculator_parameters``
    convenience fields and fall back to the ``kcp.system`` Pydantic block;
    ``ecutrho`` has no top-level field — read from ``kcp.system`` and default
    to ``4 * ecutwfc`` when unset.
    """
    workflow = koopmans_input.workflow
    calc_params = koopmans_input.calculator_parameters
    kcp_system = calc_params.kcp.system

    ecutwfc = calc_params.ecutwfc if calc_params.ecutwfc is not None else kcp_system.ecutwfc
    if not ecutwfc:
        raise ValueError(
            "ecutwfc is required for a Koopmans singlepoint calculation. Set it in "
            "``calculator_parameters.ecutwfc`` or ``calculator_parameters.kcp.system.ecutwfc``."
        )

    ecutrho = kcp_system.ecutrho if kcp_system.ecutrho else 4.0 * ecutwfc

    nbnd_raw = calc_params.nbnd if calc_params.nbnd is not None else kcp_system.nbnd
    if nbnd_raw is None:
        raise ValueError(
            "nbnd is required for a Koopmans singlepoint calculation. Set it in "
            "``calculator_parameters.nbnd`` or ``calculator_parameters.kcp.system.nbnd``."
        )

    return _KcpDscfInputs(
        pseudo_family=workflow.pseudo_library,
        ecutwfc=float(ecutwfc),
        ecutrho=float(ecutrho),
        nbnd=int(nbnd_raw),
        # KI requires nspin=2 for per-spin orbital-dependent screening, regardless
        # of what ``spin`` says — closed-shell molecules still need two channels.
        nspin=2,
        tot_magnetization=_coerce_optional_int(calc_params.tot_magnetization),
        correction=workflow.correction,
        init_orbitals=workflow.init_orbitals,
        alpha_numsteps=workflow.alpha_numsteps,
        fix_spin_contamination=workflow.fix_spin_contamination,
        initial_alpha=_initial_alpha_from_guess(workflow.alpha_guess),
        spin_polarized=workflow.spin == SpinType.COLLINEAR,
        orbital_groups_self_hartree_tol=_grouping_tol(workflow),
    )


def _grouping_tol(workflow: WorkflowConfig) -> float | None:
    """Translate the orbital-grouping fields into the plugin's self-Hartree tolerance.

    The schema resolves ``group_orbitals_by`` / ``group_orbitals_tol``
    (including their route-dependent defaults) at parse time; here only the
    implemented criterion passes through.
    """
    _reject_explicit_orbital_groups(workflow)
    if workflow.group_orbitals_by == GroupOrbitalsBy.NONE:
        return None
    if workflow.group_orbitals_by == GroupOrbitalsBy.SELF_HARTREE:
        return workflow.group_orbitals_tol
    criterion = workflow.group_orbitals_by.value if workflow.group_orbitals_by else None
    raise NotImplementedError(
        f"group_orbitals_by={criterion!r} is not implemented; supported: 'self_hartree', 'none'."
    )


def _coerce_optional_int(value: float | None) -> int | None:
    """Return ``int(value)`` when value is given, else ``None``."""
    return int(value) if value is not None else None
