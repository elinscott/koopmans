"""Workflow building logic for koopmans AiiDA integration.

This module handles selecting and constructing the appropriate AiiDA workgraph
based on the task specified in a KoopmansInput.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from aiida import orm
from aiida_koopmans.workgraphs import Codes
from aiida_quantumespresso.common.types import SpinType

from koopmans.aiida.conversion import (
    atoms_input_to_structure,
    input_to_pw_parameters,
)
from koopmans.input_file.workflow import (
    CalculateScreeningMethod,
    Correction,
    Task,
    VariationalOrbitalType,
)

if TYPE_CHECKING:
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

    overrides: dict[str, Any] = {
        key: {
            "pseudo_family": pseudo_family,
            "pw": {
                "parameters": parameters,
            },
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
        A WorkGraph for Wannier90WorkChain.
    """
    from aiida_koopmans.workgraphs.wannier90 import Wannierize
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    structure, pseudo_family, overrides = _prepare_common_inputs(koopmans_input, ["scf", "nscf"])

    # Check if external projectors are requested
    pw2w_params = koopmans_input.calculator_parameters.pw2wannier90
    extra_kwargs: dict[str, Any] = {}
    if pw2w_params.atom_proj_ext:
        extra_kwargs["projection_type"] = WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL
        extra_kwargs["external_projectors_path"] = str(pw2w_params.atom_proj_dir)

    return Wannierize.build(
        codes=codes,
        structure=structure,
        overrides=overrides,
        pseudo_family=pseudo_family,
        print_summary=False,
        **extra_kwargs,
    )


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
        # The prod(kgrid) supercell images of each primitive Wannier function
        # are physically equivalent, but constructive image grouping is not
        # implemented yet. Group them at runtime by self-Hartree energy
        # instead: a tight tolerance only merges numerically identical
        # orbitals, so the per-orbital fan-out collapses to the primitive
        # count without risking real physics.
        if inputs["orbital_groups_self_hartree_tol"] is None:
            inputs["orbital_groups_self_hartree_tol"] = 1.0e-4

    return KoopmansDSCFWorkflow.build(
        code=codes["kcp"],
        structure=structure,
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

    Unlike the DFPT manifolds (one occupied + at most one empty block), the
    DSCF route wannierises every user block separately and merges them per
    (filling, spin) via merge_evc.x, so any number of blocks is allowed.
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
    periodic, MLWF/projwf variational orbitals, and explicit projections
    forming exactly one occupied manifold block plus at most one empty block
    per spin channel (multi-block manifolds are not yet supported).
    """
    from aiida_koopmans.workgraphs.dfpt import SinglepointDFPTWorkflow

    from koopmans.aiida.conversion import (
        get_pseudos_from_family,
        kpoints_input_to_kpoints_mesh,
        kpoints_input_to_kpoints_path,
    )

    workflow = koopmans_input.workflow

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
    from aiida_koopmans.types import SpinChannel
    from aiida_koopmans.workgraphs.dfpt import (
        ManifoldBlocks,
        derive_dfpt_manifolds,
        normalize_alpha_guess,
    )

    workflow = koopmans_input.workflow
    spin_channel = SpinChannel.NONE if spin == SpinType.NONE else SpinChannel.SPINOR
    occ_block, emp_block, n_orbitals = derive_dfpt_manifolds(
        structure=structure,
        projection_blocks=koopmans_input.calculator_parameters.wannier90.projections,
        nelec=nelec,
        nbnd=nbnd,
        spin_channel=spin_channel,
    )
    manifold = ManifoldBlocks(occ=occ_block)
    if emp_block is not None:
        manifold["emp"] = emp_block
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
    from aiida_koopmans.types import SpinChannel
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
        occ_block, emp_block, n_orbitals = derive_dfpt_manifolds(
            structure=structure,
            projection_blocks=w90_channel.projections,
            nelec=nelec,
            nbnd=nbnd,
            spin_channel=channel,
            nocc=(nelec + sign * magnetization) // 2,
        )
        manifold = ManifoldBlocks(occ=occ_block)
        if emp_block is not None:
            manifold["emp"] = emp_block
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
    - Only the ``self_hartree`` descriptor is wired; ``orbital_density``
      needs kcp.x orbital-density retrieval.
    - Multi-snapshot (xyz trajectory) input is not representable in the
      ``KoopmansInput`` schema yet, so the single input structure is run as
      a one-snapshot trajectory.
    """
    from json import load as json_load

    from aiida_koopmans.workgraphs.ml import TrajectoryWorkflow

    from koopmans.aiida.setup.pseudos import ensure_pseudo_family_installed

    workflow = koopmans_input.workflow

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
            f"ml:descriptor={ml_config.descriptor!r} is not yet supported: the "
            "orbital_density (power-spectrum) descriptor requires retrieving the "
            "trial KI's real-space orbital densities from kcp.x. Use "
            "ml:descriptor='self_hartree'."
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

    structure = atoms_input_to_structure(koopmans_input.atoms)
    ensure_pseudo_family_installed(workflow.pseudo_library)

    # The input schema cannot express multiple snapshots yet, so run the
    # single input structure as a one-snapshot trajectory.
    snapshots = {"snapshot_1": structure}

    return TrajectoryWorkflow.build(
        code=codes["kcp"],
        snapshots=snapshots,
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
        orbital_groups_self_hartree_tol=workflow.orbital_groups_self_hartree_tol,
    )


def _coerce_optional_int(value: float | None) -> int | None:
    """Return ``int(value)`` when value is given, else ``None``."""
    return int(value) if value is not None else None
