"""Workflow building logic for koopmans AiiDA integration.

This module handles selecting and constructing the appropriate AiiDA workgraph
based on the task specified in a KoopmansInput.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiida import orm

from koopmans.aiida.conversion import (
    atoms_input_to_structure,
    input_to_pw_parameters,
)
from koopmans.input_file.workflow import CalculateScreeningMethod, Correction, SpinType, Task

if TYPE_CHECKING:
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


def load_codes_for_task(workflow: WorkflowConfig) -> dict[str, orm.AbstractCode]:
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
        NotImplementedError: If the requested code combination is not ported yet.
    """
    task = workflow.task
    codes: dict[str, orm.AbstractCode] = {}

    # The UI (unfold-and-interpolate) task is pure python post-processing of
    # previously generated Wannier Hamiltonian files; it runs no QE codes.
    if task == Task.UI:
        return codes

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

        # TODO: projwfc is only needed when the Wannierize flow computes a
        # projected DOS / bandstructure. Silently swallowing the lookup error
        # here lets workflows run without projwfc installed, but also masks
        # "projwfc required but missing" cases. Replace with a predicate on
        # the relevant workflow flag (likely `calculate_bands` or a
        # wannier90.bands_plot override) and insist on the code being
        # installed when that predicate fires.
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
    from koopmans.aiida.setup import ensure_pseudo_family_installed

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
    elif task == Task.UI:
        return _build_ui_workgraph(koopmans_input, codes)
    else:
        raise ValueError(
            f"Task '{task.value}' is not yet implemented. "
            f"Supported tasks: {Task.DFT_BANDS.value}, {Task.WANNIERIZE.value}, "
            f"{Task.SINGLEPOINT.value}, {Task.TRAJECTORY.value}, {Task.DFT_EPS.value}, "
            f"{Task.UI.value}"
        )


def _build_dft_bands_workgraph(
    koopmans_input: KoopmansInput,
    codes: dict[str, orm.AbstractCode],
) -> WorkGraph:
    """Build a workgraph for DFT bands calculation.

    Args:
        koopmans_input: The parsed koopmans input.
        codes: Dictionary of loaded codes.

    Returns:
        A WorkGraph for PwBandsWorkChain.
    """
    from aiida_koopmans.workgraphs.pw import PwBandsTaskViaBuilder

    structure, _pseudo_family, overrides = _prepare_common_inputs(koopmans_input, ["scf", "bands"])

    return PwBandsTaskViaBuilder.build(
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
    codes: dict[str, orm.AbstractCode],
) -> WorkGraph:
    """Build a workgraph for Wannierization.

    Args:
        koopmans_input: The parsed koopmans input.
        codes: Dictionary of loaded codes.

    Returns:
        A WorkGraph for Wannier90WorkChain.
    """
    from aiida_koopmans.workgraphs.wannier90 import Wannier90TaskViaBuilder
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    structure, pseudo_family, overrides = _prepare_common_inputs(koopmans_input, ["scf", "nscf"])

    # Check if external projectors are requested
    pw2w_params = koopmans_input.calculator_parameters.pw2wannier90
    extra_kwargs: dict[str, Any] = {}
    if pw2w_params.atom_proj_ext:
        extra_kwargs["projection_type"] = WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL
        extra_kwargs["external_projectors_path"] = pw2w_params.atom_proj_dir

    return Wannier90TaskViaBuilder.build(
        codes=codes,
        structure=structure,
        overrides=overrides,
        pseudo_family=pseudo_family,
        print_summary=False,
        **extra_kwargs,
    )


def _build_singlepoint_workgraph(
    koopmans_input: KoopmansInput,
    codes: dict[str, orm.AbstractCode],
) -> WorkGraph:
    """Build a workgraph for a singlepoint Koopmans calculation.

    Dispatches on ``workflow.screening_method`` first (DSCF vs DFPT), then on
    ``workflow.correction``:

    - DSCF + ``KI``/``KIPZ`` → ``KoopmansDSCFWorkflow`` (kcp.x)
    - DFPT → ``_build_singlepoint_dfpt_workgraph`` (kcw.x)
    - ``PKIPZ``, ``NONE``, ``ALL`` → ``NotImplementedError`` (not yet ported)
    """
    from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow

    from koopmans.aiida.setup import ensure_pseudo_family_installed

    workflow = koopmans_input.workflow

    # DFPT routes on the screening method alone: calculate_alpha = False is
    # the alpha_guess path inside the DFPT builder (screen step skipped),
    # not a reason to fall through to the kcp.x/DSCF branch.
    if workflow.screening_method == CalculateScreeningMethod.DFPT:
        return _build_singlepoint_dfpt_workgraph(koopmans_input, codes)

    correction = workflow.correction
    supported = {Correction.KI, Correction.KIPZ}
    if correction not in supported:
        raise NotImplementedError(
            f"correction={correction.value!r} is not yet ported. "
            f"Supported: {sorted(c.value for c in supported)}. "
            "PKIPZ requires a perturbative post-processing step; "
            "NONE / ALL are workflow-control flags."
        )

    if workflow.spin in (SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT):
        raise NotImplementedError(
            f"spin={workflow.spin.value!r} is not supported by the DSCF (kcp.x) stream: "
            "kcp.x has no noncollinear mode. Use screening_method='dfpt'."
        )

    structure = atoms_input_to_structure(koopmans_input.atoms)
    pseudo_family = workflow.pseudo_library
    ensure_pseudo_family_installed(pseudo_family)

    ecutwfc, ecutrho, nbnd, nspin = _extract_kcp_scalar_inputs(koopmans_input)

    initial_alpha = (
        workflow.alpha_guess if isinstance(workflow.alpha_guess, float) else workflow.alpha_guess[0]
    )

    from koopmans.input_file.workflow import VariationalOrbitalType

    extra_kwargs: dict[str, Any] = {}
    self_hartree_tol = workflow.orbital_groups_self_hartree_tol
    if workflow.init_orbitals in (
        VariationalOrbitalType.MLWFS,
        VariationalOrbitalType.PROJWFS,
    ):
        extra_kwargs = _dscf_wannier_init_inputs(koopmans_input, structure, codes, nbnd)
        # The prod(kgrid) supercell images of each primitive Wannier function
        # are physically equivalent, but the constructive image grouping of
        # legacy ``_koopmans_dscf.py:141-148`` is not ported yet. Group them
        # at runtime by self-Hartree energy instead: a tight tolerance only
        # merges numerically identical orbitals, so the per-orbital fan-out
        # collapses to the primitive count without risking real physics.
        if self_hartree_tol is None:
            self_hartree_tol = 1.0e-4

    return KoopmansDSCFWorkflow.build(
        code=codes["kcp"],
        structure=structure,
        pseudo_family=pseudo_family,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nbnd=nbnd,
        nspin=nspin,
        tot_magnetization=_coerce_optional_int(
            koopmans_input.calculator_parameters.tot_magnetization
        ),
        correction=correction,
        init_orbitals=workflow.init_orbitals,
        alpha_numsteps=workflow.alpha_numsteps,
        fix_spin_contamination=workflow.fix_spin_contamination,
        initial_alpha=initial_alpha,
        spin_polarized=workflow.spin == SpinType.COLLINEAR,
        orbital_groups_self_hartree_tol=self_hartree_tol,
        **extra_kwargs,
    )


def _band_range_complement(start: int, end: int, nbnd: int) -> str | None:
    """Return the wannier90 ``exclude_bands`` string for a [start, end] block."""
    parts = []
    if start > 1:
        parts.append(f"1-{start - 1}")
    if end < nbnd:
        parts.append(f"{end + 1}-{nbnd}")
    return ",".join(parts) if parts else None


def _derive_dscf_blocks(
    structure: orm.StructureData,
    projection_blocks: list,
    nocc: int,
    nbnd: int,
    spin_channel: Any,
) -> list:
    """Turn user projection blocks into DSCF wannierization blocks.

    Unlike the DFPT manifolds (one occupied + at most one empty block), the
    DSCF route wannierises every user block separately and merges them per
    (filling, spin) via merge_evc.x, so any number of blocks is allowed.
    Each block covers ``num_wann`` consecutive bands; a block straddling the
    occupied/empty boundary is an input error, and the occupied blocks must
    cover every occupied band (the folded ``evc_occupied`` files seed the
    complete occupied manifold of the supercell kcp.x run).
    """
    from aiida_koopmans.types import ExplicitProjectionBlock, SpinChannel
    from aiida_koopmans.workgraphs.dfpt import _projection_num_wann
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    if not projection_blocks:
        raise ValueError(
            "Wannier-function initialisation requires explicit projections in "
            "``calculator_parameters.w90.projections``."
        )

    suffix = f"_{spin_channel.value}" if spin_channel in (SpinChannel.UP, SpinChannel.DOWN) else ""
    blocks: list = []
    cursor = 0
    n_occ = n_emp = 0
    for block in projection_blocks:
        num_wann = sum(_projection_num_wann(structure, p) for p in block)
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
                exclude_bands=_band_range_complement(start, end, nbnd),
                projection_type=WannierProjectionType.ANALYTIC,
                projections=[f"{p.site}:{p.ang_mtm}" for p in block],
            )
        )
        cursor = end

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
    wannier_overrides = {
        key: {"pseudo_family": pseudo_family, "pw": {"parameters": parameters}}
        for key in ("scf", "nscf")
    }

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
    codes: dict[str, orm.AbstractCode],
) -> WorkGraph:
    """Build a workgraph for a singlepoint Koopmans calculation with DFPT screening.

    Assembles the full chain (scf + nscf → per-manifold wannierization →
    wann2kc → screen → ham) via ``aiida_koopmans.workgraphs.dfpt.SinglepointDFPT``.

    Spin regimes (``workflow.spin``): ``none`` runs the closed-shell chain;
    ``collinear`` fans the wannierization and the kcw.x chain out per spin
    channel (needs per-spin projections in ``w90.up`` / ``w90.down`` and a
    ``tot_magnetization``); ``non_collinear`` / ``spin_orbit`` run the spinor
    chain (all bands singly occupied, ``num_wann`` doubled).

    Remaining restrictions (mirroring the ``SinglepointDFPT`` scope):
    periodic, MLWF/projwf variational orbitals, and explicit projections
    forming exactly one occupied manifold block plus at most one empty block
    per spin channel (the legacy multi-block merge machinery is not yet
    ported).
    """
    from aiida_koopmans.types import SpinChannel
    from aiida_koopmans.workgraphs.dfpt import (
        SinglepointDFPT,
        derive_dfpt_manifolds,
        normalize_alpha_guess,
    )

    from koopmans.aiida.conversion import (
        get_pseudos_from_family,
        kpoints_input_to_kpoints_mesh,
        kpoints_input_to_kpoints_path,
    )
    from koopmans.input_file.workflow import VariationalOrbitalType

    workflow = koopmans_input.workflow

    if workflow.init_orbitals not in (
        VariationalOrbitalType.MLWFS,
        VariationalOrbitalType.PROJWFS,
    ):
        raise NotImplementedError(
            "DFPT screening is only ported for Wannier-function variational orbitals "
            "(init_orbitals = 'mlwfs' or 'projwfs'). The molecular kcw_at_ks path is "
            "not yet wired."
        )
    if getattr(koopmans_input.kpoints, "gamma_only", False):
        raise NotImplementedError(
            "Gamma-only DFPT (isolated systems) is not yet ported; provide a k-point grid."
        )
    if isinstance(workflow.eps_inf, str) and workflow.eps_inf != "auto":
        raise ValueError(
            f"eps_inf={workflow.eps_inf!r} is not understood: provide a numeric value "
            "or 'auto' (compute the dielectric constant with ph.x)."
        )

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

    # Electron count from the pseudopotential valences: fixes the size of the
    # occupied manifold (legacy: pseudopotentials.nelec_from_pseudos).
    pseudos = get_pseudos_from_family(pseudo_family, structure)
    nelec = round(sum(pseudos[site.kind_name].z_valence for site in structure.sites))

    nbnd = calc_params.nbnd if calc_params.nbnd is not None else calc_params.pw.system.nbnd
    nbnd = int(nbnd) if nbnd is not None else None

    if spin == SpinType.COLLINEAR:
        manifold_inputs = _collinear_dfpt_manifold_inputs(
            koopmans_input, structure, overrides, nelec, nbnd
        )
    else:
        spin_channel = SpinChannel.NONE if spin == SpinType.NONE else SpinChannel.SPINOR
        occ_block, emp_block, has_disentangle, n_orbitals = derive_dfpt_manifolds(
            structure=structure,
            projection_blocks=calc_params.wannier90.projections,
            nelec=nelec,
            nbnd=nbnd,
            spin_channel=spin_channel,
        )
        manifold_inputs = {
            "occ_block": occ_block,
            "emp_block": emp_block,
            "has_disentangle": has_disentangle,
            "alpha_guess": None
            if workflow.calculate_alpha
            else normalize_alpha_guess(workflow.alpha_guess, n_orbitals),
        }

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
    if workflow.eps_inf == "auto":
        codes.setdefault("ph", _load_code("ph", "ph.x"))

    return SinglepointDFPT.build(
        codes=codes,
        structure=structure,
        kpoints=kpoints_input_to_kpoints_mesh(koopmans_input.kpoints),
        kgrid=list(koopmans_input.kpoints.grid),
        bands_kpoints=bands_kpoints,
        pseudo_family=pseudo_family,
        overrides=overrides,
        # eps_inf is FloatGE1 | 'auto' | None after the guard above ('auto'
        # prepends the scf + ph.x dielectric chain inside SinglepointDFPT);
        # l_vcut is the Gygi-Baldereschi flag (None -> the periodic default, on).
        eps_inf=workflow.eps_inf,
        l_vcut=workflow.gb_correction,
        spin=spin,
        **manifold_inputs,
    )


def _collinear_dfpt_manifold_inputs(
    koopmans_input: KoopmansInput,
    structure: orm.StructureData,
    overrides: dict[str, Any],
    nelec: int,
    nbnd: int | None,
) -> dict[str, Any]:
    """Derive the per-spin-channel manifold inputs for a collinear DFPT run.

    Returns the ``SinglepointDFPT`` inputs describing both channels
    (``occ_block`` / ``emp_block`` / ``alpha_guess`` / ``has_disentangle``
    and their ``_down`` twins) from the per-spin projections in
    ``w90.up`` / ``w90.down`` and the per-channel occupations fixed by
    ``tot_magnetization``. Also forwards the magnetization into the scf /
    nscf PW SYSTEM overrides (mutated in place): the PW runs must see the
    physical magnetization — ``SinglepointDFPT`` only forces ``nspin=2``
    in this regime.
    """
    from aiida_koopmans.types import SpinChannel
    from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds, normalize_alpha_guess

    workflow = koopmans_input.workflow
    w90 = koopmans_input.calculator_parameters.wannier90
    magnetization = int(koopmans_input.calculator_parameters.tot_magnetization)
    if (nelec + magnetization) % 2:
        raise ValueError(
            f"nelec = {nelec} and tot_magnetization = {magnetization} do not give "
            "integer per-channel occupations."
        )
    for key in ("scf", "nscf"):
        overrides[key]["pw"]["parameters"].setdefault("SYSTEM", {})["tot_magnetization"] = (
            magnetization
        )

    inputs: dict[str, Any] = {}
    for channel, w90_channel, suffix in (
        (SpinChannel.UP, w90.up, ""),
        (SpinChannel.DOWN, w90.down, "_down"),
    ):
        sign = 1 if channel == SpinChannel.UP else -1
        occ_block, emp_block, has_disentangle, n_orbitals = derive_dfpt_manifolds(
            structure=structure,
            projection_blocks=w90_channel.projections,
            nelec=nelec,
            nbnd=nbnd,
            spin_channel=channel,
            nocc=(nelec + sign * magnetization) // 2,
        )
        inputs[f"occ_block{suffix}"] = occ_block
        inputs[f"emp_block{suffix}"] = emp_block
        inputs[f"has_disentangle{suffix}"] = has_disentangle
        if not workflow.calculate_alpha:
            inputs[f"alpha_guess{suffix}"] = normalize_alpha_guess(
                workflow.alpha_guess, n_orbitals, channel
            )
    return inputs


def _build_trajectory_workgraph(
    koopmans_input: KoopmansInput,
    codes: dict[str, orm.AbstractCode],
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

    from koopmans.aiida.setup import ensure_pseudo_family_installed

    workflow = koopmans_input.workflow

    if workflow.calculate_alpha and workflow.screening_method == CalculateScreeningMethod.DFPT:
        raise NotImplementedError(
            "The trajectory task only supports DSCF screening (kcp.x); DFPT screening "
            "is not ported for trajectories."
        )

    correction = workflow.correction
    supported = {Correction.KI, Correction.KIPZ}
    if correction not in supported:
        raise NotImplementedError(
            f"correction={correction.value!r} is not yet ported. "
            f"Supported: {sorted(c.value for c in supported)}."
        )

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
    pseudo_family = workflow.pseudo_library
    ensure_pseudo_family_installed(pseudo_family)

    ecutwfc, ecutrho, nbnd, nspin = _extract_kcp_scalar_inputs(koopmans_input)

    initial_alpha = (
        workflow.alpha_guess if isinstance(workflow.alpha_guess, float) else workflow.alpha_guess[0]
    )

    # The input schema cannot express multiple snapshots yet (legacy read them
    # from an ``atomic_positions: {snapshots: file.xyz}`` entry), so run the
    # single input structure as a one-snapshot trajectory.
    snapshots = {"snapshot_1": structure}

    return TrajectoryWorkflow.build(
        code=codes["kcp"],
        snapshots=snapshots,
        pseudo_family=pseudo_family,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nbnd=nbnd,
        nspin=nspin,
        tot_magnetization=_coerce_optional_int(
            koopmans_input.calculator_parameters.tot_magnetization
        ),
        correction=correction,
        init_orbitals=workflow.init_orbitals,
        alpha_numsteps=workflow.alpha_numsteps,
        fix_spin_contamination=workflow.fix_spin_contamination,
        initial_alpha=initial_alpha,
        spin_polarized=workflow.spin == SpinType.COLLINEAR,
        orbital_groups_self_hartree_tol=workflow.orbital_groups_self_hartree_tol,
        ml_mode=ml_mode,
        ml_model=ml_model,
        estimator=ml_config.estimator,
        descriptor=ml_config.descriptor,
        occ_and_emp_together=ml_config.occ_and_emp_together,
    )


def _ui_file_to_node(path: Any, keyword: str, description: str) -> orm.SinglefileData:
    """Load a file referenced by a ``calculator_parameters.ui`` path into a ``SinglefileData``.

    Raises:
        ValueError: If the path is unset or does not point to an existing file.
    """
    from pathlib import Path

    if path is None:
        raise ValueError(
            f"The UI task requires `calculator_parameters.ui.{keyword}` ({description})."
        )
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(
            f"`calculator_parameters.ui.{keyword}` points to `{path}`, which does not exist."
        )
    return orm.SinglefileData(resolved)


def _build_ui_workgraph(
    koopmans_input: KoopmansInput,
    codes: dict[str, orm.AbstractCode],
) -> WorkGraph:
    """Build a workgraph for the standalone unfold-and-interpolate (``ui``) task.

    Pure python post-processing — ``codes`` is empty. Consumes previously
    generated files referenced by paths in ``calculator_parameters.ui``:
    the Hamiltonian to interpolate (``kc_ham_file``), the Wannier90 output
    supplying centres and spreads (``<wannier90_seedname>.wout``), and —
    when ``smooth_int_factor`` asks for the smooth-interpolation method —
    the coarse and dense-grid DFT Hamiltonians. The band path comes from
    ``kpoints`` (optionally overridden by ``ui.kpath``); the DOS window
    and smearing from the ``plotting`` block.
    """
    from aiida_koopmans.workgraphs.ui import UnfoldAndInterpolateTask

    from koopmans.aiida.conversion import kpoints_input_to_kpoints_path

    ui_config = koopmans_input.calculator_parameters.ui

    kc_ham_file = _ui_file_to_node(
        ui_config.kc_ham_file, "kc_ham_file", "the Hamiltonian file to interpolate"
    )
    wannier90_wout = _ui_file_to_node(
        f"{ui_config.wannier90_seedname}.wout",
        "wannier90_seedname",
        "the Wannier90 output file providing the Wannier centres and spreads",
    )

    smooth_kwargs: dict[str, Any] = {}
    if ui_config.do_smooth_interpolation:
        smooth_kwargs["dft_ham_file"] = _ui_file_to_node(
            ui_config.dft_ham_file,
            "dft_ham_file",
            "the coarse DFT Hamiltonian, required for smooth interpolation",
        )
        smooth_kwargs["dft_smooth_ham_file"] = _ui_file_to_node(
            ui_config.dft_smooth_ham_file,
            "dft_smooth_ham_file",
            "the dense-grid DFT Hamiltonian, required for smooth interpolation",
        )

    structure = atoms_input_to_structure(koopmans_input.atoms)

    kpoints = koopmans_input.kpoints
    if ui_config.kpath is not None:
        if not isinstance(ui_config.kpath, str):
            raise NotImplementedError(
                "`calculator_parameters.ui.kpath` only supports high-symmetry-point "
                "strings (e.g. 'GXG'); explicit BandPath objects are not wired."
            )
        kpoints = kpoints.model_copy(update={"path": ui_config.kpath})
    kpath = kpoints_input_to_kpoints_path(kpoints, structure)

    return UnfoldAndInterpolateTask.build(
        kc_ham_file=kc_ham_file,
        wannier90_wout=wannier90_wout,
        structure=structure,
        kpath=kpath,
        kgrid=list(koopmans_input.kpoints.grid),
        do_map=ui_config.do_map,
        use_ws_distance=ui_config.use_ws_distance,
        w90_input_sc=ui_config.wannier90_input_sc or ui_config.wannier90_calc == "sc",
        do_dos=ui_config.do_dos,
        plotting=koopmans_input.plotting.model_dump(),
        **smooth_kwargs,
    )


def _extract_kcp_scalar_inputs(
    koopmans_input: KoopmansInput,
) -> tuple[float, float, int, int]:
    """Pull ``(ecutwfc, ecutrho, nbnd, nspin)`` out of the ``KoopmansInput``.

    Prefers the top-level ``calculator_parameters.{ecutwfc,nbnd}`` convenience
    fields; falls back to the ``kcp.system`` Pydantic block when they are unset.
    ``ecutrho`` has no top-level convenience field — read from ``kcp.system``
    and default to ``4 * ecutwfc`` when unset.
    """
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
    nbnd = int(nbnd_raw)

    # KI requires nspin=2 for per-spin orbital-dependent screening, regardless of
    # what ``spin`` says — closed-shell molecules still need two channels.
    nspin = 2

    return float(ecutwfc), float(ecutrho), nbnd, nspin


def _coerce_optional_int(value: float | None) -> int | None:
    """Return ``int(value)`` when value is given, else ``None``."""
    return int(value) if value is not None else None
