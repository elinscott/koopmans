"""The DFPT (kcw.x) singlepoint route."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from aiida_quantumespresso.common.types import SpinType

from koopmans.aiida.workflows import (
    collinear_magnetization,
    load_codes,
    pin_step_kpoints,
    prepare_common_inputs,
    require_configured_codes,
)
from koopmans.aiida.workflows.grouping import dfpt_grouping_tol
from koopmans.input_file.workflow import Correction, VariationalOrbitalType

if TYPE_CHECKING:
    from aiida import orm
    from aiida_koopmans.workgraphs.block_wannierize import WannierizeOverrides
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput


def build_singlepoint_dfpt_workgraph(koopmans_input: KoopmansInput) -> WorkGraph:
    """Build a workgraph for a singlepoint Koopmans calculation with DFPT screening.

    Assembles the full sequence (scf + nscf → per-manifold wannierization →
    wann2kc → screen → ham) via ``aiida_koopmans.workgraphs.dfpt.SinglepointDFPTWorkflow``.

    Spin regimes (``workflow.spin``): ``none`` runs the closed-shell
    sequence; ``collinear`` fans the wannierization and the kcw.x steps out
    per spin channel (needs per-spin projections in ``w90.up`` / ``w90.down``
    and a ``tot_magnetization``); ``non_collinear`` / ``spin_orbit`` run the
    spinor variant (all bands singly occupied, ``num_wann`` doubled).

    Remaining restrictions (mirroring the ``SinglepointDFPTWorkflow`` scope):
    periodic, MLWF/projwf variational orbitals, and explicit projections.
    A manifold may span several projection blocks; their Wannier products
    are merged back into one file set before kcw.x consumes them.
    """
    from aiida_koopmans.workgraphs.dfpt import DfptCodes, SinglepointDFPTWorkflow

    from koopmans.aiida.conversion import (
        get_pseudos_from_family,
        kpoints_input_to_interpolation_path,
        step_kpoints_mesh,
    )

    workflow = koopmans_input.workflow

    group_orbitals_tol = dfpt_grouping_tol(workflow)
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

    structure, pseudo_family, overrides = prepare_common_inputs(koopmans_input, ["scf", "nscf"])

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

    bands_kpoints = kpoints_input_to_interpolation_path(koopmans_input.kpoints, structure)

    # load_codes loads every configured member of DfptCodes. ph.x is only
    # actually needed for the `eps_inf: auto` dielectric pre-computation,
    # and projwfc only for the quality-check projected DOS; whether either
    # runs, and whether a missing code the run does need is fatal, is now
    # the graph's own structural requirement — checked at graph validation,
    # not here. require_configured_codes only ever looks at pw/kcw (the
    # required members): it has no notion of eps_inf, so ph never gets
    # demanded here.
    codes = load_codes(DfptCodes)
    require_configured_codes(DfptCodes, codes)

    # The nscf mesh is the one the Wannier functions and kcw.x count in
    # (``CONTROL.mp1-3``); the scf may converge the density on another.
    nscf_mesh = step_kpoints_mesh(koopmans_input.kpoints, "nscf")

    return SinglepointDFPTWorkflow.build(
        codes=codes,
        structure=structure,
        kpoints=nscf_mesh,
        scf_kpoints=pin_step_kpoints(overrides, "scf", koopmans_input),
        bands_kpoints=bands_kpoints,
        pseudo_family=pseudo_family,
        overrides=overrides,
        # 'auto' prepends the scf + ph.x dielectric steps inside
        # SinglepointDFPT; l_vcut is the Gygi-Baldereschi flag (None -> the
        # periodic default, on).
        eps_inf=eps_inf,
        l_vcut=workflow.gb_correction,
        spin=spin,
        manifolds=manifolds,
        group_orbitals_tol=group_orbitals_tol,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
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

    Both regimes run one kcw.x sequence keyed ``"none"``; the spinor case
    differs only in the manifold derivation (all bands singly occupied,
    ``num_wann`` doubled).
    """
    from aiida_koopmans.projections import ProjectionBlock, derive_dfpt_manifolds
    from aiida_koopmans.spin import SpinChannel
    from aiida_koopmans.workgraphs.dfpt import ManifoldBlocks, normalize_alpha_guess

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
    from aiida_koopmans.projections import ProjectionBlock, derive_dfpt_manifolds
    from aiida_koopmans.spin import SpinChannel
    from aiida_koopmans.workgraphs.dfpt import ManifoldBlocks, normalize_alpha_guess

    workflow = koopmans_input.workflow
    w90 = koopmans_input.calculator_parameters.wannier90
    if w90.up is None or w90.down is None:
        # Already validated by build_singlepoint_dfpt_workgraph; re-checked
        # here so the collinear helper narrows its own inputs.
        raise ValueError(
            "spin='collinear' DFPT screening needs per-spin projections "
            "(``w90.up`` / ``w90.down``)."
        )
    magnetization = collinear_magnetization(koopmans_input)
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
