"""The DFPT (kcw.x) singlepoint route."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

from aiida_koopmans.spin import SpinChannel
from aiida_quantumespresso.common.types import SpinType

from koopmans.aiida.workflows import (
    collinear_magnetization,
    load_codes,
    name_run,
    pin_step_kpoints,
    prepare_common_inputs,
    reject_kpoint_overrides,
    require_configured_codes,
)
from koopmans.aiida.workflows.grouping import dfpt_grouping_tol
from koopmans.input_file.workflow import Correction, VariationalOrbitalType

if TYPE_CHECKING:
    from aiida import orm
    from aiida_koopmans.workgraphs.block_wannierize import WannierizeOverrides
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput


class DfptChainInputs(TypedDict):
    """Every ``SinglepointDFPTWorkflow.build`` keyword argument except ``codes``.

    Returned by :func:`assemble_dfpt_chain_inputs` alongside each spin
    channel's total Wannierized orbital count (which only the BSE route
    needs), shared by the plain DFPT singlepoint route and the BSE route,
    which composes the same ground-state/wannierization/screening chain in
    front of its own yambo steps. Named exactly after
    ``SinglepointDFPTWorkflow.build``'s own parameters so a caller building
    the plain DFPT chain can unpack it directly as ``**chain_inputs``.

    ``smooth_kpoints`` / ``smooth_mp_grid`` are absent unless the caller
    adds them (:func:`add_smooth_interpolation_inputs`): they shape a band
    structure, and only the plain DFPT singlepoint publishes one.
    """

    structure: orm.StructureData
    kpoints: orm.KpointsData
    scf_kpoints: orm.KpointsData | None
    bands_kpoints: orm.KpointsData | None
    smooth_kpoints: NotRequired[orm.KpointsData]
    smooth_mp_grid: NotRequired[list[int]]
    eps_kpoints: orm.KpointsData | None
    pseudo_family: str
    overrides: dict[str, Any]
    eps_inf: float | str | None
    l_vcut: bool | None
    spin: SpinType
    manifolds: dict[str, Any]
    group_orbitals_tol: float | None
    kcw_overrides: dict[str, Any] | None
    parallelization: dict[str, Any] | None


def assemble_dfpt_chain_inputs(
    koopmans_input: KoopmansInput,
) -> tuple[DfptChainInputs, dict[str, int]]:
    """Validate and assemble every ``SinglepointDFPTWorkflow`` input except ``codes``.

    Shared by :func:`build_singlepoint_dfpt_workgraph` and the BSE route
    (:mod:`koopmans.aiida.workflows.bse`), which composes the same chain in
    front of its own yambo steps: both need the same ground-state,
    wannierization and screening inputs, validated the same way.

    Returns:
        The ``SinglepointDFPTWorkflow.build`` keyword arguments, and each
        spin channel's total Wannierized orbital count (``"none"`` /
        ``"up"`` / ``"down"``) — the highest kcw.x ``ham`` band a route
        reading that channel's eigenvalues may ask for. The plain DFPT
        route has no use for the counts; the BSE route bounds
        ``yambo.BSEBands`` against them.
    """
    from koopmans.aiida.conversion import (
        eps_inf_kpoints_mesh,
        get_pseudos_from_family,
        input_to_kcw_overrides,
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

    # Checked last among the pure-Python guards, after every workflow-scope
    # rejection above (correction, init_orbitals, gamma_only, spin): an
    # explicit override paired with one of those should surface the scope
    # blocker first, not send the reader to fix the override and only then
    # learn the run is unsupported regardless.
    reject_kpoint_overrides(
        koopmans_input,
        {
            "wannier90": "`kpoints.overrides.wannier90.path_density` is not yet wired "
            "into the DFPT route: its own wannierization step interpolates along "
            "`kpoints.path` at the top-level `kpoints.path_density`, with no socket "
            "of its own yet for a denser interpolation."
        },
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
        manifolds, manifold_band_counts = _collinear_dfpt_manifolds(
            koopmans_input, structure, overrides, nelec, nbnd
        )
    else:
        manifolds, n_orbitals = _single_channel_dfpt_manifolds(
            koopmans_input, structure, nelec, nbnd, spin
        )
        manifold_band_counts = {SpinChannel.NONE.value: n_orbitals}

    bands_kpoints = kpoints_input_to_interpolation_path(koopmans_input.kpoints, structure)

    # The nscf mesh is the one the Wannier functions and kcw.x count in
    # (``CONTROL.mp1-3``); the scf may converge the density on another.
    nscf_mesh = step_kpoints_mesh(koopmans_input.kpoints, "nscf")

    kcw_overrides = input_to_kcw_overrides(koopmans_input)

    # Only when the factor densifies: at the default (1, 1, 1) the
    # dielectric scf must keep sampling exactly what it did before this
    # keyword existed (the chain's own scf), not a freshly built mesh that
    # merely happens to equal it.
    eps_inf_factor = koopmans_input.kpoints.eps_inf_factor
    eps_kpoints = (
        eps_inf_kpoints_mesh(koopmans_input.kpoints, eps_inf_factor)
        if any(f > 1 for f in eps_inf_factor)
        else None
    )

    return (
        DfptChainInputs(
            structure=structure,
            kpoints=nscf_mesh,
            scf_kpoints=pin_step_kpoints(overrides, "scf", koopmans_input),
            bands_kpoints=bands_kpoints,
            eps_kpoints=eps_kpoints,
            pseudo_family=pseudo_family,
            overrides=overrides,
            eps_inf=eps_inf,
            l_vcut=workflow.gb_correction,
            spin=spin,
            manifolds=manifolds,
            group_orbitals_tol=group_orbitals_tol,
            kcw_overrides=kcw_overrides or None,
            parallelization=(
                koopmans_input.parallelization.as_mapping(koopmans_input.computer) or None
            ),
        ),
        manifold_band_counts,
    )


def build_singlepoint_dfpt_workgraph(koopmans_input: KoopmansInput) -> WorkGraph:
    """Build a workgraph for a singlepoint Koopmans calculation with DFPT screening.

    Assembles the full sequence (scf + nscf → per-manifold wannierization →
    wann2kc → screen → ham) via ``aiida_koopmans.workgraphs.dfpt.SinglepointDFPTWorkflow``.

    Spin regimes (``workflow.spin``): ``none`` runs the closed-shell
    sequence; ``collinear`` fans the wannierization and the kcw.x steps out
    per spin channel (needs per-spin projections in ``w90.up`` / ``w90.down``
    and a ``tot_magnetization``); ``non_collinear`` / ``spin_orbit`` run the
    spinor variant (all bands singly occupied, ``num_wann`` doubled).

    A ``kpoints.path`` in the input reaches the kcw.x ham step as its bands
    path, so the run also emits the Koopmans band structure interpolated
    along it. A ``kpoints.smooth_interpolation_factor`` above 1 interpolates
    that band structure with the DFT Hamiltonian from a mesh ``factor``
    times denser instead of kcw.x's own coarse-grid one
    (:func:`add_smooth_interpolation_inputs`).

    Remaining restrictions (mirroring the ``SinglepointDFPTWorkflow`` scope):
    periodic, MLWF/projwf variational orbitals, and explicit projections.
    A manifold may span several projection blocks; their Wannier products
    are merged back into one file set before kcw.x consumes them.
    """
    from aiida_koopmans.workgraphs.dfpt import DfptCodes, SinglepointDFPTWorkflow

    chain_inputs, _manifold_band_counts = assemble_dfpt_chain_inputs(koopmans_input)
    add_smooth_interpolation_inputs(chain_inputs, koopmans_input)

    # load_codes loads every configured member of DfptCodes. ph.x is only
    # actually needed for the `eps_inf: auto` dielectric pre-computation,
    # and projwfc only for the quality-check projected DOS; whether either
    # runs, and whether a missing code the run does need is fatal, is now
    # the graph's own structural requirement — checked at graph validation,
    # not here. require_configured_codes only ever looks at pw/kcw (the
    # required members): it has no notion of eps_inf, so ph never gets
    # demanded here.
    codes = load_codes(DfptCodes, koopmans_input.computer.name)
    require_configured_codes(DfptCodes, codes, koopmans_input.computer.name)

    return name_run(
        SinglepointDFPTWorkflow.build(codes=codes, **chain_inputs),
        "Koopmans DFPT",
    )


def add_smooth_interpolation_inputs(
    chain_inputs: DfptChainInputs, koopmans_input: KoopmansInput
) -> None:
    """Add the denser mesh the smooth-interpolation method Wannierizes, in place.

    kcw.x interpolates the whole Koopmans Hamiltonian from the grid the
    Wannier functions were built on, DFT part and all, so on a coarse grid
    its band structure is only as good as a Wannier interpolation of the
    DFT bands. A ``kpoints.smooth_interpolation_factor`` above 1 replaces
    that DFT part with the same quantity from a Wannierization on a mesh
    ``factor`` times denser, stated as the explicit k-point list and its
    Monkhorst-Pack dimensions. The neutral default adds nothing.

    Args:
        chain_inputs: The assembled chain inputs, mutated in place.
        koopmans_input: The parsed koopmans input.

    Raises:
        ValueError: If the factor shapes an interpolation the input does
            not ask for.
    """
    from koopmans.aiida.conversion import smooth_grid, smooth_kpoints_mesh

    kpoints_input = koopmans_input.kpoints
    factor = kpoints_input.smooth_interpolation_factor
    if not any(f > 1 for f in factor):
        return
    if chain_inputs["bands_kpoints"] is None:
        raise ValueError(
            "`kpoints.smooth_interpolation_factor` shapes the band structure "
            "interpolation, and this input asks for none. Add the path to interpolate "
            "along as `kpoints: {path: ...}`, or restore the default `[1, 1, 1]`."
        )
    chain_inputs["smooth_kpoints"] = smooth_kpoints_mesh(kpoints_input, factor)
    chain_inputs["smooth_mp_grid"] = smooth_grid(kpoints_input, factor)


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
) -> tuple[dict[str, Any], int]:
    """Derive the single-channel ``manifolds`` input for an unpolarized or spinor DFPT run.

    Both regimes run one kcw.x sequence keyed ``"none"``; the spinor case
    differs only in the manifold derivation (all bands singly occupied,
    ``num_wann`` doubled). Returns the manifold dict alongside its total
    Wannierized orbital count (occupied plus empty, if any) — the highest
    kcw.x ``ham`` band this channel's eigenvalues carry.
    """
    from aiida_koopmans.projections import ProjectionBlock, derive_dfpt_manifolds
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
    return {SpinChannel.NONE.value: manifold}, n_orbitals


def _collinear_dfpt_manifolds(
    koopmans_input: KoopmansInput,
    structure: orm.StructureData,
    overrides: dict[str, Any],
    nelec: int,
    nbnd: int | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Derive the per-spin-channel ``manifolds`` input for a collinear DFPT run.

    Returns the ``SinglepointDFPTWorkflow`` ``manifolds`` dict — one
    ``ManifoldBlocks`` per spin channel, keyed ``"up"`` / ``"down"`` — from
    the per-spin projections in ``w90.up`` / ``w90.down`` and the
    per-channel occupations fixed by ``tot_magnetization``, alongside each
    channel's total Wannierized orbital count. Also forwards the
    magnetization into the scf / nscf PW SYSTEM overrides (mutated in
    place): the PW runs must see the physical magnetization —
    ``SinglepointDFPTWorkflow`` only forces ``nspin=2`` in this regime.
    """
    from aiida_koopmans.projections import ProjectionBlock, derive_dfpt_manifolds
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
    band_counts: dict[str, int] = {}
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
        band_counts[channel.value] = n_orbitals
    return manifolds, band_counts
