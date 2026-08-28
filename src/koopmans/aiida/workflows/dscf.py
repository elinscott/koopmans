"""The DSCF (kcp.x) singlepoint route."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from aiida_koopmans.projections import validate_projection_block_sequence
from aiida_koopmans.spin import SpinChannel
from aiida_quantumespresso.common.types import SpinType

from koopmans.aiida.conversion import (
    NORM_CONSERVING_DUAL,
    atoms_input_to_structure,
    input_to_pw_parameters,
    kpoints_input_to_interpolation_path,
)
from koopmans.aiida.workflows import (
    collinear_magnetization,
    load_codes,
    name_run,
    optional_magnetization,
    reject_kpoint_overrides,
    require_configured_codes,
    require_cutoffs_for_family,
)
from koopmans.aiida.workflows.blocks import (
    create_explicit_blocks,
    validate_blocks_cover_all_occ_bands,
    validate_blocks_separate_occ_and_emp,
)
from koopmans.aiida.workflows.dfpt import build_singlepoint_dfpt_workgraph
from koopmans.aiida.workflows.grouping import grouping_tol
from koopmans.aiida.workflows.projectors import reject_unwired_external_projectors
from koopmans.input_file.unfold_and_interpolate import UnfoldAndInterpolateConfig
from koopmans.input_file.workflow import (
    CalculateScreeningMethod,
    Correction,
    VariationalOrbitalType,
)

if TYPE_CHECKING:
    from aiida import orm
    from aiida_koopmans.workgraphs.block_wannierize import WannierizeOverrides
    from aiida_workgraph import WorkGraph
    from wannier90_input.models.parameters import Projection

    from koopmans.input_file import KoopmansInput


#: Why no kcp.x route carries a second mesh: ``MlwfInitialization`` takes one
#: ``kpoints``, which its scf samples and its supercell is folded from, and the
#: molecular route runs no pw.x step at all. A scope boundary, not physics.
_KCP_TAKES_ONE_MESH = (
    "`kpoints.overrides.{step}` cannot take effect on the kcp.x route: every step it "
    "runs samples the one mesh `kpoints.grid` describes, which is also the supercell "
    "its kcp.x steps fold to, and no step there takes a mesh of its own. Set "
    "`kpoints.grid`.{alternative}"
)

#: The Wannier-seeded route folds Wannier functions to a supercell for
#: kcp.x initialisation, not an interpolated band structure along a path,
#: so a wannier90 interpolation density has nothing to describe.
_KCP_HAS_NO_INTERPOLATION = (
    "`kpoints.overrides.wannier90.path_density` cannot take effect on the kcp.x route: "
    "its Wannier initialisation folds Wannier functions to a supercell off the one mesh "
    "`kpoints.grid` describes, not an interpolated band structure along a path."
)

KPOINT_OVERRIDES_ON_DSCF = {
    step: _KCP_TAKES_ONE_MESH.format(
        step=step,
        alternative=" Screening with `screening_method = 'dfpt'` gives the scf a mesh of its own.",
    )
    for step in ("scf", "nscf")
}
KPOINT_OVERRIDES_ON_DSCF["wannier90"] = _KCP_HAS_NO_INTERPOLATION

#: The same rejection without the DFPT alternative, which the trajectory task
#: does not offer.
KPOINT_OVERRIDES_ON_TRAJECTORY = {
    step: _KCP_TAKES_ONE_MESH.format(step=step, alternative="") for step in ("scf", "nscf")
}
KPOINT_OVERRIDES_ON_TRAJECTORY["wannier90"] = _KCP_HAS_NO_INTERPOLATION


def build_singlepoint_workgraph(koopmans_input: KoopmansInput) -> WorkGraph:
    """Build a workgraph for a singlepoint Koopmans calculation.

    Dispatches on ``workflow.screening_method`` first (DSCF vs DFPT), then on
    ``workflow.correction``:

    - DSCF + ``KI``/``KIPZ`` → ``KoopmansDSCFWorkflow`` (kcp.x)
    - DFPT + ``KI`` → ``build_singlepoint_dfpt_workgraph`` (kcw.x; KI only)
    - anything else → ``NotImplementedError``
    """
    from aiida_koopmans.workgraphs.kcp import DscfCodes, KoopmansDSCFWorkflow

    from koopmans.aiida.setup.pseudos import ensure_pseudo_family_installed

    workflow = koopmans_input.workflow

    reject_unwired_external_projectors(koopmans_input, "singlepoint")

    # DFPT routes on the screening method alone: calculate_alpha = False is
    # the alpha_guess path inside the DFPT builder (screen step skipped),
    # not a reason to fall through to the kcp.x/DSCF branch.
    if workflow.screening_method == CalculateScreeningMethod.DFPT:
        return build_singlepoint_dfpt_workgraph(koopmans_input)

    reject_kpoint_overrides(koopmans_input, KPOINT_OVERRIDES_ON_DSCF)
    require_supported_correction(workflow.correction)

    if workflow.spin in (SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT):
        raise NotImplementedError(
            f"spin={workflow.spin.value!r} is not supported by the DSCF (kcp.x) stream: "
            "kcp.x has no noncollinear mode. Use screening_method='dfpt'."
        )

    structure = atoms_input_to_structure(koopmans_input.atoms)
    ensure_pseudo_family_installed(workflow.pseudo_library)

    inputs = kcp_dscf_inputs(koopmans_input)

    wannier_init = workflow.init_orbitals in (
        VariationalOrbitalType.MLWFS,
        VariationalOrbitalType.PROJWFS,
    )
    extra_kwargs: dict[str, Any] = {}
    if wannier_init:
        extra_kwargs = dscf_wannier_init_inputs(koopmans_input, structure, inputs["nbnd"])
        # On this route the input file's nbnd sizes the pw.x runs — the
        # states the blocks disentangle among — while kcp.x takes one
        # variational orbital per Wannier function the projections
        # describe. They are different numbers and both are needed.
        inputs["nbnd"] = int(extra_kwargs.pop("nbnd"))
    extra_kwargs.update(band_interpolation_inputs(koopmans_input, structure))

    # load_codes loads every configured member of DscfCodes. Every
    # NotRequired member exists for the Wannier-seeded initialisation;
    # whether the route needs them, and whether a missing one is fatal, is
    # the graph's own structural requirement. KoopmansDSCFWorkflow binds
    # kcp eagerly (aiida-koopmans#90: a deliberate, permanent choice, not
    # a follow-up); the pre-flight catches a missing kcp before that bare
    # subscript can raise a bare KeyError.
    codes = load_codes(DscfCodes)
    require_configured_codes(DscfCodes, codes)

    return name_run(
        KoopmansDSCFWorkflow.build(
            codes=codes,
            structure=structure,
            parallelization=koopmans_input.parallelization.as_mapping() or None,
            **inputs,
            **extra_kwargs,
        ),
        "Koopmans ΔSCF",
    )


def _kcp_nbnd_from_projections(
    structure: orm.StructureData,
    projection_blocks: list[list[Projection]],
    num_occ_bands: int,
    spin_label: str = "",
) -> int:
    """Return kcp.x's orbital count: one variational orbital per Wannier function.

    The merged evc_occupied/evc_empty files that seed the supercell kcp.x
    run carry one orbital per projected Wannier function, so the
    projections fix the count. They must cover the whole occupied
    manifold; the empty ones they add on top become kcp.x's empty states.

    Raises:
        ValueError: If the projections leave an occupied band unspanned.
    """
    from aiida_koopmans.projections import projection_num_wann

    nwann = int(
        sum(projection_num_wann(structure, p) for block in projection_blocks for p in block)
    )
    if nwann < num_occ_bands:
        raise ValueError(
            f"The {spin_label}projections in `calculator_parameters.w90.projections` "
            f"describe only {nwann} Wannier functions, fewer than the {num_occ_bands} "
            f"{spin_label}occupied bands. Every occupied band needs a Wannier function to "
            "seed the supercell kcp.x initialisation; add projections covering the whole "
            "occupied manifold."
        )
    return nwann


def _require_explicit_kcp_nbnd_matches(
    koopmans_input: KoopmansInput, nwann: int, spin_label: str = ""
) -> None:
    """Reject a hand-written kcp.x ``nbnd`` the projections contradict.

    On the Wannier route the orbital count is derived, so this fires only
    when the user states one themselves.
    """
    stated = koopmans_input.calculator_parameters.kcp.system.nbnd
    if stated is None or int(stated) == nwann:
        return
    raise ValueError(
        f"`calculator_parameters.kcp.system.nbnd` = {int(stated)} is inconsistent with the "
        f"{spin_label}projections in `calculator_parameters.w90.projections`, which describe "
        f"{nwann} Wannier functions — one kcp.x variational orbital each. Set it to {nwann}, "
        "drop it (the projections fix it on this route), or add or remove projections to "
        "match."
    )


def _require_nscf_covers_nbnd(nscf_nbnd: int, nbnd: int) -> None:
    """Reject an nscf band count too small to feed kcp.x's ``nbnd`` orbitals."""
    if nscf_nbnd < nbnd:
        raise ValueError(
            f"The nscf runs {nscf_nbnd} bands but the kcp.x steps need {nbnd} "
            "variational orbitals, which the Wannier functions cannot span. Raise "
            "``calculator_parameters.pw.system.nbnd`` to at least "
            f"{nbnd}."
        )


def dscf_wannier_init_inputs(
    koopmans_input: KoopmansInput,
    structure: orm.StructureData,
    nbnd: int,
) -> dict[str, Any]:
    """Assemble the extra ``KoopmansDSCFWorkflow`` inputs for the Wannier route.

    Covers the periodic mlwfs/projwfs initialisation: the projection blocks
    (primitive band indices; per spin channel when ``spin='collinear'``),
    the k-mesh, and the Makov-Payne knobs. The molecular/kohn-sham route
    needs none of this; the wannierize + fold-to-supercell codes ride the
    caller's ``DscfCodes`` namespace.

    Args:
        koopmans_input: The parsed input file.
        structure: The primitive cell.
        nbnd: The band count the pw.x steps run — the states the blocks
            disentangle among.

    Returns:
        The extra workflow kwargs, ``nbnd`` among them: on this route it
        is kcp.x's own orbital count, derived from the projections rather
        than taken from the input file.
    """
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

    parameters = input_to_pw_parameters(koopmans_input)
    # A block's band window indexes the *nscf* bands: ``num_bands`` counts what
    # wannier90 reads out of the mmn, and ``exclude_bands`` must name every
    # nscf band the block does not use. Sizing them by the kcp.x orbital count
    # instead leaves the bands above it neither included nor excluded, and
    # wannier90 rejects the mmn it is then handed.
    nscf_nbnd = int(parameters.get("SYSTEM", {}).get("nbnd") or nbnd)

    if workflow.spin == SpinType.COLLINEAR:
        w90 = calc_params.wannier90
        if w90.up is None or w90.down is None:
            raise ValueError(
                "spin='collinear' Wannier initialisation needs per-spin projections: set "
                "``calculator_parameters.w90.up.projections`` and "
                "``calculator_parameters.w90.down.projections``."
            )
        magnetization = collinear_magnetization(koopmans_input)
        if (nelec + magnetization) % 2:
            raise ValueError(
                f"nelec = {nelec} and tot_magnetization = {magnetization} do not give "
                "integer per-channel occupations."
            )
        # The scf and nscf feeding the Wannierisation run at nspin = 2 with
        # fixed occupations, which pw.x refuses unless it is told how to
        # split the electrons between the two channels.
        parameters.setdefault("SYSTEM", {})["tot_magnetization"] = magnetization
        nocc_up = (nelec + magnetization) // 2
        nocc_down = (nelec - magnetization) // 2
        # kcp.x takes one variational orbital per projected Wannier
        # function: the merged evc_occupied/evc_empty files carry exactly
        # those. Both channels feed one kcp.x run, so their counts must
        # agree.
        kcp_nbnd = _kcp_nbnd_from_projections(structure, w90.up.projections, nocc_up, "spin up ")
        nwann_down = _kcp_nbnd_from_projections(
            structure, w90.down.projections, nocc_down, "spin down "
        )
        if nwann_down != kcp_nbnd:
            raise ValueError(
                f"The spin up projections describe {kcp_nbnd} Wannier functions and the "
                f"spin down ones {nwann_down}. Both channels share one kcp.x orbital "
                "count, so `calculator_parameters.w90.up.projections` and "
                "`calculator_parameters.w90.down.projections` must describe equally many."
            )
        _require_explicit_kcp_nbnd_matches(koopmans_input, kcp_nbnd)
        _require_nscf_covers_nbnd(nscf_nbnd, kcp_nbnd)
        up_blocks = create_explicit_blocks(
            structure, w90.up.projections, nscf_nbnd, nocc_up, SpinChannel.UP
        )
        validate_projection_block_sequence(up_blocks)
        validate_blocks_separate_occ_and_emp(up_blocks, nocc_up)
        validate_blocks_cover_all_occ_bands(up_blocks, nocc_up)
        down_blocks = create_explicit_blocks(
            structure, w90.down.projections, nscf_nbnd, nocc_down, SpinChannel.DOWN
        )
        validate_projection_block_sequence(down_blocks)
        validate_blocks_separate_occ_and_emp(down_blocks, nocc_down)
        validate_blocks_cover_all_occ_bands(down_blocks, nocc_down)
        blocks = up_blocks + down_blocks
    else:
        if nelec % 2:
            raise ValueError(
                f"Odd electron count ({nelec}) requires spin='collinear' for the "
                "Wannier-initialised DSCF route."
            )
        nocc = nelec // 2
        kcp_nbnd = _kcp_nbnd_from_projections(structure, calc_params.wannier90.projections, nocc)
        _require_explicit_kcp_nbnd_matches(koopmans_input, kcp_nbnd)
        _require_nscf_covers_nbnd(nscf_nbnd, kcp_nbnd)
        blocks = create_explicit_blocks(
            structure, calc_params.wannier90.projections, nscf_nbnd, nocc, SpinChannel.NONE
        )
        validate_projection_block_sequence(blocks)
        validate_blocks_separate_occ_and_emp(blocks, nocc)
        validate_blocks_cover_all_occ_bands(blocks, nocc)

    # The DSCF route never calls ``prepare_common_inputs``, so the family
    # checks reach its pw steps only from here.
    from koopmans.aiida.setup.pseudos import require_norm_conserving_family

    require_norm_conserving_family(pseudo_family, structure)
    require_cutoffs_for_family(pseudo_family, parameters)
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

    return {
        "nbnd": kcp_nbnd,
        "blocks": blocks,
        "kgrid": list(kpoints_input.grid),
        "kpoints": kpoints_input_to_kpoints_mesh(kpoints_input),
        "gamma_only": bool(getattr(kpoints_input, "gamma_only", False)),
        "wannier_overrides": wannier_overrides,
        "mp_correction": workflow.mp_correction,
        "eps_inf": workflow.eps_inf,
    }


def band_interpolation_inputs(
    koopmans_input: KoopmansInput,
    structure: orm.StructureData,
) -> dict[str, Any]:
    """Assemble the unfold-and-interpolate inputs, or none when no path is named.

    A ΔSCF run computes on a Γ-point supercell, so its band structure has
    to be recovered by unfolding the Koopmans Hamiltonian in the Wannier
    basis and interpolating it along ``kpoints.path``. An input naming no
    path asks for no band structure, and must then leave
    ``unfold_and_interpolate`` at its defaults.

    Raises:
        ValueError: If the input shapes an interpolation it does not ask for.
    """
    kpath = kpoints_input_to_interpolation_path(koopmans_input.kpoints, structure)
    settings = koopmans_input.calculator_parameters.unfold_and_interpolate
    if kpath is None:
        if settings.model_dump() != UnfoldAndInterpolateConfig().model_dump():
            raise ValueError(
                "`calculator_parameters.unfold_and_interpolate` shapes the band "
                "structure interpolation, and this input asks for none. Add the path "
                "to interpolate along as `kpoints: {path: ...}`, or restore the "
                "block's defaults."
            )
        return {}
    # The DOS keeps the interpolation's own smearing and window: the input
    # file has no block naming them.
    return {"kpath": kpath, "unfold_and_interpolate": settings.model_dump()}


def require_supported_correction(correction: Correction) -> None:
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


def kcp_dscf_inputs(koopmans_input: KoopmansInput) -> _KcpDscfInputs:
    """Assemble the scalar kwargs shared by the kcp.x DSCF builders.

    ``ecutwfc`` comes from ``calculator_parameters.ecutwfc``, with ``ecutrho``
    derived at :data:`NORM_CONSERVING_DUAL` times it; ``nbnd`` prefers the
    top-level ``calculator_parameters`` convenience field and falls back to
    the ``kcp.system`` Pydantic block.
    """
    workflow = koopmans_input.workflow
    calc_params = koopmans_input.calculator_parameters
    kcp_system = calc_params.kcp.system

    ecutwfc = calc_params.ecutwfc
    if not ecutwfc:
        raise ValueError(
            "ecutwfc is required for a Koopmans singlepoint calculation. Set "
            "``calculator_parameters.ecutwfc``."
        )

    ecutrho = NORM_CONSERVING_DUAL * ecutwfc

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
        tot_magnetization=optional_magnetization(koopmans_input),
        correction=workflow.correction,
        init_orbitals=workflow.init_orbitals,
        alpha_numsteps=workflow.alpha_numsteps,
        fix_spin_contamination=workflow.fix_spin_contamination,
        initial_alpha=_initial_alpha_from_guess(workflow.alpha_guess),
        spin_polarized=workflow.spin == SpinType.COLLINEAR,
        orbital_groups_self_hartree_tol=grouping_tol(workflow),
    )
