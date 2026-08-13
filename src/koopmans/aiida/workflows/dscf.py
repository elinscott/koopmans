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
)
from koopmans.aiida.workflows import (
    load_codes,
    reject_kpoint_overrides,
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
from koopmans.input_file.workflow import (
    CalculateScreeningMethod,
    Correction,
    VariationalOrbitalType,
)

if TYPE_CHECKING:
    from aiida import orm
    from aiida_koopmans.workgraphs.block_wannierize import WannierizeOverrides
    from aiida_workgraph import WorkGraph

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

KPOINT_OVERRIDES_ON_DSCF = {
    step: _KCP_TAKES_ONE_MESH.format(
        step=step,
        alternative=" Screening with `screening_method = 'dfpt'` gives the scf a mesh of its own.",
    )
    for step in ("scf", "nscf")
}

#: The same rejection without the DFPT alternative, which the trajectory task
#: does not offer.
KPOINT_OVERRIDES_ON_TRAJECTORY = {
    step: _KCP_TAKES_ONE_MESH.format(step=step, alternative="") for step in ("scf", "nscf")
}


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

    # Every NotRequired member of DscfCodes exists for the Wannier-seeded
    # initialisation, so that route turns them all on.
    codes = load_codes(DscfCodes, require=DscfCodes.__optional_keys__ if wannier_init else ())

    return KoopmansDSCFWorkflow.build(
        codes=codes,
        structure=structure,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        **inputs,
        **extra_kwargs,
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
    if nscf_nbnd < nbnd:
        raise ValueError(
            f"The nscf runs {nscf_nbnd} bands but the kcp.x steps need {nbnd} "
            "variational orbitals, which the Wannier functions cannot span. Raise "
            "``calculator_parameters.pw.system.nbnd`` to at least "
            f"{nbnd}."
        )

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
        nocc_up = (nelec + magnetization) // 2
        nocc_down = (nelec - magnetization) // 2
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
        "blocks": blocks,
        "kgrid": list(kpoints_input.grid),
        "kpoints": kpoints_input_to_kpoints_mesh(kpoints_input),
        "gamma_only": bool(getattr(kpoints_input, "gamma_only", False)),
        "wannier_overrides": wannier_overrides,
        "mp_correction": workflow.mp_correction,
        "eps_inf": workflow.eps_inf,
    }


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
        tot_magnetization=_coerce_optional_int(calc_params.tot_magnetization),
        correction=workflow.correction,
        init_orbitals=workflow.init_orbitals,
        alpha_numsteps=workflow.alpha_numsteps,
        fix_spin_contamination=workflow.fix_spin_contamination,
        initial_alpha=_initial_alpha_from_guess(workflow.alpha_guess),
        spin_polarized=workflow.spin == SpinType.COLLINEAR,
        orbital_groups_self_hartree_tol=grouping_tol(workflow),
    )


def _coerce_optional_int(value: float | None) -> int | None:
    """Return ``int(value)`` when value is given, else ``None``."""
    return int(value) if value is not None else None
