"""Workflow building logic for koopmans AiiDA integration.

This module handles selecting and constructing the appropriate AiiDA workgraph
based on the task specified in a KoopmansInput.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, TypedDict, cast

from aiida import orm
from aiida_koopmans.ml import MLDescriptor, MLMode
from aiida_koopmans.projections import validate_projection_block_sequence
from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.workgraphs import Codes
from aiida_quantumespresso.common.types import SpinType

from koopmans.aiida.conversion import (
    atoms_input_to_structure,
    atoms_input_to_structures,
    code_parallelization,
    input_to_pw_parameters,
)
from koopmans.aiida.workflows.blocks import (
    _create_automatic_blocks,
    _create_explicit_blocks,
    _validate_blocks_cover_all_occ_bands,
    _validate_blocks_separate_occ_and_emp,
)
from koopmans.aiida.workflows.projectors import (
    _load_external_projectors,
    _reject_unwired_external_projectors,
)
from koopmans.input_file.workflow import (
    CalculateScreeningMethod,
    Correction,
    GroupOrbitalsBy,
    Task,
    VariationalOrbitalType,
)

if TYPE_CHECKING:
    from aiida_koopmans.workgraphs.block_wannierize import WannierizeOverrides
    from aiida_workgraph import WorkGraph
    from wannier90_input.models.parameters import Projection

    from koopmans.input_file import KoopmansInput
    from koopmans.input_file.ml import MLConfig
    from koopmans.input_file.workflow import WorkflowConfig


#: Raised wherever a Wannierization is asked for and nothing says what to
#: Wannierize. Both routes that need projections reach this state, so both
#: say the same thing.
_NO_PROJECTIONS_PROVIDED_MESSAGE = (
    "Nothing defines the Wannier projections. Either (a) set "
    "`workflow.auto_projections` to `True` to derive them from the "
    "pseudopotentials, or from external projector files by also pointing "
    "`pw2wannier90.atom_proj_ext` at them; or (b) provide explicit projections "
    "in `calculator_parameters.w90.projections`."
)


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

    if koopmans_input.workflow.auto_projections and task != Task.WANNIERIZE:
        raise NotImplementedError(
            f"`workflow.auto_projections` is not wired into the {task.value} route; "
            "automatic projections are currently supported by the `wannierize` task only."
        )

    ml_config = koopmans_input.ml
    if ml_config.mode != MLMode.NONE and task != Task.TRAJECTORY:
        raise NotImplementedError(
            f"`ml` is wired into the trajectory task only, not {task.value!r}; legacy "
            "permitted singlepoint prediction — not yet ported."
        )

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

    from koopmans.aiida.conversion import kpoints_input_to_kpoints_mesh

    structure, _pseudo_family, overrides = _prepare_common_inputs(koopmans_input, ["scf", "bands"])

    return RunPwBands.build(
        code=codes["pw"],
        structure=structure,
        overrides=overrides,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        scf_kpoints=kpoints_input_to_kpoints_mesh(koopmans_input.kpoints),
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

    from koopmans.aiida.conversion import kpoints_input_to_kpoints_mesh

    structure, pseudo_family, overrides = _prepare_common_inputs(koopmans_input, ["scf"])
    overrides["scf"]["pw"]["parameters"].get("SYSTEM", {}).pop("nbnd", None)

    return DielectricTask.build(
        pw_code=codes["pw"],
        ph_code=codes["ph"],
        structure=structure,
        pseudo_family=pseudo_family,
        overrides=overrides,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        scf_kpoints=kpoints_input_to_kpoints_mesh(koopmans_input.kpoints),
    )


def _keywords_setting_projections(
    koopmans_input: KoopmansInput, *, channels_only: bool = False
) -> list[str]:
    """List the keywords the input uses to set explicit Wannier projections.

    Covers the two spin-channel projection blocks and, unless
    ``channels_only``, the top-level one. Empty when the input sets none of
    them, so callers can both test for explicit projections and name them.
    """
    w90 = koopmans_input.calculator_parameters.wannier90
    keywords = (
        ["`calculator_parameters.w90.projections`"] if w90.projections and not channels_only else []
    )
    keywords += [
        f"`calculator_parameters.w90.{name}.projections`"
        for name in ("up", "down")
        if getattr(w90, name) is not None and getattr(w90, name).projections
    ]
    return keywords


def _and_list(items: list[str]) -> str:
    """Join items into an ``a``, ``b`` and ``c`` phrase."""
    if len(items) < 2:
        return "".join(items)
    return " and ".join([", ".join(items[:-1]), items[-1]])


def _validate_projection_sources(koopmans_input: KoopmansInput) -> None:
    """Validate how the input asks for the Wannier projections to be defined.

    Three inputs can speak for the projections: explicit blocks in
    ``calculator_parameters.w90(.up/.down).projections``, the automatic
    derivation requested by ``workflow.auto_projections``, and the external
    projector files of ``pw2wannier90.atom_proj_ext``. Explicit blocks
    define the full set themselves, so they combine with neither of the
    others; the external files only choose where the projector functions
    come from, not whether the projections are derived at all, so they
    require the flag.
    """
    external = koopmans_input.calculator_parameters.pw2wannier90.atom_proj_ext
    automatic = koopmans_input.workflow.auto_projections
    explicit = _keywords_setting_projections(koopmans_input)
    if explicit and automatic:
        raise ValueError(
            f"`workflow.auto_projections` and explicit projections (in {_and_list(explicit)}) were "
            "both given; the automatic derivation and the explicit blocks each define "
            "the full set of projections. Drop one of the two."
        )
    if explicit and external:
        raise ValueError(
            f"Explicit projections ({explicit}) and `pw2wannier90.atom_proj_ext` were "
            "both given; the explicit projections define every block, so the external "
            "projectors would be silently ignored. Drop one of the two."
        )
    if external and not automatic:
        raise ValueError(
            "`pw2wannier90.atom_proj_ext` was given without `workflow.auto_projections`; "
            "external projector files choose where the projector functions come from, "
            "but they do not by themselves ask for the projections to be derived "
            "automatically. Set `workflow.auto_projections` as well."
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
        A WorkGraph wrapping a single ``Wannier90WorkChain`` over the whole
        manifold, or :func:`_build_wannierize_blocks_workgraph` when explicit
        projections or ``block_wannierization_threshold`` ask for one
        Wannierization per block.
    """
    from aiida_koopmans.workgraphs.wannier90 import Wannierize
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    if koopmans_input.workflow.spin != SpinType.NONE:
        raise NotImplementedError(
            "Wannierization currently supports spin='none' only: no route sets "
            "`nspin`, and the per-block group detection and split are "
            "single-channel."
        )

    if koopmans_input.workflow.block_wannierization_threshold is not None:
        return _build_wannierize_blocks_workgraph(koopmans_input, codes)

    _validate_projection_sources(koopmans_input)
    if _keywords_setting_projections(koopmans_input):
        return _build_wannierize_blocks_workgraph(koopmans_input, codes)
    if not koopmans_input.workflow.auto_projections:
        raise ValueError(_NO_PROJECTIONS_PROVIDED_MESSAGE)

    structure, pseudo_family, overrides = _prepare_common_inputs(koopmans_input, ["scf", "nscf"])

    # The automatically derived projections are the pseudopotentials' atomic
    # orbitals (upstream's ATOMIC_PROJECTORS_QE mechanism) unless external
    # projector files supply the projector functions instead.
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


def _build_wannierize_blocks_workgraph(
    koopmans_input: KoopmansInput,
    codes: Codes,
) -> WorkGraph:
    """Build the Wannierization workgraph that Wannierizes block by block.

    One scf + nscf feeds a separate Wannierization per projection block.
    With explicit projections in ``calculator_parameters.w90.projections``
    each user block becomes a wannierization block. With
    ``workflow.auto_projections`` instead, a single atomic-projector block
    spans the whole manifold; its projectors come from the external
    projector directory ``pw2wannier90.atom_proj_dir`` when
    ``pw2wannier90.atom_proj_ext`` is set and from the pseudopotentials
    otherwise (:func:`_create_automatic_blocks`).

    Setting ``block_wannierization_threshold`` adds the automated splitting:
    a pw.x bands run along the k-path feeds a runtime band-group detection
    (splitting at every gap wider than the threshold in eV and at the
    occupied/empty boundary), and each block whose bands fall into several
    groups is Wannierized once, split with Wannier.jl parallel transport,
    re-Wannierized group by group and its products merged back together. An
    automatic-projector block always splits this way, since its band groups
    exist only at runtime.

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
    _validate_projection_sources(koopmans_input)
    channels = _keywords_setting_projections(koopmans_input, channels_only=True)
    if channels:
        raise NotImplementedError(
            f"Explicit projections in {_and_list(channels)} are not wired into the block-by-block "
            "wannierize route, which is single-channel: give them in "
            "`calculator_parameters.w90.projections` instead."
        )
    projections = calc_params.wannier90.projections
    if threshold is not None and koopmans_input.kpoints.path is None:
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
            "occupied/empty boundary the blocks are built around does not support yet."
        )
    num_occ_bands = nelec // 2

    nbnd = calc_params.nbnd if calc_params.nbnd is not None else calc_params.pw.system.nbnd
    nbnd = int(nbnd) if nbnd is not None else None
    external_kwargs: dict[str, Any] = {}
    if projections:
        if nbnd is None:
            nbnd = _num_wann_total(structure, projections)
        blocks = _create_explicit_blocks(
            structure, projections, nbnd, num_occ_bands, SpinChannel.NONE
        )
    elif workflow.auto_projections:
        external_projectors = None
        if calc_params.pw2wannier90.atom_proj_ext:
            external_projectors, projector_path = _load_external_projectors(
                structure, calc_params.pw2wannier90.atom_proj_dir
            )
            external_kwargs = {
                "external_projectors_path": projector_path,
                "external_projectors": external_projectors,
            }
        blocks, nbnd = _create_automatic_blocks(
            structure, pseudos, external_projectors, nbnd, num_occ_bands
        )
    else:
        raise ValueError(_NO_PROJECTIONS_PROVIDED_MESSAGE)

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
    # here and carry the grid separately. The scf takes the mesh itself and
    # may reduce it by symmetry.
    kmesh = kpoints_input_to_kpoints_mesh(koopmans_input.kpoints)
    mp_grid = [int(x) for x in kmesh.get_kpoints_mesh()[0]]  # type: ignore[no-untyped-call]

    # Without a threshold the graph splits nothing, and WannierizeBlocks
    # rejects the split-only inputs rather than ignore them.
    split_kwargs: dict[str, Any] = {}
    if threshold is not None:
        split_kwargs = {
            "bands_kpoints": kpoints_input_to_kpoints_path(koopmans_input.kpoints, structure),
            "num_occ_bands": num_occ_bands,
            "split_threshold": float(threshold),
        }

    return WannierizeBlocks.build(
        codes=codes,
        structure=structure,
        blocks=blocks,
        kpoints=get_explicit_kpoints(kmesh),
        mp_grid=mp_grid,
        scf_kpoints=kmesh,
        **split_kwargs,
        pseudo_family=pseudo_family,
        overrides=wannier_overrides,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        **external_kwargs,
    )


def _num_wann_total(structure: orm.StructureData, projection_blocks: list[list[Projection]]) -> int:
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
        up_blocks = _create_explicit_blocks(
            structure, w90.up.projections, nscf_nbnd, nocc_up, SpinChannel.UP
        )
        validate_projection_block_sequence(up_blocks)
        _validate_blocks_separate_occ_and_emp(up_blocks, nocc_up)
        _validate_blocks_cover_all_occ_bands(up_blocks, nocc_up)
        down_blocks = _create_explicit_blocks(
            structure, w90.down.projections, nscf_nbnd, nocc_down, SpinChannel.DOWN
        )
        validate_projection_block_sequence(down_blocks)
        _validate_blocks_separate_occ_and_emp(down_blocks, nocc_down)
        _validate_blocks_cover_all_occ_bands(down_blocks, nocc_down)
        blocks = up_blocks + down_blocks
    else:
        if nelec % 2:
            raise ValueError(
                f"Odd electron count ({nelec}) requires spin='collinear' for the "
                "Wannier-initialised DSCF route."
            )
        nocc = nelec // 2
        blocks = _create_explicit_blocks(
            structure, calc_params.wannier90.projections, nscf_nbnd, nocc, SpinChannel.NONE
        )
        validate_projection_block_sequence(blocks)
        _validate_blocks_separate_occ_and_emp(blocks, nocc)
        _validate_blocks_cover_all_occ_bands(blocks, nocc)

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
    """Build a workgraph for a trajectory (machine-learning) task.

    Fans the snapshots out over per-snapshot ``KoopmansDSCFWorkflow`` runs via
    ``aiida_koopmans.workgraphs.ml.TrajectoryWorkflow`` and, depending on the
    ``ml`` configuration, trains a screening-parameter model on the computed
    alphas (``ml: {mode: train}``), scores an existing model against them
    (``mode: test``), or applies an existing model in place of the Delta-SCF
    refinement (``mode: predict`` — each snapshot runs one trial KI at the
    guess alphas, the model predicts every screening parameter from the
    trial's self-Hartrees, and the final KI applies the predictions).

    ``self_hartree`` needs nothing beyond the kcp.x runs themselves.
    ``power_spectrum`` builds its power spectra from a pw2wannier90.x
    ``wan_mode='decompose'`` pass over each snapshot's per-block Wannier
    functions, so it requires the Wannier-initialised route
    (``init_orbitals`` in ``mlwfs`` / ``projwfs``); the ``ml``
    radial-basis settings become that pass's namelist keys. ``mode: predict``
    supports ``self_hartree`` only: the decompose pass that builds the
    power-spectrum descriptors is not wired into the DSCF's screening
    stage, where the prediction runs.

    Each frame of the ``atoms.snapshots`` xyz becomes one ``snapshot_N``
    structure fed to the dynamic snapshots namespace. All frames share one
    cell, composition and projection set, so the Wannier-route inputs are
    derived once from the first frame.
    """
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
    ml_mode, ml_model = _resolve_trajectory_ml(ml_config, workflow)

    snapshots = atoms_input_to_structures(koopmans_input.atoms)
    ensure_pseudo_family_installed(workflow.pseudo_library)

    inputs = _kcp_dscf_inputs(koopmans_input)

    extra_kwargs: dict[str, Any] = {}
    if workflow.init_orbitals in (
        VariationalOrbitalType.MLWFS,
        VariationalOrbitalType.PROJWFS,
    ):
        extra_kwargs = _dscf_wannier_init_inputs(
            koopmans_input, next(iter(snapshots.values())), codes, inputs["nbnd"]
        )

    if ml_mode != MLMode.NONE and ml_config.descriptor == MLDescriptor.POWER_SPECTRUM:
        extra_kwargs["pw2wannier90_code"] = _load_code("pw2wannier90", "pw2wannier90.x")
        extra_kwargs["decompose_parameters"] = _decompose_parameters(ml_config)

    return TrajectoryWorkflow.build(
        code=codes["kcp"],
        snapshots=snapshots,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        **inputs,
        **extra_kwargs,
        ml_mode=ml_mode,
        ml_model=ml_model,
        estimator=ml_config.estimator,
        descriptor=ml_config.descriptor,
        occ_and_emp_together=ml_config.occ_and_emp_together,
    )


def _resolve_trajectory_ml(
    ml_config: MLConfig, workflow: WorkflowConfig
) -> tuple[MLMode, dict[str, Any] | None]:
    """Map the ``ml`` block onto a trajectory mode and its loaded model.

    ``test`` and ``predict`` modes load the JSON model from
    ``ml:model_file``. Predict-mode inputs that cannot take effect raise
    here: the ``power_spectrum`` descriptor (its decompose pass is not
    wired into the DSCF's screening stage, where the prediction runs)
    and ``alpha_numsteps != 1``.
    """
    from json import load as json_load

    ml_mode = ml_config.mode

    if ml_mode == MLMode.PREDICT:
        if ml_config.descriptor == MLDescriptor.POWER_SPECTRUM:
            raise NotImplementedError(
                "ml:mode='predict' supports only descriptor='self_hartree': the "
                "decompose pass that builds the power-spectrum descriptors is not wired "
                "into the DSCF's screening stage, where the prediction runs. Use "
                "descriptor='self_hartree'."
            )
        if workflow.alpha_numsteps != 1:
            raise ValueError(
                "ml:mode='predict' replaces the Delta-SCF refinement with a single "
                "trial-KI prediction, so workflow:alpha_numsteps cannot take effect; "
                "set it to 1."
            )

    ml_model = None
    if ml_mode in (MLMode.TEST, MLMode.PREDICT):
        if ml_config.model_file is None:
            raise ValueError(
                f"ml:mode='{ml_mode.value}' requires ml:model_file (the JSON model "
                "produced by a mode='train' run)."
            )
        with open(ml_config.model_file) as handle:
            ml_model = json_load(handle)
    return ml_mode, ml_model


def _decompose_parameters(ml_config: MLConfig) -> dict[str, float | int]:
    """Map the ``ml`` radial-basis settings onto the decompose namelist keys.

    The power spectrum is defined by the Gaussian x spherical-harmonic
    basis the density is projected onto, so ``n_max`` / ``l_max`` /
    ``r_min`` / ``r_max`` have to reach pw2wannier90.x rather than being
    left at the CalcJob's defaults.
    """
    return {
        "decompose_n_max": ml_config.n_max,
        "decompose_l_max": ml_config.l_max,
        "decompose_r_min": ml_config.r_min,
        "decompose_r_max": ml_config.r_max,
    }


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
