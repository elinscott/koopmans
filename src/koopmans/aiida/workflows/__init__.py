"""Workflow building logic for koopmans AiiDA integration.

This module handles selecting and constructing the appropriate AiiDA workgraph
based on the task specified in a KoopmansInput.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiida import orm
from aiida_koopmans.ml import MLMode
from aiida_koopmans.workgraphs import Codes

from koopmans.aiida.conversion import (
    atoms_input_to_structure,
    code_parallelization,
    input_to_pw_parameters,
    step_grid_spacing,
    step_kpoints_mesh,
)
from koopmans.input_file.workflow import (
    CalculateScreeningMethod,
    Correction,
    Task,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiida_koopmans.ml import ModelMismatchError
    from aiida_koopmans.parallelization import ParallelizationError
    from aiida_koopmans.projections import (
        BlockBoundaryError,
        BlockDisentanglementError,
        EmptyCoverageError,
        OccupiedCoverageError,
        ProjectionSiteError,
    )
    from aiida_koopmans.workgraphs.block_wannierize import FrozenWindowError
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput
    from koopmans.input_file.workflow import WorkflowConfig


def load_code(name: str, executable: str) -> orm.AbstractCode:
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
    codes["pw"] = load_code("pw", "pw.x")

    # A corrected singlepoint — or a trajectory, which runs one DSCF
    # singlepoint per snapshot — needs a screening-method-specific code
    # regardless of ``calculate_alpha``: when alphas are guessed instead
    # of computed, kcp.x/kcw.x still evaluate the corrected functional — only
    # the screening step itself is skipped.
    if task in (Task.SINGLEPOINT, Task.TRAJECTORY) and workflow.correction != Correction.NONE:
        if workflow.screening_method == CalculateScreeningMethod.DSCF:
            codes["kcp"] = load_code("kcp", "kcp.x")
        elif workflow.screening_method == CalculateScreeningMethod.DFPT:
            # kcw.x runs all three DFPT steps (wann2kc, screen, ham) selected
            # via its ``control.calculation`` flag, so a single code suffices.
            codes["kcw"] = load_code("kcw", "kcw.x")

    # The dielectric-constant task runs ph.x on top of the scf
    if task == Task.DFT_EPS:
        codes["ph"] = load_code("ph", "ph.x")

    # Wannierize task needs additional codes
    if task == Task.WANNIERIZE:
        codes["pw2wannier90"] = load_code("pw2wannier90", "pw2wannier90.x")
        codes["wannier90"] = load_code("wannier90", "wannier90.x")

        # Automated block splitting runs the Wannier.jl CalcJobs (the julia
        # binary registered via aiida_wannierjl.helpers.get_wannierjl_code).
        if workflow.block_wannierization_threshold is not None:
            codes["wannierjl"] = load_code("wannierjl", "julia (Wannier.jl)")

        # projwfc is only needed when the Wannierize flow computes a projected
        # DOS / bandstructure, so treat it as optional rather than required.
        try:
            codes["projwfc"] = orm.load_code("projwfc@localhost")
        except Exception:  # noqa: S110
            pass

    return codes


def require_cutoffs_for_family(pseudo_family: str, parameters: dict[str, Any]) -> None:
    """Reject an input that names no cutoffs against a family recommending none.

    Args:
        pseudo_family: Label of the family the pw.x steps will use.
        parameters: The pw.x parameters the input file produced, whose
            ``SYSTEM`` block carries ``ecutwfc`` when the input states it.

    Raises:
        ValueError: If the family publishes no recommended cutoffs and the
            input states none either.
    """
    from koopmans.aiida.setup.pseudos import pseudo_family_has_cutoffs

    if pseudo_family_has_cutoffs(pseudo_family):
        return

    if "ecutwfc" not in parameters.get("SYSTEM", {}):
        raise ValueError(
            f"The pseudopotential family `{pseudo_family}` publishes no recommended "
            "cutoffs, so they must come from the input file: set "
            "`calculator_parameters.ecutwfc`. `ecutrho` follows at four times it "
            "unless `calculator_parameters.pw.system.ecutrho` states otherwise."
        )


def prepare_common_inputs(
    koopmans_input: KoopmansInput,
    override_keys: list[str],
) -> tuple[orm.StructureData, str, dict[str, Any]]:
    """Prepare the common inputs shared by all workgraph builders.

    Converts the koopmans input into a structure, ensures the pseudo family is
    installed, and builds an overrides dict with a PW parameters entry for each
    of the requested sub-workflow keys. An input naming no cutoffs against a
    family recommending none is rejected here
    (:func:`require_cutoffs_for_family`).

    Args:
        koopmans_input: The parsed koopmans input.
        override_keys: Sub-workflow keys to include in overrides (e.g. ["scf", "bands"]).

    Returns:
        Tuple of (structure, pseudo_family, overrides).
    """
    from koopmans.aiida.setup.pseudos import (
        ensure_pseudo_family_installed,
        require_norm_conserving_family,
    )

    structure = atoms_input_to_structure(koopmans_input.atoms)
    parameters = input_to_pw_parameters(koopmans_input)
    pseudo_family = koopmans_input.workflow.pseudo_library

    ensure_pseudo_family_installed(pseudo_family)

    require_norm_conserving_family(pseudo_family, structure)
    require_cutoffs_for_family(pseudo_family, parameters)

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


def reject_kpoint_overrides(koopmans_input: KoopmansInput, messages: dict[str, str]) -> None:
    """Raise for a per-step k-point mesh the route about to be built cannot honour.

    ``messages`` maps a ``kpoints.overrides`` step name to what the user
    should write instead.

    Args:
        koopmans_input: The parsed koopmans input.
        messages: The message to raise for each step this route rejects.

    Raises:
        ValueError: If the input gives a mesh for one of those steps.
    """
    for step, message in messages.items():
        if getattr(koopmans_input.kpoints.overrides, step) is not None:
            raise ValueError(message)


def pin_step_kpoints(
    overrides: dict[str, Any],
    step: str,
    koopmans_input: KoopmansInput,
) -> orm.KpointsData | None:
    """Return the mesh a step samples, recording a grid spacing in ``overrides`` instead.

    ``PwBaseWorkChain`` takes either an explicit mesh or a
    ``kpoints_distance``, so a step whose entry states a ``grid_spacing``
    gets the distance written into its override namespace and no mesh.

    Args:
        overrides: The route's per-step override dict (mutated in place).
        step: Name of a ``kpoints.overrides`` entry.
        koopmans_input: The parsed koopmans input.

    Returns:
        The mesh the step samples, or ``None`` where the spacing fixes it.
    """
    spacing = step_grid_spacing(koopmans_input.kpoints, step)
    if spacing is not None:
        overrides.setdefault(step, {})["kpoints_distance"] = spacing
        return None
    return step_kpoints_mesh(koopmans_input.kpoints, step)


def _projection_site_advice(exc: ProjectionSiteError) -> str:
    """Phrase the unknown-site advice in input-file vocabulary."""
    site = f" ({exc.site!r})" if exc.site else ""
    return (
        "Check that the projections you provided name only elements of the "
        f"structure: the offending `site`{site} must be an element label "
        "appearing in `atomic_positions`."
    )


def _block_boundary_advice(exc: BlockBoundaryError) -> str:
    """Phrase the boundary-straddle advice in input-file vocabulary."""
    return (
        "Check that the projections you provided split at the occupied/empty "
        "boundary: every block must lie wholly in one manifold, so divide the "
        "straddling block's projections into an occupied and an empty block."
    )


def _occupied_coverage_advice(exc: OccupiedCoverageError) -> str:
    """Phrase the occupied-coverage advice in input-file vocabulary."""
    return (
        "Check that the projections you provided cover the occupied manifold: "
        "the occupied blocks must supply exactly one Wannier function per "
        "occupied band."
    )


def _empty_coverage_advice(exc: EmptyCoverageError) -> str:
    """Phrase the empty-headroom advice in input-file vocabulary."""
    return (
        "Check `nbnd` against the projections you provided for the empty "
        "manifold: raise `calculator_parameters.nbnd` until it covers every "
        "empty Wannier function, or trim the empty projections."
    )


def _block_disentanglement_advice(exc: BlockDisentanglementError) -> str:
    """Phrase the lower-block-disentanglement advice in input-file vocabulary."""
    label = f" ({exc.label!r})" if exc.label else ""
    return (
        "Check that only the last of the projection blocks you provided is "
        f"left extra bands to disentangle over: a lower block{label} reads "
        "more bands than it Wannierises, so either lower `nbnd` to remove the "
        "extra bands or move those bands' projections into the final block."
    )


def _frozen_window_advice(exc: FrozenWindowError) -> str:
    """Phrase the frozen-window advice in input-file vocabulary."""
    subject = f"block {exc.label!r}" if exc.label else "the disentangling block"
    return (
        "The frozen window comes from `dis_froz_max` in "
        f"`calculator_parameters.w90`: decrease it until {subject} freezes no "
        "more bands than it Wannierises."
    )


def _parallelization_advice(exc: ParallelizationError) -> str:
    """Phrase the parallelization advice in input-file vocabulary."""
    entry = f"`parallelization.{exc.code}`" if exc.code else "the offending entry"
    return (
        "Per-code ranks and flags come from the input file's top-level "
        f"`parallelization` block; adjust {entry} "
        "(`ntasks` / `npool` / `pd` / `omp`) there."
    )


def _model_mismatch_advice(exc: ModelMismatchError) -> str:
    """Phrase the model-mismatch advice in input-file vocabulary."""
    stamp = f" (its {exc.field!r} stamp)" if exc.field else ""
    return (
        f"The model loaded from `ml.model_file` was trained under different "
        f"settings{stamp}: retrain with `ml: {{mode: train}}` under this run's "
        "settings, or point `ml.model_file` at a matching model."
    )


def _plugin_advice() -> tuple[tuple[type[ValueError], Callable[[Any], str]], ...]:
    """Return the advice table for the plugin's typed errors.

    One advice per class — the plugin defines each class for exactly one
    piece of advice — and the class's structured attribute (block label,
    code name, model stamp) sharpens the sentence when the raise site
    filled it in.

    Advice attaches where ``build_workgraph`` catches: the build boundary.
    ``FrozenWindowError`` and ``ModelMismatchError`` currently raise
    daemon-side only (their validators need runtime data — nscf
    eigenvalues, trial-KI descriptors), so their entries provision for
    validators that reach the build path rather than translate anything
    today.

    The classes are imported here, not at module level: importing
    ``aiida_koopmans.workgraphs.block_wannierize`` loads the AiiDA
    configuration (via aiida-pythonjob, aiidateam/aiida-pythonjob#84),
    which a configuration-less interpreter cannot do. Advice runs only
    after a failure, so the import cost here is irrelevant.
    """
    from aiida_koopmans.ml import ModelMismatchError
    from aiida_koopmans.parallelization import ParallelizationError
    from aiida_koopmans.projections import (
        BlockBoundaryError,
        BlockDisentanglementError,
        EmptyCoverageError,
        OccupiedCoverageError,
        ProjectionSiteError,
    )
    from aiida_koopmans.workgraphs.block_wannierize import FrozenWindowError

    return (
        (ProjectionSiteError, _projection_site_advice),
        (BlockBoundaryError, _block_boundary_advice),
        (OccupiedCoverageError, _occupied_coverage_advice),
        (EmptyCoverageError, _empty_coverage_advice),
        (BlockDisentanglementError, _block_disentanglement_advice),
        (FrozenWindowError, _frozen_window_advice),
        (ParallelizationError, _parallelization_advice),
        (ModelMismatchError, _model_mismatch_advice),
    )


def advice_for(exc: BaseException) -> str | None:
    """Return input-file advice for a typed plugin error, or None.

    Dispatches on the exception's type, so an untyped error — the
    plugin's own plain ``ValueError``s included — passes through
    untranslated, and an error the dispatcher replaced via
    ``raise ... from exc`` is translated only if the replacement is
    itself a typed plugin error.
    """
    for exc_type, advise in _plugin_advice():
        if isinstance(exc, exc_type):
            return advise(exc)
    return None


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

    # Build the workgraph based on task. An error raised inside the plugin
    # speaks its vocabulary (derived blocks, `num_bands`), which the user
    # never wrote; attach the input-file advice at this boundary.
    try:
        if task == Task.DFT_BANDS:
            from koopmans.aiida.workflows.dft import build_dft_bands_workgraph

            return build_dft_bands_workgraph(koopmans_input, codes)
        elif task == Task.WANNIERIZE:
            from koopmans.aiida.workflows.wannierize import build_wannierize_workgraph

            return build_wannierize_workgraph(koopmans_input, codes)
        elif task == Task.SINGLEPOINT:
            from koopmans.aiida.workflows.dscf import build_singlepoint_workgraph

            return build_singlepoint_workgraph(koopmans_input, codes)
        elif task == Task.TRAJECTORY:
            from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

            return build_trajectory_workgraph(koopmans_input, codes)
        elif task == Task.DFT_EPS:
            from koopmans.aiida.workflows.eps import build_dft_eps_workgraph

            return build_dft_eps_workgraph(koopmans_input, codes)
        else:
            raise ValueError(
                f"Task '{task.value}' is not yet implemented. "
                f"Supported tasks: {Task.DFT_BANDS.value}, {Task.WANNIERIZE.value}, "
                f"{Task.SINGLEPOINT.value}, {Task.TRAJECTORY.value}, {Task.DFT_EPS.value}"
            )
    except Exception as exc:
        advice = advice_for(exc)
        if advice is not None:
            # A PEP 678 note survives exception types whose constructors do
            # not take a single message, and keeps type, args and chaining
            # intact; it renders under the message in the traceback.
            exc.add_note(advice)
        raise
