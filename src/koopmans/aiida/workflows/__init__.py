"""Workflow building logic for koopmans AiiDA integration.

This module handles selecting and constructing the appropriate AiiDA workgraph
based on the task specified in a KoopmansInput.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    NotRequired,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from aiida import orm
from aiida_koopmans.ml import MLMode

from koopmans.aiida.conversion import (
    atoms_input_to_structure,
    code_parallelization,
    input_to_pw_parameters,
    step_grid_spacing,
    step_kpoints_mesh,
)
from koopmans.input_file.workflow import Task

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
    from aiida_workgraph.errors import MissingRequiredInputsError

    from koopmans.input_file import KoopmansInput


def load_code(name: str, executable: str) -> orm.AbstractCode:
    """Load the code labelled ``<name>@localhost``, with a setup hint on failure."""
    try:
        return orm.load_code(f"{name}@localhost")
    except Exception as exc:
        raise ValueError(
            f"Could not load {executable} code: {exc}\n"
            "Please run 'koopmans install' first to set up the AiiDA backend."
        ) from exc


def load_codes[CodesT: Mapping[str, Any]](codes_spec: type[CodesT]) -> CodesT:
    """Load every configured ``<member>@localhost`` code a codes TypedDict declares.

    ``codes_spec`` is the workflow's graph-input TypedDict, declared beside
    the workflow entry point it feeds (e.g.
    ``aiida_koopmans.workgraphs.kcp.DscfCodes``) — the single declaration of
    which codes the workflow wires. Every member — required and
    ``NotRequired`` alike — is loaded when a ``<member>@localhost`` code is
    configured, and left out when it is not: which codes a run actually
    needs is now a structural property of the graph it builds (its
    TypedDict specs carry the requiredness), not a decision made here from
    the input file. A route missing a code the run actually needs finds out
    at graph validation, translated to install advice at the CLI boundary
    (:func:`advice_for`).
    """
    from aiida.common.exceptions import NotExistent

    codes: dict[str, orm.AbstractCode] = {}
    for name in get_type_hints(codes_spec, include_extras=True):
        try:
            codes[name] = orm.load_code(f"{name}@localhost")
        except NotExistent:
            pass
    return cast("CodesT", codes)


def _socket_help(hint: Any) -> str | None:
    """Return the ``SocketMeta`` help attached to a codes-TypedDict member annotation."""
    if get_origin(hint) is NotRequired:
        hint = get_args(hint)[0]
    for meta in getattr(hint, "__metadata__", ()):
        text = getattr(meta, "help", None)
        if text:
            return str(text)
    return None


def _render_missing_codes_advice(help_by_name: Mapping[str, str | None]) -> str:
    """Render a ``name -> declared help`` mapping as the shared install-advice message.

    The one rendering both :func:`_missing_inputs_advice` (the submit-time
    translation of a structural ``MissingRequiredInputsError``) and
    :func:`require_configured_codes` (the build-time pre-flight) use, so the
    two call sites can never drift into two different wordings for the same
    fact.
    """
    lines = [
        f"  - `{name}@localhost`" + (f" ({help_text})" if help_text else "")
        for name, help_text in sorted(help_by_name.items())
    ]
    return (
        "This calculation needs codes that are not configured:\n"
        + "\n".join(lines)
        + "\nPlease run 'koopmans install' to set up the AiiDA backend."
    )


def require_configured_codes[CodesT: Mapping[str, Any]](
    codes_spec: type[CodesT], codes: CodesT
) -> None:
    """Raise install advice for any of ``codes_spec``'s required members absent from ``codes``.

    A build-time fast path for what the graph's own structural check
    already enforces at submit (:func:`advice_for`'s translation of
    ``MissingRequiredInputsError``) — redundant by design, and only worth
    keeping because some entry graphs still bind a required code by direct
    dict subscript rather than through ``node_graph.reference``, which dies as a
    bare ``KeyError`` mid-``build()`` with no chance for the structural
    check to run at all (see ``tests/test_code_loading.py``'s
    ``TestPreFlightAdvice`` for which routes currently need this). Safe to
    delete once every such body defers instead.

    Checks ``__required_keys__`` alone, never ``NotRequired`` members: which
    of those a particular input additionally turns on is conditional
    knowledge this function does not have and must not guess at (e.g.
    DFPT's ``ph``, needed only under ``eps_inf: auto``) — that stays the
    structural check's job, at submit.
    """
    hints = get_type_hints(codes_spec, include_extras=True)
    # Every codes_spec argument is a TypedDict class, which carries
    # __required_keys__ at runtime; the Mapping bound cannot say so.
    required_keys: frozenset[str] = codes_spec.__required_keys__  # type: ignore[attr-defined]
    missing = {name: _socket_help(hints[name]) for name in required_keys if name not in codes}
    if missing:
        raise ValueError(_render_missing_codes_advice(missing))


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
            "`calculator_parameters.ecutwfc`. `ecutrho` follows at four times it."
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


def reject_band_path(
    koopmans_input: KoopmansInput,
    message: str,
    error: type[Exception] = ValueError,
) -> None:
    """Raise for a ``kpoints.path`` the route about to be built cannot interpolate along.

    Every route that can produce a band structure does so whenever the input
    names a path, so a route that cannot must say so rather than build a
    graph the path never reaches. The predicate matches
    :func:`koopmans.aiida.conversion.kpoints_input_to_interpolation_path`: a
    gamma-only input's fixed ``path`` names the zone centre alone and so
    defines no segment to interpolate along.

    Args:
        koopmans_input: The parsed koopmans input.
        message: What the user should write instead.
        error: ``ValueError`` where the route's physics produces no band
            structure at all, ``NotImplementedError`` where koopmans has
            simply not ported the stage that would.

    Raises:
        Exception: An instance of ``error``, if the input names a path with
            segments.
    """
    kpoints = koopmans_input.kpoints
    if not kpoints.gamma_only and kpoints.path is not None:
        raise error(message)


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


def _missing_inputs_advice(exc: MissingRequiredInputsError) -> str | None:
    """Phrase graph-level missing-code sockets as install advice.

    A workflow body that wires a code member it was not given surfaces as
    unfilled ``workgraph.code`` sockets at graph validation — one per socket
    the missing code was threaded to, so the same code can be named several
    times over (a route's own top-level ``codes.pw`` alongside every
    downstream task it feeds). Grouped here by member name, read off each
    socket path's last segment (dropping a ``_code`` suffix a task's own
    kwarg may add): every entry for the same code names the same
    ``<name>@localhost``, so one line covers all of them. A segment of
    plain ``code`` names no member and is skipped, since ``code@localhost``
    would tell the reader nothing to install. The purpose shown is any
    ``help`` an entry carries, preferring one under the route's own
    top-level ``graph_inputs.codes.*`` namespace, since that names the code
    in the vocabulary of the route the user asked for. Entries of other
    socket types are not code-installation problems, so an error naming
    only those earns no advice.
    """
    help_by_name: dict[str, str | None] = {}
    for entry in exc.missing:
        if entry.identifier != "workgraph.code":
            continue
        name = entry.socket_path.rsplit(".", 1)[-1].removesuffix("_code")
        if name == "code":
            continue
        route_level = entry.socket_path.startswith("graph_inputs.codes.")
        if name not in help_by_name:
            help_by_name[name] = entry.help
        elif entry.help and (route_level or help_by_name[name] is None):
            help_by_name[name] = entry.help
    if not help_by_name:
        return None
    return _render_missing_codes_advice(help_by_name)


def _model_mismatch_advice(exc: ModelMismatchError) -> str:
    """Phrase the model-mismatch advice in input-file vocabulary."""
    stamp = f" (its {exc.field!r} stamp)" if exc.field else ""
    return (
        f"The model loaded from `ml.model_file` was trained under different "
        f"settings{stamp}: retrain with `ml: {{mode: train}}` under this run's "
        "settings, or point `ml.model_file` at a matching model."
    )


def _plugin_advice() -> tuple[tuple[type[ValueError], Callable[[Any], str | None]], ...]:
    """Return the advice table for the plugin's typed errors.

    One advice per class — the plugin defines each class for exactly one
    piece of advice — and the class's structured attribute (block label,
    code name, model stamp) sharpens the sentence when the raise site
    filled it in. An advisor may decline with ``None`` when the instance
    carries nothing its advice speaks to (missing graph inputs that are
    not codes).

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
    from aiida_workgraph.errors import MissingRequiredInputsError

    return (
        (MissingRequiredInputsError, _missing_inputs_advice),
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
    itself a typed plugin error. A matching advisor may still decline
    with ``None`` for an instance its advice does not speak to.
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

    # Build the workgraph based on task. Each route loads its workflow's
    # codes itself (:func:`load_codes`) once its input validation has passed.
    # An error raised inside the plugin speaks its vocabulary (derived
    # blocks, `num_bands`), which the user never wrote; attach the
    # input-file advice at this boundary.
    try:
        if task == Task.DFT_BANDS:
            from koopmans.aiida.workflows.dft import build_dft_bands_workgraph

            return build_dft_bands_workgraph(koopmans_input)
        elif task == Task.WANNIERIZE:
            from koopmans.aiida.workflows.wannierize import build_wannierize_workgraph

            return build_wannierize_workgraph(koopmans_input)
        elif task == Task.SINGLEPOINT:
            from koopmans.aiida.workflows.dscf import build_singlepoint_workgraph

            return build_singlepoint_workgraph(koopmans_input)
        elif task == Task.TRAJECTORY:
            from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

            return build_trajectory_workgraph(koopmans_input)
        elif task == Task.DFT_EPS:
            from koopmans.aiida.workflows.eps import build_dft_eps_workgraph

            return build_dft_eps_workgraph(koopmans_input)
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
