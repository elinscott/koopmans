"""Workflow building logic for koopmans AiiDA integration.

This module handles selecting and constructing the appropriate AiiDA workgraph
based on the task specified in a KoopmansInput.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, cast

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
from koopmans.input_file.parallelization import POOL_SUPPORTING_CODES
from koopmans.input_file.workflow import Task

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiida_koopmans.ml import ModelMismatchError
    from aiida_koopmans.parallelization import ParallelizationDict, ParallelizationError
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


def code_executables() -> dict[str, str]:
    """Return the executable behind every code name the dispatcher can load.

    The names are the labels ``koopmans install`` registers, so a name outside
    this mapping names a code no profile holds. ``wannierjl`` is the one entry
    naming no binary on PATH: its julia executable is registered by
    ``aiida_wannierjl.helpers.get_wannierjl_code``.
    """
    from koopmans.aiida.setup.codes import code_specs

    return {label: executable for label, (executable, _) in code_specs().items()} | {
        "wannierjl": "julia (Wannier.jl)"
    }


def executable_for(name: str) -> str:
    """Return the executable ``name`` labels, or raise if no such code is registered."""
    executables = code_executables()
    if name not in executables:
        raise ValueError(
            f"'{name}' is not a code koopmans registers, so no profile can hold it. "
            f"Name one of {sorted(executables)}."
        )
    return executables[name]


def code_install_hint(name: str) -> str:
    """Return the sentence telling the reader how to register the code ``name``.

    ``koopmans install`` scans the machine for the Quantum ESPRESSO binaries
    and registers one code per binary it finds. ``wannierjl`` is the one code
    it cannot find that way: it runs Julia against a pinned Wannier.jl
    project that has to be built first.
    """
    if name == "wannierjl":
        return (
            "Build the Julia environment with "
            "`aiida_wannierjl.helpers.setup_julia_environment`, then register the code "
            "with `aiida_wannierjl.helpers.get_wannierjl_code`."
        )
    return "Run `koopmans install` to register every koopmans code this machine has."


def load_code(name: str, task: Task) -> orm.AbstractCode:
    """Load the code labelled ``<name>@localhost``, or say how to install it.

    Args:
        name: The code label, as ``koopmans install`` registers it.
        task: The task whose chain runs the code, named in the message.

    Returns:
        The stored code.

    Raises:
        ValueError: If this profile holds no such code.
    """
    executable = executable_for(name)
    try:
        return orm.load_code(f"{name}@localhost")
    except Exception as exc:
        raise ValueError(
            f"The `{task.value}` task runs {executable}, but this AiiDA profile has no "
            f"`{name}@localhost` code. {code_install_hint(name)}"
        ) from exc


def load_codes_for_task(koopmans_input: KoopmansInput) -> tuple[Codes, ParallelizationDict]:
    """Load every code the task in ``koopmans_input`` runs, and settle their ranks.

    The route decides which codes those are (:func:`route_for`), and it
    decides once, up front: a code the run might reach has to be in the
    profile before the run starts, so its absence is reported as a setup
    error rather than discovered part-way through.

    Loading a code and settling how many MPI ranks it runs on are one act,
    so the mapping this returns covers every code it loaded.

    Returns:
        Tuple of (codes, per-code mapping with every rank count settled).

    Raises:
        ValueError: If a code is not in the profile, or if a code's ``npool``
            does not divide the ranks it will run on.
    """
    task = koopmans_input.workflow.task
    loaded: dict[str, orm.AbstractCode] = {}
    for name in route_for(task).required_codes(koopmans_input):
        # Before the profile is asked: an unregistered name is a bug in the
        # route's declaration, not a code the user has yet to install.
        executable_for(name)
        if name not in loaded:
            loaded[name] = load_code(name, task)

    requested = koopmans_input.parallelization.as_mapping()
    completed = complete_rank_counts(requested, cast("Codes", loaded))
    check_pools_divide_ranks(completed, requested)
    return cast("Codes", loaded), completed


def default_rank_count(name: str, code: orm.AbstractCode) -> int | None:
    """Return the MPI ranks ``code`` takes when the input file names no ``ntasks``.

    One rank for a code that does not run under MPI, and the code's computer's
    ``default_mpiprocs_per_machine`` otherwise. ``None`` means nothing declares
    a count, which leaves it to the scheduler.

    A code named in :data:`~koopmans.aiida.setup.codes.SERIAL_CODES` takes one
    rank whatever its node says, because that list states a property of the
    program rather than of the node.
    """
    from koopmans.aiida.setup.codes import SERIAL_CODES, effective_with_mpi

    if name in SERIAL_CODES or effective_with_mpi(code) is False:
        return 1
    computer = getattr(code, "computer", None)
    if computer is None:
        return None
    ranks = computer.get_default_mpiprocs_per_machine()
    return int(ranks) if ranks is not None else None


def complete_rank_counts(parallelization: ParallelizationDict, codes: Codes) -> ParallelizationDict:
    """Return ``parallelization`` with an explicit ``ntasks`` for every loaded code.

    An entry with no ``ntasks`` reaches the scheduler as a rank count nothing
    stores, resolved at submission against the computer's mutable default; two
    calculations with identical stored inputs can then run on different numbers
    of ranks. Filling the count in makes it an input of the workgraph.

    Only codes in ``codes`` are completed, and only those the ``parallelization``
    block names (:data:`~aiida_koopmans.parallelization.CODE_NAMES`) — an entry
    for any other key is rejected downstream. A code outside that vocabulary
    (merge_evc, wannierjl) gets no entry at all, because the input file has no
    way to name one: its rank count is its CalcJob's to declare. An ``ntasks``
    the input file set is left as it is, so this never overrides the user.

    Args:
        parallelization: The per-code mapping, as ``ParallelizationInput.as_mapping``
            returns it.
        codes: The codes loaded for this task, keyed by the same code names.

    Returns:
        A new mapping; the argument is not modified.
    """
    from aiida_koopmans.parallelization import CODE_NAMES

    completed: dict[str, dict[str, Any]] = {
        name: dict(entry) for name, entry in parallelization.items()
    }
    for name, code in codes.items():
        if name not in CODE_NAMES:
            continue
        entry = completed.setdefault(name, {})
        if entry.get("ntasks") is not None:
            continue
        ranks = default_rank_count(name, code)
        if ranks is not None:
            entry["ntasks"] = ranks
    return cast("ParallelizationDict", {name: entry for name, entry in completed.items() if entry})


def _divisors(number: int) -> list[int]:
    """Return every positive divisor of ``number``, ascending."""
    return [candidate for candidate in range(1, number + 1) if number % candidate == 0]


def check_pools_divide_ranks(
    parallelization: ParallelizationDict, requested: ParallelizationDict
) -> None:
    """Raise if a code's ``npool`` does not divide the MPI ranks it will run on.

    Quantum ESPRESSO splits a run's ranks into ``npool`` equal k-point pools,
    so the rank count must be a multiple of ``npool``.

    Checks only codes that accept ``-npool`` (:data:`POOL_SUPPORTING_CODES`)
    and that carry a rank count, which after :func:`complete_rank_counts` means
    the codes this task loads — an entry for a code it never runs is left
    alone.

    Args:
        parallelization: The completed per-code mapping, every loaded code
            carrying the ``ntasks`` it will run with.
        requested: What the input file named, which decides whether the message
            points the reader at their own ``ntasks`` or at the computer's
            default.

    Raises:
        ValueError: If a code's ``npool`` does not divide its rank count.
    """
    for name, config in parallelization.items():
        npool = config.get("npool")
        ranks = config.get("ntasks")
        if npool is None or ranks is None or name not in POOL_SUPPORTING_CODES:
            continue
        if ranks % npool == 0:
            continue

        if requested.get(name, {}).get("ntasks") is not None:
            ranks_come_from = f"`parallelization.{name}.ntasks` asks for {ranks} MPI ranks"
        else:
            ranks_come_from = (
                f"`parallelization.{name}` sets no `ntasks`, so {name} takes this "
                f"computer's default of {ranks} MPI ranks"
            )

        raise ValueError(
            f"{ranks_come_from}, and `parallelization.{name}.npool` is {npool}, which "
            f"does not divide {ranks}. Quantum ESPRESSO splits a run's ranks into "
            f"equal k-point pools, so {name} would abort at startup. Set `npool` to "
            f"one of {_divisors(ranks)}, or set `parallelization.{name}.ntasks` to a "
            f"multiple of {npool}."
        )


def prepare_common_inputs(
    koopmans_input: KoopmansInput,
    override_keys: list[str],
    parallelization: ParallelizationDict,
) -> tuple[orm.StructureData, str, dict[str, Any]]:
    """Prepare the common inputs shared by all workgraph builders.

    Converts the koopmans input into a structure, ensures the pseudo family is
    installed, and builds an overrides dict with a PW parameters entry for each
    of the requested sub-workflow keys.

    Args:
        koopmans_input: The parsed koopmans input.
        override_keys: Sub-workflow keys to include in overrides (e.g. ["scf", "bands"]).
        parallelization: The per-code mapping, rank counts already completed by
            :func:`complete_rank_counts`.

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
    options, settings = code_parallelization(parallelization.get("pw"))
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


class Route(NamedTuple):
    """A task's route: the codes its chain runs on, and its graph builder."""

    required_codes: Callable[[KoopmansInput], list[str]]
    build: Callable[[KoopmansInput, Codes, ParallelizationDict], WorkGraph]


def route_for(task: Task) -> Route:
    """Return the route that builds ``task``, or raise if nothing builds it.

    Each route module states its own code set, because every condition that
    picks one reads an input-file field.

    The route modules are imported here rather than at module level:
    importing one pulls in its plugin dependencies, and a caller that only
    wants the input file parsed should not pay for the graphs.

    Raises:
        ValueError: If no route builds ``task``.
    """
    from koopmans.aiida.workflows import dft, dscf, eps, trajectory, wannierize

    routes = {
        Task.DFT_BANDS: Route(dft.required_codes, dft.build_dft_bands_workgraph),
        Task.WANNIERIZE: Route(wannierize.required_codes, wannierize.build_wannierize_workgraph),
        Task.SINGLEPOINT: Route(dscf.required_codes, dscf.build_singlepoint_workgraph),
        Task.TRAJECTORY: Route(trajectory.required_codes, trajectory.build_trajectory_workgraph),
        Task.DFT_EPS: Route(eps.required_codes, eps.build_dft_eps_workgraph),
    }
    if task not in routes:
        raise ValueError(
            f"Task '{task.value}' is not yet implemented. "
            f"Supported tasks: {', '.join(sorted(supported.value for supported in routes))}"
        )
    return routes[task]


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

    # Load the codes the route runs, each carrying the MPI ranks it will run on
    route = route_for(task)
    codes, parallelization = load_codes_for_task(koopmans_input)

    # An error raised inside the plugin speaks its vocabulary (derived blocks,
    # `num_bands`), which the user never wrote; attach the input-file advice
    # at this boundary.
    try:
        return route.build(koopmans_input, codes, parallelization)
    except Exception as exc:
        advice = advice_for(exc)
        if advice is not None:
            # A PEP 678 note survives exception types whose constructors do
            # not take a single message, and keeps type, args and chaining
            # intact; it renders under the message in the traceback.
            exc.add_note(advice)
        raise
