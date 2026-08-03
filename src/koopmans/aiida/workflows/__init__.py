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
)
from koopmans.input_file.workflow import (
    CalculateScreeningMethod,
    Correction,
    Task,
)

if TYPE_CHECKING:
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


def prepare_common_inputs(
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


#: Input-file advice keyed by the plugin module that raised. The plugin
#: raises builtin ValueError/NotImplementedError everywhere, so the raise
#: site is the only structural key available until it grows typed
#: exceptions to dispatch on. The key is module-wide: any error escaping a
#: keyed module carries that module's advice, and an error raised in a
#: helper module that a keyed module calls carries none.
_PLUGIN_ADVICE = {
    "aiida_koopmans.projections": (
        "The blocks named above are derived from the input file's projections "
        "(`calculator_parameters.w90.projections`, or its `up`/`down` variants "
        "for collinear spin): they tile the bands in listed order, each taking "
        "its own Wannier-function count, and the last block reads any bands "
        "left up to `nbnd` for disentanglement. Adjust the projections or "
        "`nbnd` there."
    ),
}


def advice_for(exc: BaseException) -> str | None:
    """Return input-file advice for an exception the plugin raised, if any.

    Keyed on the module of the raise site (the innermost traceback frame).
    An exception the dispatcher replaced via ``raise ... from exc`` carries
    a koopmans raise site and gets no translation; a bare ``raise`` in a
    koopmans ``except`` block keeps the plugin raise site and still does.
    """
    module = None
    tb = exc.__traceback__
    while tb is not None:
        module = tb.tb_frame.f_globals.get("__name__")
        tb = tb.tb_next
    return _PLUGIN_ADVICE.get(module) if isinstance(module, str) else None


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
