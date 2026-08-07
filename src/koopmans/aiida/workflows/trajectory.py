"""The trajectory (machine-learning) route."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiida_koopmans.ml import MLDescriptor, MLMode
from aiida_quantumespresso.common.types import SpinType

from koopmans.aiida.conversion import atoms_input_to_structures
from koopmans.aiida.workflows import load_codes, reject_kpoint_overrides
from koopmans.aiida.workflows.dscf import (
    KPOINT_OVERRIDES_ON_TRAJECTORY,
    WANNIER_INIT_CODES,
    dscf_wannier_init_inputs,
    kcp_dscf_inputs,
    require_supported_correction,
)
from koopmans.aiida.workflows.projectors import reject_unwired_external_projectors
from koopmans.input_file.workflow import CalculateScreeningMethod, VariationalOrbitalType

if TYPE_CHECKING:
    from aiida import orm
    from aiida_koopmans.parallelization import ParallelizationDict
    from aiida_koopmans.workgraphs import Codes
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput
    from koopmans.input_file.ml import MLConfig
    from koopmans.input_file.workflow import WorkflowConfig


def build_trajectory_workgraph(
    koopmans_input: KoopmansInput,
    codes: Codes,
    parallelization: ParallelizationDict,
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

    reject_unwired_external_projectors(koopmans_input, "trajectory")

    if workflow.calculate_alpha and workflow.screening_method == CalculateScreeningMethod.DFPT:
        raise NotImplementedError(
            "The trajectory task only supports DSCF screening (kcp.x); DFPT screening "
            "is not yet implemented for trajectories."
        )

    # After the screening-method guard: whichever method the input asks for,
    # this route runs kcp.x, and the reader has to hear about the method they
    # asked for before they hear about the mesh.
    reject_kpoint_overrides(koopmans_input, KPOINT_OVERRIDES_ON_TRAJECTORY)

    require_supported_correction(workflow.correction)

    if workflow.spin in (SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT):
        raise NotImplementedError(
            f"spin={workflow.spin.value!r} is not supported by the trajectory (kcp.x) "
            "stream: kcp.x has no noncollinear mode."
        )

    ml_config = koopmans_input.ml
    ml_mode, ml_model = _resolve_trajectory_ml(ml_config, workflow)

    snapshots = atoms_input_to_structures(koopmans_input.atoms)
    ensure_pseudo_family_installed(workflow.pseudo_library)

    inputs = kcp_dscf_inputs(koopmans_input)

    extra_kwargs: dict[str, Any] = {}
    if workflow.init_orbitals in (
        VariationalOrbitalType.MLWFS,
        VariationalOrbitalType.PROJWFS,
    ):
        extra_kwargs = dscf_wannier_init_inputs(
            koopmans_input, next(iter(snapshots.values())), inputs["nbnd"]
        )
        codes, parallelization = load_codes(koopmans_input, codes, WANNIER_INIT_CODES)
        extra_kwargs["codes"] = dict(codes)

    if ml_mode != MLMode.NONE and ml_config.descriptor == MLDescriptor.POWER_SPECTRUM:
        # The descriptors come from a pw2wannier90.x decompose pass.
        codes, parallelization = load_codes(koopmans_input, codes, ["pw2wannier90"])
        extra_kwargs["pw2wannier90_code"] = codes["pw2wannier90"]
        extra_kwargs["decompose_parameters"] = _decompose_parameters(ml_config)

    return TrajectoryWorkflow.build(
        code=codes["kcp"],
        snapshots=snapshots,
        parallelization=parallelization or None,
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
) -> tuple[MLMode, dict[str, Any] | orm.Dict | None]:
    """Map the ``ml`` block onto a trajectory mode and its loaded model.

    ``test`` and ``predict`` modes take the model from exactly one of two
    sources: the stored node named by ``ml:model`` (set as the graph input
    ``orm.Dict`` itself, so the run's provenance links back to the
    training artifact) or the JSON file named by ``ml:model_file``.
    Predict-mode inputs that cannot take effect raise here: the
    ``power_spectrum`` descriptor (its decompose pass is not wired into
    the DSCF's screening stage, where the prediction runs) and
    ``alpha_numsteps != 1``.
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

    ml_model: dict[str, Any] | orm.Dict | None = None
    if ml_mode in (MLMode.TEST, MLMode.PREDICT):
        if ml_config.model is not None:
            ml_model = _load_model_node(ml_config.model)
        elif ml_config.model_file is not None:
            with open(ml_config.model_file) as handle:
                ml_model = json_load(handle)
        else:
            raise ValueError(
                f"ml:mode='{ml_mode.value}' requires a trained model: name its stored "
                "node via ml:model or its JSON copy via ml:model_file (both produced "
                "by a mode='train' run)."
            )
    return ml_mode, ml_model


def _load_model_node(identifier: int | str) -> orm.Dict:
    """Load the stored trained-model ``Dict`` node named by PK or UUID.

    The node is set as the trajectory graph's ``ml_model`` input, so the
    run's provenance links back to the training artifact; the DSCF
    sub-graphs receive its payload.
    """
    from aiida import orm

    raw = str(identifier)
    node = orm.load_node(int(raw) if raw.isdigit() else raw)
    if not isinstance(node, orm.Dict):
        raise ValueError(
            f"ml:model must name the stored trained-model Dict node (the `model` "
            f"output of a mode='train' run); node {raw} is a {type(node).__name__}."
        )
    return node


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
