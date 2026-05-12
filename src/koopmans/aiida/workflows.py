"""Workflow building logic for koopmans AiiDA integration.

This module handles selecting and constructing the appropriate AiiDA workgraph
based on the task specified in a KoopmansInput.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiida import orm

from koopmans.aiida.conversion import (
    atoms_input_to_structure,
    input_to_pw_parameters,
)
from koopmans.input_file.workflow import CalculateScreeningMethod, Correction, Task

if TYPE_CHECKING:
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput
    from koopmans.input_file.workflow import WorkflowConfig


def load_codes_for_task(workflow: WorkflowConfig) -> dict[str, orm.AbstractCode]:
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
        NotImplementedError: If the requested code combination is not ported yet.
    """
    task = workflow.task
    codes: dict[str, orm.AbstractCode] = {}

    # All tasks need pw.x
    try:
        codes["pw"] = orm.load_code("pw@localhost")
    except Exception as exc:
        raise ValueError(
            f"Could not load pw.x code: {exc}\n"
            "Please run 'koopmans install' first to set up the AiiDA backend."
        ) from exc

    # Singlepoint with a Koopmans correction needs a screening-method-specific code.
    if (
        task == Task.SINGLEPOINT
        and workflow.correction != Correction.NONE
        and workflow.calculate_alpha
    ):
        if workflow.screening_method == CalculateScreeningMethod.DSCF:
            try:
                codes["kcp"] = orm.load_code("kcp@localhost")
            except Exception as exc:
                raise ValueError(
                    f"Could not load kcp.x code: {exc}\n"
                    "Please run 'koopmans install' first to set up the AiiDA backend."
                ) from exc
        else:
            raise NotImplementedError(
                f"screening_method={workflow.screening_method.value!r} is not yet "
                f"ported. Only {CalculateScreeningMethod.DSCF.value!r} (which uses "
                "kcp.x) is implemented."
            )

    # Wannierize task needs additional codes
    if task == Task.WANNIERIZE:
        try:
            codes["pw2wannier90"] = orm.load_code("pw2wannier90@localhost")
        except Exception as exc:
            raise ValueError(
                f"Could not load pw2wannier90.x code: {exc}\n"
                "Please run 'koopmans install' first to set up the AiiDA backend."
            ) from exc

        try:
            codes["wannier90"] = orm.load_code("wannier90@localhost")
        except Exception as exc:
            raise ValueError(
                f"Could not load wannier90.x code: {exc}\n"
                "Please run 'koopmans install' first to set up the AiiDA backend."
            ) from exc

        # TODO: projwfc is only needed when the Wannierize flow computes a
        # projected DOS / bandstructure. Silently swallowing the lookup error
        # here lets workflows run without projwfc installed, but also masks
        # "projwfc required but missing" cases. Replace with a predicate on
        # the relevant workflow flag (likely `calculate_bands` or a
        # wannier90.bands_plot override) and insist on the code being
        # installed when that predicate fires.
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
    from koopmans.aiida.setup import ensure_pseudo_family_installed

    structure = atoms_input_to_structure(koopmans_input.atoms)
    parameters = input_to_pw_parameters(koopmans_input)
    pseudo_family = koopmans_input.workflow.pseudo_library

    ensure_pseudo_family_installed(pseudo_family)

    overrides: dict[str, Any] = {
        key: {
            "pseudo_family": pseudo_family,
            "pw": {
                "parameters": parameters,
            },
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

    # Load required codes
    codes = load_codes_for_task(koopmans_input.workflow)

    # Build the workgraph based on task
    if task == Task.DFT_BANDS:
        return _build_dft_bands_workgraph(koopmans_input, codes)
    elif task == Task.WANNIERIZE:
        return _build_wannierize_workgraph(koopmans_input, codes)
    elif task == Task.SINGLEPOINT:
        return _build_singlepoint_workgraph(koopmans_input, codes)
    else:
        raise ValueError(
            f"Task '{task.value}' is not yet implemented. "
            f"Supported tasks: {Task.DFT_BANDS.value}, {Task.WANNIERIZE.value}, "
            f"{Task.SINGLEPOINT.value}"
        )


def _build_dft_bands_workgraph(
    koopmans_input: KoopmansInput,
    codes: dict[str, orm.AbstractCode],
) -> WorkGraph:
    """Build a workgraph for DFT bands calculation.

    Args:
        koopmans_input: The parsed koopmans input.
        codes: Dictionary of loaded codes.

    Returns:
        A WorkGraph for PwBandsWorkChain.
    """
    from aiida_koopmans.workgraphs.pw import PwBandsTaskViaBuilder

    structure, _pseudo_family, overrides = _prepare_common_inputs(koopmans_input, ["scf", "bands"])

    return PwBandsTaskViaBuilder.build(
        code=codes["pw"],
        structure=structure,
        overrides=overrides,
    )


def _build_wannierize_workgraph(
    koopmans_input: KoopmansInput,
    codes: dict[str, orm.AbstractCode],
) -> WorkGraph:
    """Build a workgraph for Wannierization.

    Args:
        koopmans_input: The parsed koopmans input.
        codes: Dictionary of loaded codes.

    Returns:
        A WorkGraph for Wannier90WorkChain.
    """
    from aiida_koopmans.workgraphs.wannier90 import Wannier90TaskViaBuilder
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    structure, pseudo_family, overrides = _prepare_common_inputs(koopmans_input, ["scf", "nscf"])

    # Check if external projectors are requested
    pw2w_params = koopmans_input.calculator_parameters.pw2wannier90
    extra_kwargs: dict[str, Any] = {}
    if pw2w_params.atom_proj_ext:
        extra_kwargs["projection_type"] = WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL
        extra_kwargs["external_projectors_path"] = pw2w_params.atom_proj_dir

    return Wannier90TaskViaBuilder.build(
        codes=codes,
        structure=structure,
        overrides=overrides,
        pseudo_family=pseudo_family,
        print_summary=False,
        **extra_kwargs,
    )


def _build_singlepoint_workgraph(
    koopmans_input: KoopmansInput,
    codes: dict[str, orm.AbstractCode],
) -> WorkGraph:
    """Build a workgraph for a singlepoint Koopmans calculation.

    Dispatches on ``workflow.correction``:

    - ``KI`` → ``KoopmansDSCFTask`` (MVP: DFT init + KI correction, two kcp.x calls)
    - ``KIPZ``, ``PKIPZ``, ``NONE``, ``ALL`` → ``NotImplementedError`` (not yet ported)
    """
    from aiida_koopmans.workgraphs.kcp import KoopmansDSCFTask

    from koopmans.aiida.setup import ensure_pseudo_family_installed

    workflow = koopmans_input.workflow
    correction = workflow.correction
    if correction != Correction.KI:
        raise NotImplementedError(
            f"correction={correction.value!r} is not yet ported. "
            f"Only {Correction.KI.value!r} is implemented in the current MVP."
        )

    structure = atoms_input_to_structure(koopmans_input.atoms)
    pseudo_family = workflow.pseudo_library
    ensure_pseudo_family_installed(pseudo_family)

    ecutwfc, ecutrho, nbnd, nspin = _extract_kcp_scalar_inputs(koopmans_input)

    initial_alpha = (
        workflow.alpha_guess if isinstance(workflow.alpha_guess, float) else workflow.alpha_guess[0]
    )

    return KoopmansDSCFTask.build(
        code=codes["kcp"],
        structure=structure,
        pseudo_family=pseudo_family,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nbnd=nbnd,
        nspin=nspin,
        tot_magnetization=_coerce_optional_int(
            koopmans_input.calculator_parameters.tot_magnetization
        ),
        functional=correction.value,
        init_orbitals=workflow.init_orbitals.value,
        alpha_numsteps=workflow.alpha_numsteps,
        fix_spin_contamination=workflow.fix_spin_contamination,
        initial_alpha=initial_alpha,
        spin_polarized=workflow.spin_polarized,
    )


def _extract_kcp_scalar_inputs(
    koopmans_input: KoopmansInput,
) -> tuple[float, float, int, int]:
    """Pull ``(ecutwfc, ecutrho, nbnd, nspin)`` out of the ``KoopmansInput``.

    Prefers the top-level ``calculator_parameters.{ecutwfc,nbnd}`` convenience
    fields; falls back to the ``kcp.system`` Pydantic block when they are unset.
    ``ecutrho`` has no top-level convenience field — read from ``kcp.system``
    and default to ``4 * ecutwfc`` when unset.
    """
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
    nbnd = int(nbnd_raw)

    # KI requires nspin=2 for per-spin orbital-dependent screening, regardless of
    # what ``spin_polarized`` says — closed-shell molecules still need two channels.
    nspin = 2

    return float(ecutwfc), float(ecutrho), nbnd, nspin


def _coerce_optional_int(value: float | None) -> int | None:
    """Return ``int(value)`` when value is given, else ``None``."""
    return int(value) if value is not None else None
