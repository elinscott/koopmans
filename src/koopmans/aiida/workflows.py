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
from koopmans.input_file.workflow import Task

if TYPE_CHECKING:
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput


def load_codes_for_task(task: Task) -> dict[str, orm.AbstractCode]:
    """Load the required AiiDA codes for a given task.

    Args:
        task: The workflow task to run.

    Returns:
        Dictionary mapping code names to Code instances.

    Raises:
        ValueError: If required codes are not found.
    """
    codes: dict[str, orm.AbstractCode] = {}

    # All tasks need pw.x
    try:
        codes["pw"] = orm.load_code("pw@localhost")
    except Exception as exc:
        raise ValueError(
            f"Could not load pw.x code: {exc}\n"
            "Please run 'koopmans install' first to set up the AiiDA backend."
        ) from exc

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

        # projwfc is optional but commonly used
        try:
            codes["projwfc"] = orm.load_code("projwfc@localhost")
        except Exception:
            pass  # projwfc is optional

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
                "parameters": parameters.get_dict(),
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
    codes = load_codes_for_task(task)

    # Build the workgraph based on task
    if task == Task.DFT_BANDS:
        return _build_dft_bands_workgraph(koopmans_input, codes)
    elif task == Task.WANNIERIZE:
        return _build_wannierize_workgraph(koopmans_input, codes)
    else:
        raise ValueError(
            f"Task '{task.value}' is not yet implemented. "
            f"Supported tasks: {Task.DFT_BANDS.value}, {Task.WANNIERIZE.value}"
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

    structure, _pseudo_family, overrides = _prepare_common_inputs(
        koopmans_input, ["scf", "bands"]
    )

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
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    from aiida_koopmans.workgraphs.wannier90 import Wannier90TaskViaBuilder

    structure, pseudo_family, overrides = _prepare_common_inputs(
        koopmans_input, ["scf", "nscf"]
    )

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
