"""Translation of the orbital-grouping keywords into per-route tolerances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koopmans.input_file.workflow import GroupOrbitalsBy

if TYPE_CHECKING:
    from koopmans.input_file.workflow import WorkflowConfig


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
