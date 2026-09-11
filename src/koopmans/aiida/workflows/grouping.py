"""Translation of the orbital-grouping keywords into per-route tolerances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koopmans.input_file.workflow import (
    CalculateScreeningMethod,
    GroupOrbitalsBy,
    VariationalOrbitalType,
)

if TYPE_CHECKING:
    from koopmans.input_file.workflow import WorkflowConfig

#: Default tolerance per grouping criterion, applied when the input file
#: chooses a criterion but leaves the tolerance unset.
_DEFAULT_TOLERANCE: dict[GroupOrbitalsBy, float] = {
    GroupOrbitalsBy.SELF_HARTREE: 1.0e-4,
    GroupOrbitalsBy.SPREAD: 0.05,
}


def resolve_orbital_grouping(workflow: WorkflowConfig) -> tuple[GroupOrbitalsBy, float | None]:
    """Resolve the effective orbital-grouping criterion and tolerance.

    Left unset, ``group_orbitals_by`` resolves to ``self_hartree`` for
    Wannier-initialized DSCF runs — supercell images of one primitive
    orbital are physically equivalent and must share a screening parameter
    — and ``none`` otherwise (grouping is opt-in elsewhere); an explicit
    criterion passes through unchanged. A tolerance left unset then takes
    the criterion's default (1e-4 eV for self_hartree, 0.05 Angstrom^2 for
    spread); an explicit tolerance passes through unchanged. The criterion
    is in principle independent of the screening method — the defaults
    simply reflect the combinations wired up today, and ``grouping_tol``/
    ``dfpt_grouping_tol`` reject the rest explicitly.
    """
    criterion = workflow.group_orbitals_by
    if criterion is None:
        wannier_init = workflow.init_orbitals in (
            VariationalOrbitalType.MLWFS,
            VariationalOrbitalType.PROJWFS,
        )
        dscf = workflow.screening_method == CalculateScreeningMethod.DSCF
        criterion = (
            GroupOrbitalsBy.SELF_HARTREE if (wannier_init and dscf) else GroupOrbitalsBy.NONE
        )
    tol = workflow.group_orbitals_tol
    if criterion != GroupOrbitalsBy.NONE and tol is None:
        tol = _DEFAULT_TOLERANCE[criterion]
    return criterion, tol


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


def grouping_tol(workflow: WorkflowConfig) -> float | None:
    """Translate the orbital-grouping fields into the plugin's self-Hartree tolerance.

    Resolves ``group_orbitals_by`` / ``group_orbitals_tol`` (including
    their route-dependent defaults) via :func:`resolve_orbital_grouping`;
    only the implemented criterion passes through.
    """
    _reject_explicit_orbital_groups(workflow)
    criterion, tol = resolve_orbital_grouping(workflow)
    if criterion == GroupOrbitalsBy.NONE:
        return None
    if criterion == GroupOrbitalsBy.SELF_HARTREE:
        return tol
    raise NotImplementedError(
        f"group_orbitals_by={criterion.value!r} is not implemented; "
        "supported: 'self_hartree', 'none'."
    )


def dfpt_grouping_tol(workflow: WorkflowConfig) -> float | None:
    """Resolve the workflow-level orbital-grouping tolerance for the DFPT route.

    Returns the tolerance for ``'spread'`` (grouping on), ``None`` for
    ``'none'`` / unset (no workflow-level grouping), and raises for
    ``'self_hartree'``, which the DFPT route has no metric for.
    """
    _reject_explicit_orbital_groups(workflow)
    criterion, tol = resolve_orbital_grouping(workflow)
    if criterion == GroupOrbitalsBy.NONE:
        return None
    if criterion == GroupOrbitalsBy.SPREAD:
        return tol
    raise NotImplementedError(
        f"group_orbitals_by={criterion.value!r} is not implemented for DFPT "
        "screening: the DFPT route clusters orbitals by their wannier90 spread. "
        "Use group_orbitals_by = 'spread' (or 'none')."
    )
