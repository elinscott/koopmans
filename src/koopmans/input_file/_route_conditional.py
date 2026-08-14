"""Refusals for the keywords one route determines and the others honour.

``aiida_koopmans.owned_keywords.ROUTE_CONDITIONAL`` names the Quantum
ESPRESSO keywords a single route forces on its own steps. They keep their
input-file spelling, because every other route honours them end to end;
what koopmans owes the reader is a refusal on the route that does not.

:data:`ROUTE_REFUSALS` pairs each declared keyword with the input-file
paths that spell it and the message each is refused with, and
:func:`check_route_refusals` raises if the two declarations have drifted
apart. :data:`SHARED_REFUSALS` carries the same for koopmans' own fields,
which no namelist spells and ``ROUTE_CONDITIONAL`` cannot declare.
:func:`raise_for_route_conditional` applies both to a parsed input.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from aiida_koopmans.owned_keywords import ROUTE_CONDITIONAL
from aiida_quantumespresso.common.types import SpinType

from koopmans.input_file.workflow import CalculateScreeningMethod, Task, WorkflowConfig

if TYPE_CHECKING:
    from koopmans.input_file import KoopmansInput

__all__ = [
    "ROUTE_REFUSALS",
    "SHARED_REFUSALS",
    "check_route_refusals",
    "raise_for_route_conditional",
]


def _runs_kcw(workflow: WorkflowConfig) -> bool:
    """Return whether ``workflow`` runs the kcw.x chain."""
    return (
        workflow.task == Task.SINGLEPOINT
        and workflow.screening_method == CalculateScreeningMethod.DFPT
    )


def _refuse_symmetry(workflow: WorkflowConfig, path: str) -> str | None:
    """Refuse a symmetry switch on the kcw.x route, whose nscf sets both."""
    if not _runs_kcw(workflow):
        return None
    return (
        f"`{path}` cannot be set with `workflow.screening_method = 'dfpt'`. kcw.x reads "
        "the Wannier functions off the whole k-point mesh, so koopmans runs the nscf "
        "with `nosym` and `noinv` switched on whatever you ask for, while the scf keeps "
        "your value — one keyword, two meanings. Remove it."
    )


def _refuse_shared_magnetization(workflow: WorkflowConfig, path: str) -> str | None:
    """Refuse the shared moment where a closed-shell kcw.x run forces zero."""
    if not _runs_kcw(workflow) or workflow.spin != SpinType.NONE:
        return None
    return (
        f"`{path}` cannot be set with `workflow.screening_method = 'dfpt'` and "
        "`workflow.spin = 'none'`. kcw.x needs a two-channel ground state even for a "
        "closed-shell system, so koopmans runs pw.x at zero moment. Set "
        "`workflow.spin = 'collinear'` if the system has one."
    )


#: For each keyword of :data:`~aiida_koopmans.owned_keywords.ROUTE_CONDITIONAL`,
#: the input-file paths that spell it, each mapped to the refusal it raises.
#: A refusal returns the message to refuse a stated value with, or ``None``
#: to accept it.
ROUTE_REFUSALS: dict[str, dict[str, dict[str, Callable[[WorkflowConfig, str], str | None]]]] = {
    "pw.SYSTEM": {
        "nosym": {"calculator_parameters.pw.system.nosym": _refuse_symmetry},
        "noinv": {"calculator_parameters.pw.system.noinv": _refuse_symmetry},
    },
}

#: Refusals for koopmans' own input-file fields, which no Quantum ESPRESSO
#: namelist spells and :data:`ROUTE_CONDITIONAL` therefore cannot declare.
SHARED_REFUSALS: dict[str, Callable[[WorkflowConfig, str], str | None]] = {
    "calculator_parameters.tot_magnetization": _refuse_shared_magnetization,
}


def check_route_refusals() -> None:
    """Check every route-conditional keyword has a refusal, and no more.

    Raises:
        ValueError: If a keyword declared in
            :data:`~aiida_koopmans.owned_keywords.ROUTE_CONDITIONAL` has no
            refusal, or a refusal names a keyword that is not declared.
    """
    for block in ROUTE_CONDITIONAL.keys() | ROUTE_REFUSALS.keys():
        declared = ROUTE_CONDITIONAL.get(block, frozenset())
        refused = ROUTE_REFUSALS.get(block, {})
        unrefused = sorted(declared - refused.keys())
        if unrefused:
            raise ValueError(
                f"{block}: {', '.join(unrefused)} would stay settable on the route that "
                f"forces it, and the stated value would be discarded without a word. Add "
                f"an entry to koopmans.input_file._route_conditional.ROUTE_REFUSALS."
            )
        stale = sorted(refused.keys() - declared)
        if stale:
            raise ValueError(
                f"{block}: {', '.join(stale)} is refused in ROUTE_REFUSALS but no longer "
                f"route-conditional in aiida_koopmans.owned_keywords.ROUTE_CONDITIONAL."
            )


def _is_stated(koopmans_input: KoopmansInput, path: str) -> bool:
    """Return whether the input file states ``path`` itself.

    ``path`` is dotted from the top of the input file. A field left to its
    default is not stated: the routes read the same distinction through
    ``model_dump(exclude_unset=True)``.
    """
    owner: Any = koopmans_input
    *parents, field = path.split(".")
    for part in parents:
        owner = getattr(owner, part)
    return field in owner.model_fields_set


def raise_for_route_conditional(koopmans_input: KoopmansInput) -> None:
    """Refuse a stated keyword this input's route determines for itself.

    Raises:
        ValueError: If the input states one, naming the field and what to
            set instead.
    """
    namelist = (refusals for block in ROUTE_REFUSALS.values() for refusals in block.values())
    for paths in (*namelist, SHARED_REFUSALS):
        for path, refuse in paths.items():
            if not _is_stated(koopmans_input, path):
                continue
            message = refuse(koopmans_input.workflow, path)
            if message is not None:
                raise ValueError(message)
