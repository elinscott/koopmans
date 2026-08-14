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

What :func:`check_route_refusals` can and cannot see: it resolves every
refused path against the input-file models, and holds each shared refusal
to the owned keyword its field re-spells. It cannot tell that a refusal
has been *deleted* — nothing outside these tables says a given field needs
refusing on a given route — so a deletion is caught only by the cases
listed in ``_KCW_ROUTE_KEYWORDS`` in ``tests/test_input_file.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, NamedTuple, get_args

from aiida_koopmans.owned_keywords import OWNED, ROUTE_CONDITIONAL
from aiida_quantumespresso.common.types import SpinType
from pydantic import BaseModel

from koopmans.input_file.workflow import CalculateScreeningMethod, Task, WorkflowConfig

if TYPE_CHECKING:
    from koopmans.input_file import KoopmansInput

__all__ = [
    "ROUTE_REFUSALS",
    "SHARED_REFUSALS",
    "SharedRefusal",
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


class SharedRefusal(NamedTuple):
    """A refusal of a koopmans field, and the owned keyword it re-spells."""

    #: The block and keyword, as keyed in
    #: :data:`~aiida_koopmans.owned_keywords.OWNED`. Owning it is what
    #: leaves the field as the only spelling, and so what makes a refusal
    #: the reader's only warning.
    keyword: tuple[str, str]
    #: Returns the message to refuse a stated value with, or ``None`` to
    #: accept it.
    refuse: Callable[[WorkflowConfig, str], str | None]


#: Refusals for koopmans' own input-file fields, which no Quantum ESPRESSO
#: namelist spells and :data:`ROUTE_CONDITIONAL` therefore cannot declare.
SHARED_REFUSALS: dict[str, SharedRefusal] = {
    "calculator_parameters.tot_magnetization": SharedRefusal(
        keyword=("pw.SYSTEM", "tot_magnetization"),
        refuse=_refuse_shared_magnetization,
    ),
}


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    """Return the input-file block ``annotation`` describes, if it is one.

    Unwraps ``Annotated[...]`` and a union with ``None``; anything else
    that is not a model class returns ``None``.
    """
    for candidate in get_args(annotation) or (annotation,):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def _check_refused_path(path: str) -> None:
    """Check ``path`` names a field of the input file, block by block.

    Raises:
        ValueError: If a segment is not a field of the model above it. A
            path that names nothing refuses nothing, silently.
    """
    from koopmans.input_file import KoopmansInput

    owner: type[BaseModel] | None = KoopmansInput
    walked: list[str] = []
    for part in path.split("."):
        if owner is None:
            raise ValueError(
                f"{path} is refused, but `{'.'.join(walked)}` holds a value rather than a "
                f"block, so `{part}` names nothing. Correct the path in "
                f"koopmans.input_file._route_conditional."
            )
        field = owner.model_fields.get(part)
        if field is None:
            raise ValueError(
                f"{path} is refused, but `{part}` is not a field of "
                f"{owner.__name__}, so the refusal never fires. Correct the path in "
                f"koopmans.input_file._route_conditional."
            )
        walked.append(part)
        owner = _nested_model(field.annotation)


def check_route_refusals() -> None:
    """Check every route-conditional keyword has a refusal, and no more.

    Also checks each refused path names a real input-file field, and each
    shared refusal names a keyword the routes still own — owning it is what
    leaves the koopmans field as its only spelling.

    Raises:
        ValueError: If a keyword declared in
            :data:`~aiida_koopmans.owned_keywords.ROUTE_CONDITIONAL` has no
            refusal, a refusal names a keyword that is not declared, a
            refused path names no input-file field, or a shared refusal
            names a keyword no longer in
            :data:`~aiida_koopmans.owned_keywords.OWNED`.
    """
    for path, shared in SHARED_REFUSALS.items():
        block, keyword = shared.keyword
        if keyword not in OWNED.get(block, frozenset()):
            raise ValueError(
                f"{path} is refused as the koopmans spelling of {block}.{keyword}, which "
                f"aiida_koopmans.owned_keywords.OWNED no longer claims. The keyword is back "
                f"in the input file, so the refusal now hides one of its two spellings: drop "
                f"the entry from koopmans.input_file._route_conditional.SHARED_REFUSALS."
            )

    for paths in _refused_paths():
        for path in paths:
            _check_refused_path(path)

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


def _refused_paths() -> Iterator[dict[str, Callable[[WorkflowConfig, str], str | None]]]:
    """Yield every table of refused paths, namelist keywords and shared fields alike."""
    for block in ROUTE_REFUSALS.values():
        yield from block.values()
    yield {path: shared.refuse for path, shared in SHARED_REFUSALS.items()}


def raise_for_route_conditional(koopmans_input: KoopmansInput) -> None:
    """Refuse a stated keyword this input's route determines for itself.

    Raises:
        ValueError: If the input states one, naming the field and what to
            set instead.
    """
    for paths in _refused_paths():
        for path, refuse in paths.items():
            if not _is_stated(koopmans_input, path):
                continue
            message = refuse(koopmans_input.workflow, path)
            if message is not None:
                raise ValueError(message)
