"""Shared helpers used across input_file Pydantic models.

Kept out of ``__init__.py`` so sibling modules can import from it without
creating circular dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel


def raise_for_owned_keywords(data: Any, block: str, owned: dict[str, str]) -> Any:
    """Return ``data`` unless it states a keyword koopmans determines itself.

    The generated input-file models (see ``koopmans.input_file._codegen``)
    do not declare these keywords, so ``extra='forbid'`` would already
    refuse them; this replaces that generic complaint with the reason and
    the field to set instead.

    Args:
        data: The raw block contents, before field validation.
        block: The dotted path of the block in a koopmans input file.
        owned: Keyword to explanation, as emitted alongside the model.

    Returns:
        ``data``, unchanged.

    Raises:
        ValueError: If ``data`` states an owned keyword.
    """
    if not isinstance(data, dict):
        return data
    stated = [keyword for keyword in data if keyword in owned]
    if stated:
        keyword = stated[0]
        raise ValueError(f"`{block}.{keyword}` is not a koopmans keyword. {owned[keyword]}")
    return data


def reject_route_owned_fields(model: BaseModel, owners: dict[str, str], block: str) -> None:
    """Raise if ``model`` states a value for a field a workflow route derives itself.

    ``owners`` maps a field name to the input-file setting that determines
    its value; ``block`` is the dotted path to ``model`` in the input file,
    for the error message. Compares against the field's own declared
    default rather than ``model_fields_set``: ``model_dump()`` states every
    field explicitly, so a "was it written" check misfires on any input
    round-tripped through ``model_dump()`` -> ``model_validate()`` — a
    pattern koopmans itself uses to re-validate a modified input (e.g.
    ``KoopmansInput.model_copy``, which dumps, mutates, and re-validates).
    A field the caller always forces to a value other than its inherited
    default should redeclare that default on the subclass, so "the field's
    own declared default" and "what actually runs" agree.

    Raises:
        ValueError: If a key of ``owners`` is set away from its default.
    """
    for field, owner in owners.items():
        default = type(model).model_fields[field].get_default(call_default_factory=True)
        if getattr(model, field) != default:
            raise ValueError(f"`{block}.{field}` is set by `{owner}`; do not set it directly.")


def tidy_units(value: str) -> str:
    """Normalize unit strings to a canonical form.

    Lowercases the input and maps common aliases (``angstrom`` → ``ang``).
    Used as a Pydantic ``BeforeValidator`` on ``units`` fields across the
    input_file models.
    """
    value = value.lower()
    value = value.replace("angstrom", "ang")
    return value
