"""Shared helpers used across input_file Pydantic models.

Kept out of ``__init__.py`` so sibling modules can import from it without
creating circular dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel


def reject_route_owned_fields(model: BaseModel, owners: dict[str, str], block: str) -> None:
    """Raise if ``model`` states a value for a field a workflow route derives itself.

    ``owners`` maps a field name to the input-file setting that determines
    its value; ``block`` is the dotted path to ``model`` in the input file,
    for the error message. Compares against the field's own declared
    default rather than ``model_fields_set``: ``model_dump()`` states every
    field explicitly, so a "was it written" check misfires on any input
    round-tripped through dump + re-validate (as
    ``KoopmansInput.model_copy`` — shallow — dumps, mutates, and
    re-validates).

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
