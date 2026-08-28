"""Shared helpers used across input_file Pydantic models.

Kept out of ``__init__.py`` so sibling modules can import from it without
creating circular dependencies.
"""

from __future__ import annotations

from typing import Any


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


def raise_for_unreachable_keywords(data: Any, block: str, unreachable: dict[str, str]) -> Any:
    """Return ``data`` unless it states a keyword koopmans cannot pass through.

    Each such keyword is a defect in the generic model the block is
    generated from, not a koopmans decision: the generated model does not
    declare it, and this says why rather than leaving ``extra='forbid'`` to
    complain about an unknown field.

    Args:
        data: The raw block contents, before field validation.
        block: The dotted path of the block in a koopmans input file.
        unreachable: Keyword to explanation, as emitted alongside the model.

    Returns:
        ``data``, unchanged.

    Raises:
        ValueError: If ``data`` states an unreachable keyword.
    """
    if not isinstance(data, dict):
        return data
    stated = [keyword for keyword in data if keyword in unreachable]
    if stated:
        keyword = stated[0]
        raise ValueError(
            f"`{block}.{keyword}` cannot be set from a koopmans input file. {unreachable[keyword]}"
        )
    return data


def tidy_units(value: str) -> str:
    """Normalize unit strings to a canonical form.

    Lowercases the input and maps common aliases (``angstrom`` → ``ang``).
    Used as a Pydantic ``BeforeValidator`` on ``units`` fields across the
    input_file models.
    """
    value = value.lower()
    value = value.replace("angstrom", "ang")
    return value
