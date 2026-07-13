"""Shared helpers used across input_file Pydantic models.

Kept out of ``__init__.py`` so sibling modules can import from it without
creating circular dependencies.
"""

from __future__ import annotations


def tidy_units(value: str) -> str:
    """Normalize unit strings to a canonical form.

    Lowercases the input and maps common aliases (``angstrom`` → ``ang``).
    Used as a Pydantic ``BeforeValidator`` on ``units`` fields across the
    input_file models.
    """
    value = value.lower()
    value = value.replace("angstrom", "ang")
    return value
