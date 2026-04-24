"""Shared pytest fixtures for koopmans tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tutorials_dir() -> Path:
    """Return the path to the tutorials directory shipped with the docs."""
    return Path(__file__).parent.parent / "docs" / "source" / "tutorials"
