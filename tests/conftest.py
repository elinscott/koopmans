"""Shared pytest fixtures for koopmans tests.

In addition to plain Pydantic-input tests, this conftest loads AiiDA's
pytest fixtures (``aiida_profile``, ``aiida_localhost``, ...) so dispatcher
regression tests can build real ``WorkGraph`` objects against a throwaway
profile without running a daemon.

Project-specific fixtures live in ``tests/fixtures.py`` and are re-exported
here so pytest's collection machinery picks them up for every test module.
Mirrors the layout used by the sibling ``aiida-koopmans2/tests/``.
"""

from __future__ import annotations

from tests.fixtures import (  # noqa: F401
    clear_database,
    clear_database_after_test,
    fake_sg15_cutoffs_family,
    fake_sg15_pseudo_family,
    installed_fold_codes,
    installed_kcp_code,
    installed_kcw_code,
    installed_ph_code,
    installed_pw_code,
    installed_wannier_codes,
    serialize_workgraph,
    tutorials_dir,
)

pytest_plugins = ["aiida.tools.pytest_fixtures"]
