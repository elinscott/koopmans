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
    assert_ranks_settled_for_every_loaded_code,
    binary_probe,
    clear_database,
    clear_database_after_test,
    code_without_mpi_flag,
    compiled_binaries,
    fake_pseudodojo_lda_family,
    fake_sg15_cutoffs_family,
    fake_sg15_fr_cutoffs_family,
    fake_sg15_pseudo_family,
    hyperqueue_localhost_unpatched,
    installed_decompose_code,
    installed_dfpt_codes,
    installed_dscf_codes,
    installed_fold_codes,
    installed_kcp_code,
    installed_kcw_code,
    installed_ph_code,
    installed_projwfc_code,
    installed_pw_code,
    installed_wannier_codes,
    installed_wannierize_codes,
    installed_wannierjl_code,
    localhost_code,
    localhost_computer,
    localhost_default_ranks,
    replay_probes,
    serialize_workgraph,
    si_external_projector_dir,
    stub_executable,
    tutorials_dir,
    write_multiframe_xyz,
)

pytest_plugins = ["aiida.tools.pytest_fixtures"]
