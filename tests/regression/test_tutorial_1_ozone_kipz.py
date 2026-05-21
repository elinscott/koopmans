"""Dispatcher smoke test for tutorial_1 / ozone driven by the KIPZ correction.

Parses ``tutorials/ozone_kipz.json`` into a ``KoopmansInput`` and verifies
that the dispatcher accepts ``correction=kipz`` and emits the same
top-level shape as the KI build (the KI vs KIPZ difference lives inside
the ``ComputeScreeningParameters`` sub-graph, in the parameter dicts of
the alpha-step builders — covered by ``aiida-koopmans2/tests/test_kcp_workgraph.py``).

This test deliberately does *not* snapshot the WorkGraph: the
construction-level regression for KI already pins the top-level shape;
KIPZ uses the same builders and only the inner CalcJob parameter dicts
diverge. A snapshot here would mostly just double-pin the dispatcher.
"""

from __future__ import annotations

import pytest

from koopmans.aiida.workflows import build_workgraph
from koopmans.input_file import read_input_file


@pytest.fixture
def tutorial_1_ozone_kipz_input(tutorials_dir):
    """Parse the ozone_kipz.json tutorial into a ``KoopmansInput``."""
    return read_input_file(tutorials_dir / "ozone_kipz.json")


def test_dispatcher_accepts_kipz_correction(
    aiida_profile,
    installed_pw_code,
    installed_kcp_code,
    fake_sg15_pseudo_family,
    tutorial_1_ozone_kipz_input,
    serialize_workgraph,
):
    """``build_workgraph`` should produce a valid graph for ``correction=kipz``."""
    workgraph = build_workgraph(tutorial_1_ozone_kipz_input)

    snapshot = serialize_workgraph(workgraph)

    assert snapshot["workgraph_name"].startswith("KoopmansDSCFWorkflow"), snapshot["workgraph_name"]
    # Top-level shape identical to the KI build: the KIPZ-specific
    # differences are confined to the inner parameter dicts.
    expected_top_level = {
        "resolve_pseudo_family_task",
        "count_electrons_task",
        "dft_init_nspin1",
        "dft_init_nspin2_dummy",
        "convert_spin1_to_spin2",
        "dft_init_nspin2",
        "ComputeScreeningParameters",
    }
    missing = expected_top_level - set(snapshot["task_names"])
    assert not missing, (missing, snapshot["task_names"])
