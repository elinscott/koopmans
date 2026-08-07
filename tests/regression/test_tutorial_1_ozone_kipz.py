"""Dispatcher smoke test for tutorial_1 / ozone driven by the KIPZ correction.

Takes the ozone tutorial's own input, swaps its correction for KIPZ, and
verifies that the dispatcher accepts ``correction=kipz`` and emits the same
top-level shape as the KI build (the KI vs KIPZ difference lives inside
the ``ComputeScreeningParameters`` sub-graph, in the parameter dicts of
the alpha-step builders — covered by ``aiida-koopmans2/tests/test_kcp_workgraph.py``).

This test deliberately does *not* snapshot the WorkGraph: the
construction-level regression for KI already pins the top-level shape;
KIPZ uses the same builders and only the inner CalcJob parameter dicts
diverge. A snapshot here would mostly just double-pin the dispatcher.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from koopmans.aiida.workflows import build_workgraph
from koopmans.input_file import KoopmansInput, read_input_file
from koopmans.input_file.workflow import Correction


@pytest.fixture
def tutorial_1_ozone_kipz_input(tutorials_dir: Path) -> KoopmansInput:
    """Return the ozone tutorial's input, with KIPZ in place of KI.

    The tutorial ships one input file; the correction is the only thing
    this test needs to differ, so it is swapped here rather than kept as a
    second copy of the same calculation.
    """
    inp = read_input_file(tutorials_dir / "orbital_energies/ozone/ozone.yaml")
    dumped = inp.model_dump()
    dumped["workflow"]["correction"] = Correction.KIPZ
    return KoopmansInput.model_validate(dumped)


def test_dispatcher_accepts_kipz_correction(
    aiida_profile: Any,
    installed_dscf_codes: Any,
    fake_sg15_pseudo_family: Any,
    tutorial_1_ozone_kipz_input: KoopmansInput,
    serialize_workgraph: Any,
) -> None:
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
