"""Dispatcher + workgraph-shape regression test for O2 (genuinely open-shell, KI-DSCF).

Parses ``tutorials/o2.json`` into a ``KoopmansInput``, calls
``build_workgraph`` to produce an ``aiida_workgraph.WorkGraph``, and
snapshots the resulting graph structure (task names, task count, serialized
task/port definitions with UUIDs scrubbed).

O2 with ``tot_magnetization=2`` is a genuine open-shell input
(``nelup != neldw``), exercising the per-spin asymmetric DSCF wiring
that closed-shell ozone-in-spin-polarized doesn't reach:

* ``build_filled_iter_source`` emits 7 UP + 5 DOWN filled keys (not
  symmetric 6+6).
* ``build_empty_iter_source`` emits 1 UP + 3 DOWN empty keys (not
  symmetric 2+2).
* ``generate_alphas`` returns per-spin-sized lists, not halved.
* ``fixed_band`` is clamped to the per-spin LUMO position and
  ``band_index`` uses the lambda-matrix block-diag offset
  ``max(nelup, neldw)`` (per ``aiida-koopmans2/workgraphs/kcp.py``).

No AiiDA daemon is started and no kcp.x is run — this is purely a
construction-level regression test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from koopmans.aiida.workflows import build_workgraph
from koopmans.input_file import KoopmansInput, read_input_file


@pytest.fixture
def o2_input(tutorials_dir: Path) -> KoopmansInput:
    """Parse the o2.json tutorial into a ``KoopmansInput``."""
    return read_input_file(tutorials_dir / "o2.json")


def test_build_workgraph(
    aiida_profile: Any,
    installed_pw_code: Any,
    installed_kcp_code: Any,
    fake_sg15_pseudo_family: Any,
    o2_input: KoopmansInput,
    serialize_workgraph: Any,
    data_regression: Any,
) -> None:
    """Snapshot the WorkGraph shape produced by the dispatcher for O2 (open-shell).

    The snapshot file (``test_tutorial_o2_ki/test_build_workgraph.yml``)
    is written on first run and needs human review before being committed.
    Subsequent runs fail if the dispatcher's wiring drifts.
    """
    workgraph = build_workgraph(o2_input)

    snapshot = serialize_workgraph(workgraph)

    # Fail-fast structural checks before the (large) snapshot comparison.
    assert snapshot["n_tasks"] >= 1, snapshot["task_names"]
    assert snapshot["workgraph_name"].startswith("KoopmansDSCFWorkflow"), snapshot["workgraph_name"]
    # Open-shell input skips the closed-shell 3-step spin-symmetric pre-pass;
    # only the single ``dft_init`` task should appear at the top level.
    assert "dft_init" in snapshot["task_names"], snapshot["task_names"]
    for forbidden in (
        "dft_init_nspin1",
        "dft_init_nspin2_dummy",
        "convert_spin1_to_spin2",
        "dft_init_nspin2",
    ):
        assert forbidden not in snapshot["task_names"], (
            forbidden,
            snapshot["task_names"],
        )
    expected_top_level = {
        "resolve_pseudo_family_task",
        "count_electrons_task",
        "dft_init",
        "ComputeScreeningParameters",
    }
    missing = expected_top_level - set(snapshot["task_names"])
    assert not missing, (missing, snapshot["task_names"])

    data_regression.check(snapshot)
