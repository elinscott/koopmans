"""Dispatcher + workgraph-shape regression test for tutorial_1 (ozone, KI-DSCF).

Parses ``tutorials/ozone.json`` into a ``KoopmansInput``, calls
``build_workgraph`` to produce an ``aiida_workgraph.WorkGraph``, and
snapshots the resulting graph structure (task names, task count, serialized
task/port definitions with UUIDs scrubbed).

No AiiDA daemon is started and no kcp.x is run — this is purely a
construction-level regression test. The companion CalcJob and parser
regression tests live in ``aiida-koopmans2/tests/``.
"""

from __future__ import annotations

import pytest

from koopmans.aiida.workflows import build_workgraph
from koopmans.input_file import read_input_file


@pytest.fixture
def tutorial_1_ozone_input(tutorials_dir):
    """Parse the ozone.json tutorial into a ``KoopmansInput``."""
    return read_input_file(tutorials_dir / "ozone.json")


def test_build_workgraph(
    aiida_profile,
    installed_pw_code,
    installed_kcp_code,
    fake_sg15_pseudo_family,
    tutorial_1_ozone_input,
    serialize_workgraph,
    data_regression,
):
    """Snapshot the WorkGraph shape produced by the dispatcher for tutorial_1 ozone.

    The snapshot file (``test_tutorial_1_ozone_ki/test_build_workgraph.yml``)
    is written on first run and will need human review before being
    committed. Subsequent runs fail if the dispatcher's wiring drifts.
    """
    workgraph = build_workgraph(tutorial_1_ozone_input)

    snapshot = serialize_workgraph(workgraph)

    # Assertions that the snapshot-free structure is consistent with the
    # tutorial_1 / KI-DSCF expectation. These fail fast and cheaply in case
    # the dispatcher suddenly returns an unexpected object.
    assert snapshot["n_tasks"] >= 1, snapshot["task_names"]
    # The root ``@task.graph`` is called KoopmansDSCFWorkflow; inside we expect
    # the spin-symmetric DFT initialization chain (4 sub-graphs) plus a
    # nested ``ComputeScreeningParameters`` sub-graph holding the trial KI, the
    # per-orbital DSCF fan-out, and the final KI.
    assert snapshot["workgraph_name"].startswith("KoopmansDSCFWorkflow"), snapshot["workgraph_name"]
    # Top-level structural tasks at the dispatcher layer. ``ki_trial`` /
    # ``ki_final`` are *not* top-level — they live inside the
    # ``ComputeScreeningParameters`` sub-graph (whose internals are visible
    # via the scrubbed ``raw`` payload below).
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

    data_regression.check(snapshot)


def test_dispatcher_rejects_non_ki_correction(
    aiida_profile,
    installed_pw_code,
    installed_kcp_code,
    tutorial_1_ozone_input,
):
    """``build_workgraph`` should raise ``NotImplementedError`` for KIPZ/PKIPZ.

    Exercises the dispatcher's scope guard without entering the
    ``KoopmansDSCFWorkflow`` body.
    """
    from koopmans.input_file import KoopmansInput

    d = tutorial_1_ozone_input.model_dump()
    d["workflow"]["correction"] = "kipz"
    inp = KoopmansInput.model_validate(d)

    with pytest.raises(NotImplementedError, match="correction="):
        build_workgraph(inp)
