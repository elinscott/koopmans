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
    # The root ``@task.graph`` is called KoopmansDSCFTask; inside we expect the
    # DFT initialization sub-graph plus a kcp.x call for the KI correction.
    assert snapshot["workgraph_name"].startswith("KoopmansDSCFTask"), snapshot["workgraph_name"]
    # Sub-tasks carry the descriptive names threaded through ``call_link_label``
    # so they show up sensibly in ``verdi process list`` and the koopmans
    # progress display.
    assert "dft_init" in snapshot["task_names"], snapshot["task_names"]
    assert "ki_final" in snapshot["task_names"], snapshot["task_names"]

    data_regression.check(snapshot)


def test_dispatcher_rejects_non_ki_correction(
    aiida_profile,
    installed_pw_code,
    installed_kcp_code,
    tutorial_1_ozone_input,
):
    """``build_workgraph`` should raise ``NotImplementedError`` for KIPZ/PKIPZ.

    Exercises the dispatcher's scope guard without entering the
    ``KoopmansDSCFTask`` body.
    """
    from koopmans.input_file import KoopmansInput

    d = tutorial_1_ozone_input.model_dump()
    d["workflow"]["correction"] = "kipz"
    inp = KoopmansInput.model_validate(d)

    with pytest.raises(NotImplementedError, match="correction="):
        build_workgraph(inp)
