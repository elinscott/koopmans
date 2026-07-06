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


def test_build_workgraph_multi_iteration(
    aiida_profile,
    installed_pw_code,
    installed_kcp_code,
    fake_sg15_pseudo_family,
    tutorial_1_ozone_input,
    serialize_workgraph,
):
    """Dispatcher accepts ``alpha_numsteps > 1``.

    Rebuilds the tutorial input with ``alpha_numsteps=2`` and verifies
    that ``build_workgraph`` returns successfully and the top-level
    shape is unchanged (the ``While`` zone is wrapped inside
    ``ComputeScreeningParameters``, not at the dispatcher layer).

    The actual ``While`` zone wiring is verified in
    ``aiida-koopmans2/tests/test_kcp_workgraph.py`` (which builds
    ``ComputeScreeningParameters`` directly and can inspect its
    internals); this test is the koopmans2-level guard against a
    regression in the ``_validate_scope`` gate or the parameter
    plumbing.
    """
    from koopmans.input_file import KoopmansInput

    d = tutorial_1_ozone_input.model_dump()
    d["workflow"]["alpha_numsteps"] = 2
    inp = KoopmansInput.model_validate(d)

    workgraph = build_workgraph(inp)
    snapshot = serialize_workgraph(workgraph)
    assert "ComputeScreeningParameters" in snapshot["task_names"], snapshot["task_names"]


def test_build_workgraph_spin_polarized(
    aiida_profile,
    installed_pw_code,
    installed_kcp_code,
    fake_sg15_pseudo_family,
    tutorial_1_ozone_input,
    serialize_workgraph,
):
    """Dispatcher accepts ``spin='collinear'``.

    Rebuilds the ozone input with ``spin='collinear'`` and verifies
    that ``build_workgraph`` returns successfully. Ozone is physically
    closed-shell, so this is a *smoke test* for the spin-polarised code
    path — exercises:

    * the single-step DFT init branch (no spin-symmetric pre-pass);
    * ``generate_alphas`` emitting both UP and DOWN channels;
    * the per-orbital Map-zone fan-out doubling (UP + DOWN orbitals
      instead of a single representative channel);
    * the N+1 spin-direction branch (added separately).

    Deeper structural assertions about the per-spin fan-out live in
    ``aiida-koopmans2/tests/test_kcp_workgraph.py``.
    """
    from koopmans.input_file import KoopmansInput

    d = tutorial_1_ozone_input.model_dump()
    d["workflow"]["spin"] = "collinear"
    inp = KoopmansInput.model_validate(d)

    workgraph = build_workgraph(inp)
    snapshot = serialize_workgraph(workgraph)
    # Spin-polarised init skips the closed-shell 3-step pre-pass; the
    # single ``dft_init`` task appears at the top level.
    assert "dft_init" in snapshot["task_names"], snapshot["task_names"]
    assert "dft_init_nspin1" not in snapshot["task_names"], snapshot["task_names"]
    assert "convert_spin1_to_spin2" not in snapshot["task_names"], snapshot["task_names"]
    assert "ComputeScreeningParameters" in snapshot["task_names"], snapshot["task_names"]


def test_dispatcher_rejects_unsupported_correction(
    aiida_profile,
    installed_pw_code,
    installed_kcp_code,
    tutorial_1_ozone_input,
):
    """``build_workgraph`` should raise ``NotImplementedError`` for PKIPZ.

    Exercises the dispatcher's scope guard without entering the
    ``KoopmansDSCFWorkflow`` body.
    """
    from koopmans.input_file import KoopmansInput

    d = tutorial_1_ozone_input.model_dump()
    d["workflow"]["correction"] = "pkipz"
    inp = KoopmansInput.model_validate(d)

    with pytest.raises(NotImplementedError, match="correction="):
        build_workgraph(inp)
