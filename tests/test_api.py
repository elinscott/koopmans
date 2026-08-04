"""The public python API: the build/run/submit triple and Results.

No calculation runs here: ``build`` is checked against the dispatcher it
wraps, the two launching verbs are checked to funnel through the single
internal launch helper, and ``Results`` is read off a hand-built finished
node shaped like a real ``KoopmansDSCFWorkflow`` one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import koopmans
from koopmans import KoopmansInput, Results, read_input_file
from koopmans.api import launch


class TestPublicSurface:
    """The package's top level exports the documented names."""

    def test_all_names_importable(self) -> None:
        """Every ``__all__`` name resolves on the package."""
        for name in koopmans.__all__:
            assert getattr(koopmans, name) is not None
        assert set(koopmans.__all__) >= {"build", "run", "submit", "Results", "KoopmansInput"}


class TestBuild:
    """``build`` materializes the same graph as the dispatcher."""

    def test_build_matches_build_workgraph(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_kcp_code: Any,
        fake_sg15_pseudo_family: Any,
        tutorials_dir: Path,
    ) -> None:
        """The ozone input builds the same task set through both entry points.

        Also pins that ``build`` leaves the already-loaded (test) profile
        alone rather than switching to the installed koopmans one.
        """
        from koopmans.aiida.workflows import build_workgraph

        inp = read_input_file(tutorials_dir / "ozone.json")
        via_api = koopmans.build(inp)
        via_dispatcher = build_workgraph(inp)
        assert via_api.get_task_names() == via_dispatcher.get_task_names()


class TestLaunchFunnel:
    """Both launching verbs go through the one internal launch helper."""

    def test_run_and_submit_route_through_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``run`` launches blocking, ``submit`` non-blocking with its wait flag."""
        calls: list[tuple[Any, bool, bool]] = []

        def fake_launch(workgraph: Any, *, blocking: bool, wait: bool = False) -> Results:
            """Record the launch request."""
            calls.append((workgraph, blocking, wait))
            return Results(object())  # type: ignore[arg-type]

        sentinel = object()
        monkeypatch.setattr("koopmans.api.launch", fake_launch)
        monkeypatch.setattr("koopmans.api.build", lambda inp: sentinel)

        assert isinstance(koopmans.run("unused"), Results)  # type: ignore[arg-type]
        assert isinstance(koopmans.submit("unused", wait=True), Results)  # type: ignore[arg-type]
        assert calls == [(sentinel, True, False), (sentinel, False, True)]

    def test_launch_calls_the_workgraph_verbs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The helper maps blocking onto ``.run()`` and not onto ``.submit()``.

        The daemon check guards only the submit branch: an in-interpreter
        run must not require a daemon.
        """
        import koopmans.aiida.setup.daemon as daemon_module

        daemon_checks: list[bool] = []
        monkeypatch.setattr(
            daemon_module, "ensure_daemon_running", lambda: daemon_checks.append(True)
        )

        class FakeWorkGraph:
            """Record which verb the launch helper invokes."""

            process = "the-node"

            def __init__(self) -> None:
                self.calls: list[tuple[str, Any]] = []

            def run(self) -> None:
                """Record an in-interpreter run."""
                self.calls.append(("run", None))

            def submit(self, wait: bool = False) -> None:
                """Record a daemon submission and its wait flag."""
                self.calls.append(("submit", wait))

        blocking_wg = FakeWorkGraph()
        results = launch(blocking_wg, blocking=True)
        assert blocking_wg.calls == [("run", None)]
        assert isinstance(results, Results)
        assert not daemon_checks

        daemon_wg = FakeWorkGraph()
        launch(daemon_wg, blocking=False, wait=True)
        assert daemon_wg.calls == [("submit", True)]
        assert daemon_checks == [True]


def _finished_dscf_node(
    *,
    exit_status: int = 0,
    with_outputs: bool = True,
    with_calcjob: bool = False,
    remote_computer: Any = None,
    sealed: bool = True,
) -> Any:
    """Build a node shaped like a finished ``KoopmansDSCFWorkflow`` one.

    The output links (labels, node types, array names) mirror what a real
    finished run stores, so the accessor is exercised against the shapes it
    will meet.
    """
    from aiida import orm
    from aiida.common.links import LinkType
    from aiida.engine import ProcessState

    node = orm.WorkChainNode(label="fake-dscf")
    node.set_process_label("WorkGraph<KoopmansDSCFWorkflow>")
    if sealed:
        node.set_process_state(ProcessState.FINISHED)
        node.set_exit_status(exit_status)
    else:
        node.set_process_state(ProcessState.RUNNING)
    node.store()

    if with_outputs and sealed and exit_status == 0:
        parameters = orm.Dict(  # type: ignore[no-untyped-call]
            {
                "energy": -1296.39,
                "energy_units": "eV",
                "homo_energy": -12.52,
                "lumo_energy": -1.82,
            }
        )
        eigenvalues = orm.ArrayData()
        eigenvalues.set_array("eigenvalues", np.array([[-40.2, -12.52], [-40.2, -12.52]]))
        outputs = {
            "parameters": parameters,
            "eigenvalues": eigenvalues,
            "alphas__filled": orm.Dict({"none": [0.66, 0.73]}),  # type: ignore[no-untyped-call]
            "alphas__empty": orm.Dict({"none": [0.72]}),  # type: ignore[no-untyped-call]
        }
        for label, data in outputs.items():
            data.store()
            data.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label=label)
    if remote_computer is not None:
        remote = orm.RemoteData(remote_path="/scratch/run", computer=remote_computer)
        remote.store()
        remote.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label="remote_folder")
    if with_calcjob:
        calc = orm.CalcJobNode()
        calc.set_process_state(ProcessState.FINISHED)
        calc.set_exit_status(0)
        calc.base.repository.put_object_from_bytes(b"output data\n", "aiida.cpo")
        calc.base.links.add_incoming(node, link_type=LinkType.CALL_CALC, link_label="ki_final")
        calc.store()
        calc.seal()
    if sealed:
        node.seal()
    return node


class TestResults:
    """``Results`` reads a finished calculation in user vocabulary."""

    def test_reads_the_dscf_outputs(self, aiida_profile: Any) -> None:
        """Energies, orbital energies and alphas come back as plain values."""
        results = Results(_finished_dscf_node())
        assert results.finished
        assert results.total_energy == pytest.approx(-1296.39)
        assert results.homo_energy == pytest.approx(-12.52)
        assert results.lumo_energy == pytest.approx(-1.82)
        assert results.ionization_potential == pytest.approx(12.52)
        assert results.electron_affinity == pytest.approx(1.82)
        assert results.orbital_energies.shape == (2, 2)
        assert results.alphas == {
            "filled": {"none": [0.66, 0.73]},
            "empty": {"none": [0.72]},
        }

    def test_dump_writes_the_per_step_layout(self, aiida_profile: Any, tmp_path: Path) -> None:
        """``dump`` writes one folder per calculation step and returns the path."""
        results = Results(_finished_dscf_node(with_calcjob=True))
        out = results.dump(tmp_path / "ozone")
        assert out == tmp_path / "ozone"
        step_dirs = [p.name for p in out.rglob("*") if p.is_dir()]
        assert any("ki_final" in name for name in step_dirs), step_dirs

    def test_outputs_deserializes_every_socket(
        self, aiida_profile: Any, localhost_computer: Any
    ) -> None:
        """``outputs`` returns plain nested values; file handles are omitted."""
        node = _finished_dscf_node(remote_computer=localhost_computer)
        outputs = Results(node).outputs
        assert outputs["parameters"]["energy"] == pytest.approx(-1296.39)
        assert outputs["eigenvalues"].shape == (2, 2)
        assert outputs["alphas"] == {"filled": {"none": [0.66, 0.73]}, "empty": {"none": [0.72]}}
        assert "remote_folder" not in outputs

    def test_from_pk_reconnects(self, aiida_profile: Any) -> None:
        """The integer id round-trips to a working accessor."""
        node = _finished_dscf_node()
        results = Results.from_pk(node.pk)
        assert results.pk == node.pk
        assert results.total_energy == pytest.approx(-1296.39)

    def test_running_calculation_refuses_to_read(self, aiida_profile: Any) -> None:
        """A still-running calculation raises rather than returning stale data."""
        results = Results(_finished_dscf_node(sealed=False))
        assert not results.finished
        with pytest.raises(RuntimeError, match="still running"):
            _ = results.total_energy

    def test_failed_calculation_refuses_to_read(self, aiida_profile: Any) -> None:
        """A failed calculation raises, naming its exit status."""
        results = Results(_finished_dscf_node(exit_status=302, with_outputs=False))
        with pytest.raises(RuntimeError, match=r"failed \(exit status 302\)"):
            _ = results.total_energy

    def test_unmapped_task_names_the_gap(self, aiida_profile: Any) -> None:
        """A graph without the DSCF sockets raises NotImplementedError."""
        results = Results(_finished_dscf_node(with_outputs=False))
        with pytest.raises(NotImplementedError, match="singlepoint"):
            _ = results.total_energy


class TestInputConstructibleInPython:
    """A ``KoopmansInput`` built in python matches the file-parsed one."""

    def test_model_validate_equals_file_parse(self, tutorials_dir: Path) -> None:
        """The documented in-python construction path produces the same model."""
        from json import load

        with open(tutorials_dir / "ozone.json") as handle:
            raw = load(handle)
        assert KoopmansInput.model_validate(raw) == read_input_file(tutorials_dir / "ozone.json")


def test_launch_verbs_only_in_the_funnel() -> None:
    """No source module outside the funnel calls the workgraph launch verbs.

    A naming-convention tripwire (workgraph variables are ``wg`` /
    ``workgraph`` throughout the package): it is what catches a stray
    ``wg.submit()`` reappearing outside ``launch``, where the upstream
    launch-inversion migration would then miss it.
    """
    import re

    import koopmans

    root = Path(koopmans.__file__).parent
    offenders = [
        str(path)
        for path in root.rglob("*.py")
        if path.name != "api.py"
        and re.search(r"\b(?:wg|workgraph)\.(?:submit|run)\(", path.read_text())
    ]
    assert not offenders, offenders
