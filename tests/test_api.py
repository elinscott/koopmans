"""The public python API: build/run/submit and the outputs dict.

No calculation runs here: ``build`` is checked against the dispatcher it
wraps, the two launching verbs are checked to funnel through the single
internal launch helper, and the outputs dict is read off a hand-built
finished node shaped like a real ``KoopmansDSCFWorkflow`` one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import koopmans
from koopmans import KoopmansInput, outputs, read_input_file
from koopmans.api import launch


class TestPublicSurface:
    """The package's top level exports the documented names."""

    def test_all_names_importable(self) -> None:
        """Every ``__all__`` name resolves on the package."""
        for name in koopmans.__all__:
            assert getattr(koopmans, name) is not None
        assert set(koopmans.__all__) >= {"build", "run", "submit", "outputs", "KoopmansInput"}


class TestBuild:
    """``build`` materializes the same graph as the dispatcher."""

    def test_build_matches_build_workgraph(
        self,
        aiida_profile: Any,
        installed_dscf_codes: Any,
        fake_sg15_pseudo_family: Any,
        tutorials_dir: Path,
    ) -> None:
        """The ozone input builds the same task set through both entry points.

        Also pins that ``build`` leaves the already-loaded (test) profile
        alone rather than switching to the installed koopmans one.
        """
        from koopmans.aiida.workflows import build_workgraph

        inp = read_input_file(tutorials_dir / "orbital_energies/ozone/ozone.yaml")
        via_api = koopmans.build(inp)
        via_dispatcher = build_workgraph(inp)
        assert via_api.get_task_names() == via_dispatcher.get_task_names()


class TestLaunchFunnel:
    """Both launching verbs go through the one internal launch helper."""

    def test_run_and_submit_route_through_launch(
        self, aiida_profile: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``run`` launches blocking and returns the outputs; ``submit`` the id."""
        calls: list[tuple[Any, bool, bool]] = []
        node = _finished_dscf_node()

        def fake_launch(workgraph: Any, *, blocking: bool, wait: bool = False) -> Any:
            """Record the launch request and hand back the finished node."""
            calls.append((workgraph, blocking, wait))
            return node

        sentinel = object()
        monkeypatch.setattr("koopmans.api.launch", fake_launch)
        monkeypatch.setattr("koopmans.api.build", lambda inp: sentinel)

        results = koopmans.run("unused")  # type: ignore[arg-type]
        assert results["parameters"]["energy"] == pytest.approx(-1296.39)
        assert koopmans.submit("unused", wait=True) == node.pk  # type: ignore[arg-type]
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
        node = launch(blocking_wg, blocking=True)
        assert blocking_wg.calls == [("run", None)]
        assert node == "the-node"
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
        for label in ("remote_folder", "ki_final__remote_folder"):
            remote = orm.RemoteData(remote_path="/scratch/run", computer=remote_computer)
            remote.store()
            remote.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label=label)
    if with_calcjob:
        # Two steps, each with the retrieved folder a finished CalcJob
        # always has: the dump reads the input files off the calculation
        # itself and the output files off ``retrieved``.
        for label in ("dft_init", "ki_final"):
            calc = orm.CalcJobNode()
            calc.set_process_state(ProcessState.FINISHED)
            calc.set_exit_status(0)
            # Distinct content per step: two calculations that wrote the
            # same bytes would be one calculation dumped twice.
            calc.base.repository.put_object_from_bytes(f"input for {label}\n".encode(), "aiida.cpi")
            calc.base.links.add_incoming(node, link_type=LinkType.CALL_CALC, link_label=label)
            calc.store()
            retrieved = orm.FolderData()
            retrieved.base.repository.put_object_from_bytes(
                f"output of {label}\n".encode(), "aiida.cpo"
            )
            retrieved.base.links.add_incoming(
                calc, link_type=LinkType.CREATE, link_label="retrieved"
            )
            retrieved.store()
            calc.seal()
    if sealed:
        node.seal()
    return node


class TestOutputs:
    """``outputs`` reads a finished calculation back as a plain dict."""

    def test_deserializes_every_socket(self, aiida_profile: Any, localhost_computer: Any) -> None:
        """Values come back plain and nested; file handles are omitted."""
        node = _finished_dscf_node(remote_computer=localhost_computer)
        results = outputs(node.pk)
        assert results["parameters"]["energy"] == pytest.approx(-1296.39)
        assert -results["parameters"]["homo_energy"] == pytest.approx(12.52)
        assert results["eigenvalues"].shape == (2, 2)
        assert results["alphas"] == {"filled": {"none": [0.66, 0.73]}, "empty": {"none": [0.72]}}
        assert "remote_folder" not in results
        assert "ki_final" not in results  # a namespace holding only a file handle

    def test_running_calculation_refuses_to_read(self, aiida_profile: Any) -> None:
        """A still-running calculation raises rather than returning stale data."""
        node = _finished_dscf_node(sealed=False)
        with pytest.raises(RuntimeError, match="still running"):
            outputs(node.pk)

    def test_failed_calculation_refuses_to_read(self, aiida_profile: Any) -> None:
        """A failed calculation raises, naming its exit status."""
        node = _finished_dscf_node(exit_status=302, with_outputs=False)
        with pytest.raises(RuntimeError, match=r"failed \(exit status 302\)"):
            outputs(node.pk)

    def test_dump_writes_the_per_step_layout(self, aiida_profile: Any, tmp_path: Path) -> None:
        """The dump path the docs point at writes one folder per step."""
        from koopmans.aiida.dumping import dump_workgraph

        out = dump_workgraph(_finished_dscf_node(with_calcjob=True), tmp_path / "ozone")
        assert out == tmp_path / "ozone"
        step_dirs = [p.name for p in out.rglob("*") if p.is_dir()]
        assert any("ki_final" in name for name in step_dirs), step_dirs


class TestInputConstructibleInPython:
    """A ``KoopmansInput`` built in python matches the file-parsed one."""

    def test_model_validate_equals_file_parse(self, tutorials_dir: Path) -> None:
        """The documented in-python construction path produces the same model."""
        from yaml import safe_load

        ozone = tutorials_dir / "orbital_energies/ozone/ozone.yaml"
        with open(ozone) as handle:
            raw = safe_load(handle)
        assert KoopmansInput.model_validate(raw) == read_input_file(ozone)


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
