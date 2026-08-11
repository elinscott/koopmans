"""One-shot and attach-style rendering: ``render_process_once`` and ``watch_process``.

``tests/test_progress.py`` covers row assembly against stubbed node trees;
what is under test here is the layer built on top of it — the failure
summary and the outcome banner — against real ``ProcessNode`` trees, built
either directly (:func:`tests.fixtures.make_process`, for exact control
over exit status and hierarchy — the pattern ``test_plotting.py`` uses for
finished runs) or by actually executing a trivial ``WorkGraph`` in-process
(``wg.run()``, no daemon involved).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

from rich.console import Console

from koopmans.aiida import progress

if TYPE_CHECKING:
    from aiida import orm

from tests.fixtures import make_process


def _capturing_console() -> tuple[Console, io.StringIO]:
    """Return a console that renders to a plain-text buffer, no ANSI codes."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=120, no_color=True, force_terminal=False)
    return console, buffer


class TestWalkFailedDescendants:
    """Every terminated-not-ok process in the tree, root included."""

    def test_a_finished_ok_tree_reports_nothing(self, aiida_profile_clean: Any) -> None:
        """A run with no failure anywhere in it has nothing to report."""
        root = make_process(process_label="WorkGraph<Tiny>")
        make_process(caller=root, link_label="step", exit_status=0)

        assert progress._walk_failed_descendants(root) == []

    def test_a_failed_child_is_reported_with_its_exit_status_and_message(
        self, aiida_profile_clean: Any
    ) -> None:
        """A failed step's exit status and message both come through."""
        root = make_process(process_label="WorkGraph<Tiny>")
        make_process(
            caller=root,
            link_label="scf",
            process_label="PwBaseWorkChain",
            exit_status=402,
            exit_message="pw.x did not converge",
        )

        failures = progress._walk_failed_descendants(root)

        assert len(failures) == 1
        _pk, label, exit_status, message = failures[0]
        assert label == "PwBaseWorkChain"
        assert exit_status == 402
        assert message == "pw.x did not converge"

    def test_the_root_itself_can_be_the_failure(self, aiida_profile_clean: Any) -> None:
        """A failure with no failed descendant still gets reported once."""
        root = make_process(
            process_label="WorkGraph<Tiny>", exit_status=1, exit_message="graph failed"
        )

        failures = progress._walk_failed_descendants(root)

        assert [f[1:3] for f in failures] == [("WorkGraph<Tiny>", 1)]

    def test_a_cascading_failure_reports_both_wrapper_and_cause(
        self, aiida_profile_clean: Any
    ) -> None:
        """The wrapper and the step that actually failed both surface.

        The failed leaf carries the informative exit message; a caller
        wanting only the deepest cause still gets it, alongside the
        wrapper's own (often generic) propagated status.
        """
        root = make_process(process_label="WorkGraph<Tiny>", exit_status=500)
        make_process(
            caller=root,
            link_label="scf",
            process_label="PwBaseWorkChain",
            exit_status=402,
            exit_message="pw.x did not converge",
        )

        failures = progress._walk_failed_descendants(root)

        assert [label for _, label, _, _ in failures] == ["WorkGraph<Tiny>", "PwBaseWorkChain"]


class TestRenderProcessOnce:
    """The one-shot render ``koopmans status`` and a terminated ``attach`` use."""

    def test_a_successful_run_prints_the_table_and_the_success_banner(
        self, aiida_profile_clean: Any
    ) -> None:
        """A finished-ok process gets the state table and the success line."""
        root = make_process(process_label="WorkGraph<Tiny>")
        console, buffer = _capturing_console()

        progress.render_process_once(root, console=console)

        output = buffer.getvalue()
        assert "Tiny" in output
        assert "Workflow completed successfully!" in output

    def test_a_failed_step_prints_its_exit_status_and_message(
        self, aiida_profile_clean: Any
    ) -> None:
        """A failed step's detail is printed alongside the state table."""
        root = make_process(process_label="WorkGraph<Tiny>", exit_status=1)
        make_process(
            caller=root,
            link_label="scf",
            process_label="PwBaseWorkChain",
            exit_status=402,
            exit_message="pw.x did not converge",
        )
        console, buffer = _capturing_console()

        progress.render_process_once(root, console=console)

        output = buffer.getvalue()
        # The failure line prettifies the label the same way row labels are.
        assert "Pw Base Work Chain" in output
        assert "402" in output
        assert "pw.x did not converge" in output
        assert "finished with status: 1" in output

    def test_a_still_running_process_gets_no_outcome_banner(self, aiida_profile_clean: Any) -> None:
        """A process that has not terminated has no outcome to report yet."""
        from plumpy.process_states import ProcessState

        root = make_process(process_label="WorkGraph<Tiny>")
        root.set_process_state(ProcessState.RUNNING)
        console, buffer = _capturing_console()

        progress.render_process_once(root, console=console)

        output = buffer.getvalue()
        assert "Workflow completed successfully!" not in output
        assert "Workflow excepted!" not in output


class TestWatchProcessOnAnAlreadyTerminatedProcess:
    """``watch_process`` given a terminated node returns immediately.

    This is the ``koopmans attach`` degrade path exercised without a
    daemon: the polling loop itself (attaching to a process that is
    still running when ``attach`` starts) needs a live daemon and is not
    covered here — see the PR description's testing notes.
    """

    def test_it_returns_the_same_node_and_prints_the_banner(self, aiida_profile_clean: Any) -> None:
        """No polling happens; the terminated node is rendered and handed back."""
        root = make_process(process_label="WorkGraph<Tiny>")
        console, buffer = _capturing_console()

        result = progress.watch_process(root, console=console)

        assert result.pk == root.pk
        assert "Workflow completed successfully!" in buffer.getvalue()


class TestRenderAgainstARealExecutedWorkgraph:
    """Sanity check against a genuinely-run WorkGraph, not a hand-built tree."""

    def test_a_real_finished_run_renders_without_error(self, aiida_profile_clean: Any) -> None:
        """A process that actually ran through the engine renders the same way."""
        from aiida_workgraph import WorkGraph, task

        @task  # type: ignore[untyped-decorator]
        def add_one(x: int) -> int:
            """Return ``x + 1``, the whole content of this trivial run."""
            return x + 1

        wg = WorkGraph("tiny_for_progress")
        wg.add_task(add_one, name="add_one", x=1)
        wg.run()
        process_node: orm.ProcessNode = wg.process

        assert process_node.is_finished_ok, process_node.exception

        console, buffer = _capturing_console()
        progress.render_process_once(process_node, console=console)

        output = buffer.getvalue()
        assert "Workflow completed successfully!" in output
