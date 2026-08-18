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
        assert failures[0].name == "SCF"
        # A workchain runs whatever its calculations run, and is named
        # next to nothing; the calculation under it carries the binary.
        assert failures[0].code is None
        assert failures[0].exit_status == 402
        assert failures[0].message == "pw.x did not converge"
        assert failures[0].state == "failed"

    def test_the_root_itself_can_be_the_failure(self, aiida_profile_clean: Any) -> None:
        """A failure with no failed descendant still gets reported once."""
        root = make_process(
            process_label="WorkGraph<Tiny>", exit_status=1, exit_message="graph failed"
        )

        failures = progress._walk_failed_descendants(root)

        assert [(f.name, f.exit_status, f.state) for f in failures] == [("Tiny", 1, "failed")]
        # The root is the run, not a step in it, so it stands under nothing.
        assert failures[0].path == ()

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

        assert [f.name for f in failures] == ["Tiny", "SCF"]


def _two_failed_iterations() -> Any:
    """Build a screening run whose first two iterations both failed.

    The second iteration is reached through ``refine_screening_parameters``,
    the recursion that names every iteration after the first the same way,
    and which the table sees through.
    """
    root = make_process(process_label="WorkGraph<KoopmansDSCFWorkflow>", exit_status=500)
    screening = make_process(
        caller=root,
        link_label="ComputeScreeningParameters",
        process_label="WorkGraph<ComputeScreeningParameters>",
        exit_status=500,
    )
    make_process(
        caller=screening,
        link_label="ScreeningIteration",
        process_label="WorkGraph<ScreeningIteration>",
        exit_status=305,
        exit_message="alpha did not converge",
    )
    refine = make_process(
        caller=screening,
        link_label="refine_screening_parameters",
        process_label="WorkGraph<RefineScreeningParameters>",
        exit_status=500,
    )
    make_process(
        caller=refine,
        link_label="screening_iteration",
        process_label="WorkGraph<ScreeningIteration>",
        exit_status=305,
        exit_message="alpha did not converge",
    )
    return root


class TestTwoRunsOfTheSameStep:
    """A step that runs more than once is told apart by where it sits."""

    def test_two_failed_iterations_are_distinguishable(self, aiida_profile_clean: Any) -> None:
        """Same label, same exit status, same message — only position differs.

        Without the index these two lines are identical, and a reader
        cannot tell which pass of the screening loop failed.
        """
        failures = progress._walk_failed_descendants(_two_failed_iterations())

        iterations = [f for f in failures if f.name.startswith("Iteration")]
        assert [(f.name, f.path) for f in iterations] == [
            ("Iteration 1", ("Screening parameters",)),
            ("Iteration 2", ("Screening parameters",)),
        ]

    def test_the_two_lines_printed_are_not_the_same_line(self, aiida_profile_clean: Any) -> None:
        """What the reader sees, rather than what the walk returns."""
        console, buffer = _capturing_console()

        progress.render_process_once(_two_failed_iterations(), console=console)

        printed = [
            line.strip()
            for line in buffer.getvalue().splitlines()
            if "Iteration" in line and "—" in line
        ]
        assert printed == [
            "Iteration 1 in Screening parameters — exit status 305: alpha did not converge",
            "Iteration 2 in Screening parameters — exit status 305: alpha did not converge",
        ]

    def test_no_pk_reaches_the_summary(self, aiida_profile_clean: Any) -> None:
        """A pk means nothing to a reader who does not know the engine."""
        console, buffer = _capturing_console()

        progress.render_process_once(_two_failed_iterations(), console=console)

        output = buffer.getvalue()
        assert "pk" not in output
        assert "(pk " not in output


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
        self, aiida_profile_clean: Any, localhost_computer: Any
    ) -> None:
        """A failed step's detail is printed alongside the state table.

        The binary comes off the code the calculation ran, so the step
        that names one here is the pw.x call rather than the workchain
        around it.
        """
        from aiida.orm import InstalledCode

        code = InstalledCode(
            label="pw",
            computer=localhost_computer,
            default_calc_job_plugin="quantumespresso.pw",
            filepath_executable="/opt/qe/bin/pw.x",
        ).store()
        root = make_process(process_label="WorkGraph<Tiny>", exit_status=1)
        scf = make_process(
            caller=root,
            link_label="scf",
            process_label="PwBaseWorkChain",
            exit_status=402,
            exit_message="pw.x did not converge",
        )
        make_process(
            caller=scf,
            link_label="iteration_01",
            process_label="PwCalculation",
            calcjob=True,
            computer=localhost_computer,
            inputs={"code": code},
            exit_status=402,
            exit_message="pw.x did not converge",
        )
        console, buffer = _capturing_console()

        progress.render_process_once(root, console=console)

        output = buffer.getvalue()
        # The failure line names the step and the binary that ran it,
        # never its Python class.
        assert "SCF (pw.x)" in output
        assert "Pw Base Work Chain" not in output
        assert "402" in output
        assert "pw.x did not converge" in output
        assert "finished with status: 1" in output

    def test_a_killed_step_says_killed_not_excepted(self, aiida_profile_clean: Any) -> None:
        """A killed descendant has no exit status, but it was not excepted either.

        Both a killed and an excepted process have ``exit_status is None``;
        the per-step detail line must tell them apart by process state,
        not treat "no exit status" as synonymous with "excepted".
        """
        from aiida import orm
        from aiida.common.links import LinkType
        from plumpy.process_states import ProcessState

        root = orm.WorkflowNode()
        root.store()
        root.set_process_label("WorkGraph<Tiny>")

        child = orm.WorkflowNode()
        child.base.links.add_incoming(root, link_type=LinkType.CALL_WORK, link_label="scf")
        child.store()
        child.set_process_label("PwBaseWorkChain")
        child.set_process_state(ProcessState.KILLED)

        root.set_process_state(ProcessState.KILLED)
        console, buffer = _capturing_console()

        progress.render_process_once(root, console=console)

        output = buffer.getvalue()
        assert "Workflow was killed!" in output
        # The step's own detail line says killed, above the closing banner.
        killed_line_index = output.index("SCF —")
        banner_index = output.index("Workflow was killed!")
        assert killed_line_index < banner_index
        detail_line = output[killed_line_index:banner_index]
        assert "killed" in detail_line
        assert "excepted" not in detail_line

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
