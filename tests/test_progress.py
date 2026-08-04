"""Row assembly for the live progress table.

The rows are built from fake node trees rather than a live run: what is
under test is the display contract — which processes get a row, in what
order, and which state each row shows — and that contract has to hold
for node trees a two-minute test run would never produce (a ten-way
fan-out, a wrapper caught mid-flight). The AiiDA lookups the assembler
makes on each node (label, type, reload) are stubbed; the state logic it
applies to them is not.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from koopmans.aiida import progress

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiida.orm import ProcessNode

_EPOCH = datetime(2026, 1, 1, 12, 0, 0)
_pk_counter = itertools.count(1)


@dataclass
class FakeNode:
    """A stand-in for a ProcessNode, carrying only what the assembler reads."""

    label: str = ""
    state: str = "waiting"
    kind: str = "workchain"
    seconds: float = 0.0
    children: list[FakeNode] = field(default_factory=list)
    paused: bool = False
    is_pyfunction: bool = False
    finished_ok: bool = True
    process_label: str | None = None
    pk: int = field(default_factory=lambda: next(_pk_counter))

    @property
    def ctime(self) -> datetime:
        """Creation time, spaced out from a fixed epoch by ``seconds``."""
        return _EPOCH + timedelta(seconds=self.seconds)

    @property
    def called(self) -> list[FakeNode]:
        """The child processes, in the order AiiDA happens to return them."""
        return self.children

    @property
    def process_state(self) -> SimpleNamespace:
        """The process state, wrapped the way AiiDA's enum presents it."""
        return SimpleNamespace(value=self.state)

    @property
    def is_finished_ok(self) -> bool:
        """Whether a finished process finished without an error exit status."""
        return self.finished_ok


@pytest.fixture
def render(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeNode], list[progress.ProcessRow]]:
    """Return a function building progress rows from a fake node tree."""
    registry: dict[int, FakeNode] = {}

    def _register(node: FakeNode) -> None:
        registry[node.pk] = node
        for child in node.children:
            _register(child)

    monkeypatch.setattr(progress, "get_node_label", lambda node, include_code=True: node.label)
    monkeypatch.setattr(progress, "get_node_type", lambda node: node.kind)
    monkeypatch.setattr(progress, "_is_process_function_node", lambda node: node.is_pyfunction)
    monkeypatch.setattr(progress, "_reload", lambda pk: registry[pk])

    def _render(root: FakeNode) -> list[progress.ProcessRow]:
        _register(root)
        return progress.build_progress_rows(cast("ProcessNode", root))

    return _render


def _wrapped_calcjob(state: str = "waiting", **kwargs: Any) -> FakeNode:
    """Build root → ``DFT Init (nspin=1)`` → the one kcp.x call it wraps."""
    calcjob = FakeNode(label="kcp-dft_init", kind="calcjob", state=state, **kwargs)
    wrapper = FakeNode(label="dft_init_nspin_1", children=[calcjob])
    return FakeNode(process_label="WorkGraph<KoopmansDSCFWorkflow>", children=[wrapper])


class TestStableRows:
    """A row that has appeared is never withdrawn."""

    def test_the_wrapped_calcjob_never_gets_its_own_row(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """The duplicate is hidden while running, not only once it is done."""
        rows = render(_wrapped_calcjob(state="running"))

        assert [row.label for row in rows] == ["Koopmans DSCF Workflow", "DFT Init (nspin=1)"]

    def test_the_visible_row_advances_with_the_process_it_hides(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """One row, whose status tracks the kcp.x call rather than its wrapper.

        The wrapper ``@task.graph`` sits in ``waiting`` for as long as its
        child runs, so without the hidden row's state the step would look
        stalled for its whole duration.
        """
        states = []
        for calcjob_state in ("created", "running", "finished"):
            root = _wrapped_calcjob(state=calcjob_state)
            rows = render(root)
            assert [row.label for row in rows] == ["Koopmans DSCF Workflow", "DFT Init (nspin=1)"]
            states.append(rows[1].state)

        assert states == ["waiting", "running", "running"]

    def test_the_visible_row_reports_its_own_outcome(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """A hidden process that has finished does not finish its wrapper for it."""
        root = _wrapped_calcjob(state="finished")
        root.children[0].state = "finished"

        rows = render(root)

        assert rows[1].state == "finished"

    def test_a_terminal_row_keeps_its_own_state(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """A failed wrapper stays failed even while the process it hides runs on."""
        root = _wrapped_calcjob(state="running")
        root.children[0].state = "finished"
        root.children[0].finished_ok = False

        rows = render(root)

        assert rows[1].state == "failed"

    def test_the_row_set_only_grows(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """Across a run's snapshots, every label present stays present."""
        calcjob = FakeNode(label="kcp-dft_init", kind="calcjob", state="created")
        wrapper = FakeNode(label="dft_init_nspin_1", state="created", children=[calcjob])
        root = FakeNode(process_label="WorkGraph<KoopmansDSCFWorkflow>", children=[wrapper])

        seen: set[str] = set()
        for wrapper_state, calcjob_state in (
            ("created", "created"),
            ("waiting", "running"),
            ("waiting", "finished"),
            ("finished", "finished"),
        ):
            wrapper.state, calcjob.state = wrapper_state, calcjob_state
            labels = {row.label for row in render(root)}
            assert seen <= labels
            seen = labels

    def test_a_paused_process_shows_through_its_wrapper(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """Hiding a row must not hide a stuck transport task."""
        root = _wrapped_calcjob(state="running", paused=True)

        rows = render(root)

        assert rows[1].state == "paused"

    def test_a_hidden_process_lets_its_children_through(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """Descendants of a hidden row are indented against the visible ancestor."""
        scf = FakeNode(label="pw-scf", kind="calcjob", state="running")
        inner = FakeNode(label="dft_init", children=[scf])
        wrapper = FakeNode(label="dft_init_nspin_1", children=[inner])
        root = FakeNode(process_label="WorkGraph<KoopmansDSCFWorkflow>", children=[wrapper])

        rows = render(root)

        assert [(row.label, row.depth) for row in rows] == [
            ("Koopmans DSCF Workflow", 0),
            ("DFT Init (nspin=1)", 1),
            ("SCF", 2),
        ]

    def test_process_functions_and_their_subtrees_stay_hidden(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """Plumbing tasks add no rows, nor do the processes they call."""
        buried = FakeNode(label="pw-scf", kind="calcjob")
        helper = FakeNode(label="build_iter_source", is_pyfunction=True, children=[buried])
        root = FakeNode(process_label="WorkGraph<KoopmansDSCFWorkflow>", children=[helper])

        assert [row.label for row in render(root)] == ["Koopmans DSCF Workflow"]


class TestSiblingOrder:
    """Siblings read in the order a user counts them."""

    def _fan_out(self, indices: list[int], seconds: list[float]) -> FakeNode:
        """Build a per-orbital fan-out whose creation order is not its index order."""
        children = [
            FakeNode(label=f"compute_alpha_orb_{index}", seconds=second)
            for index, second in zip(indices, seconds, strict=True)
        ]
        return FakeNode(process_label="ComputeScreeningParameters", children=children)

    def test_indices_sort_numerically(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """``Orb 10`` sorts after ``Orb 2``, whatever order the engine created them in."""
        root = self._fan_out([10, 1, 2], [0.0, 0.1, 0.2])

        rows = render(root)

        assert [row.label for row in rows[1:]] == [
            "Compute Alpha Orb 1",
            "Compute Alpha Orb 2",
            "Compute Alpha Orb 10",
        ]

    def test_distinct_steps_keep_execution_order(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """Sequential steps are not alphabetized: the table still reads as the run did."""
        root = FakeNode(
            process_label="WorkGraph<KoopmansDSCFWorkflow>",
            children=[
                FakeNode(label="wannierize", seconds=1.0),
                FakeNode(label="dft_bands", seconds=2.0),
            ],
        )

        rows = render(root)

        assert [row.label for row in rows[1:]] == ["Wannierize", "DFT Bands"]

    def test_an_interleaved_family_keeps_its_creation_order(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """A family split by another step is left exactly as it ran.

        The spin initialization is one such family: the dummy nspin=2
        step lays out the restart files the real nspin=2 step reads, so
        sorting the two nspin rows together would put the reader above
        the writer.
        """
        root = FakeNode(
            process_label="WorkGraph<KoopmansDSCFWorkflow>",
            children=[
                FakeNode(label="dft_init_nspin_1", seconds=1.0),
                FakeNode(label="dft_init_nspin_2_dummy", seconds=2.0),
                FakeNode(label="dft_init_nspin_2", seconds=3.0),
            ],
        )

        rows = render(root)

        assert [row.label for row in rows[1:]] == [
            "DFT Init (nspin=1)",
            "DFT Init (nspin=2; dummy)",
            "DFT Init (nspin=2)",
        ]

    def test_a_contiguous_family_sorts_amid_unmoved_rows(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """The fan-out counts properly; the rows around it do not move."""
        root = FakeNode(
            process_label="WorkGraph<KoopmansDSCFWorkflow>",
            children=[
                FakeNode(label="ki_trial", seconds=1.0),
                FakeNode(label="compute_alpha_orb_1", seconds=2.0),
                FakeNode(label="compute_alpha_orb_10", seconds=3.0),
                FakeNode(label="compute_alpha_orb_2", seconds=4.0),
                FakeNode(label="final_scf", seconds=5.0),
            ],
        )

        rows = render(root)

        assert [row.label for row in rows[1:]] == [
            "KI Trial",
            "Compute Alpha Orb 1",
            "Compute Alpha Orb 2",
            "Compute Alpha Orb 10",
            "Final SCF",
        ]

    def test_simultaneous_namesakes_break_by_pk(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """Two identical labels created in the same instant still get a fixed order."""
        later = FakeNode(label="pw-scf", kind="calcjob", pk=99, seconds=1.0)
        earlier = FakeNode(label="pw-scf", kind="calcjob", pk=7, seconds=1.0)
        root = FakeNode(process_label="WorkGraph<Wannierize>", children=[later, earlier])

        rows = render(root)

        assert [row.depth for row in rows] == [0, 1, 1]
        assert progress._ordered_children(cast("ProcessNode", root))[0][1].pk == 7

    def test_the_order_applies_at_every_level(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """A fan-out nested under a step is ordered like a top-level one."""
        fan_out = self._fan_out([2, 10, 1], [0.0, 0.1, 0.2])
        fan_out.label = "screening_iteration"
        root = FakeNode(process_label="WorkGraph<KoopmansDSCFWorkflow>", children=[fan_out])

        rows = render(root)

        assert [(row.label, row.depth) for row in rows] == [
            ("Koopmans DSCF Workflow", 0),
            ("Iteration 1", 1),
            ("Compute Alpha Orb 1", 2),
            ("Compute Alpha Orb 2", 2),
            ("Compute Alpha Orb 10", 2),
        ]


class TestPrettifyLabel:
    """Label rewriting the ordering rules depend on."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("orb_1_filled_orbital_screening", "Orbital 1 (filled)"),
            ("orb_10_empty_orbital_screening", "Orbital 10 (empty)"),
            ("screening_iteration", "Iteration 1"),
            ("screening_iteration_1", "Iteration 2"),
            ("dft_init_nspin_1", "DFT Init (nspin=1)"),
            ("dft_init_nspin_2_dummy", "DFT Init (nspin=2; dummy)"),
            ("kcp-ki_n_minus_1", "KI N-1"),
        ],
    )
    def test_labels_collapse_to_their_display_form(self, raw: str, expected: str) -> None:
        """The Map-zone key and the physics-paper conventions survive."""
        assert progress.prettify_label(raw) == expected
