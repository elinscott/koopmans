"""Row assembly for the live progress table.

The rows are built from fake node trees rather than a live run: what is
under test is the display contract — which processes get a row, what each
row is called, in what order, and which state it shows — and that contract
has to hold for node trees a two-minute test run would never produce (a
ten-way fan-out, a wrapper caught mid-flight, a restarting SCF). The AiiDA
lookups the assembler makes on each node (label, type, reload) are stubbed;
the logic it applies to them is not.

:data:`ROUTES` reconstructs the process tree of every route the CLI can
run, and ``tests/data/progress_tables.txt`` records what each renders as.
Regenerate that file after an intended change with::

    python -c "import tests.test_progress as t; \
        t.TABLES_FILE.write_text(t.render_route_tables(), encoding='utf-8')"
"""

from __future__ import annotations

import itertools
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from koopmans.aiida import progress

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from aiida.orm import ProcessNode

_EPOCH = datetime(2026, 1, 1, 12, 0, 0)
_pk_counter = itertools.count(1)

TABLES_FILE = Path(__file__).parent / "data" / "progress_tables.txt"


@dataclass
class FakeNode:
    """A stand-in for a ProcessNode, carrying only what the assembler reads.

    ``link`` is the call link label provenance reads; ``label`` is the
    name ``aiida-koopmans`` puts on the process node, which is what a
    reader is shown. A node left without one stands for a process from a
    run made before the plugin named its steps.
    """

    link: str = ""
    label: str = ""
    executable: str | None = None
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
    def inputs(self) -> SimpleNamespace:
        """The process's inputs, of which only ``code`` is ever read."""
        if self.executable is None:
            return SimpleNamespace()
        return SimpleNamespace(
            code=SimpleNamespace(filepath_executable=f"/usr/local/bin/{self.executable}")
        )

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


def _fake_node_label(node: FakeNode, include_code: bool = True) -> str:
    """Stand in for :func:`~koopmans.aiida.utils.get_node_label`.

    The real one prefixes a CalcJob's link label with its code's label
    when asked; the assembler never asks, and this raises rather than
    quietly return the wrong string if it starts to.
    """
    if include_code:
        raise AssertionError("the display reads link labels without a code prefix")
    return node.link


@contextmanager
def stubbed_lookups(registry: dict[int, FakeNode]) -> Iterator[None]:
    """Point the assembler's three AiiDA lookups at a fake node tree."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(progress, "get_node_label", _fake_node_label)
        patch.setattr(progress, "get_node_type", lambda node: node.kind)
        patch.setattr(progress, "_is_process_function_node", lambda node: node.is_pyfunction)
        patch.setattr(progress, "_reload", lambda pk: registry[pk])
        yield


def _register(node: FakeNode, registry: dict[int, FakeNode]) -> None:
    """Record ``node`` and its descendants so ``_reload`` can find them by pk."""
    registry[node.pk] = node
    for child in node.children:
        _register(child, registry)


@pytest.fixture
def render() -> Iterator[Callable[[FakeNode], list[progress.ProcessRow]]]:
    """Return a function building progress rows from a fake node tree."""
    registry: dict[int, FakeNode] = {}

    with stubbed_lookups(registry):

        def _render(root: FakeNode) -> list[progress.ProcessRow]:
            _register(root, registry)
            return progress.build_progress_rows(cast("ProcessNode", root))

        yield _render


#: What the ``describe`` fixture hands back: one fake node, described.
type _Describe = Callable[..., progress.LabelDisplay]


@pytest.fixture
def describe() -> Iterator[Callable[..., progress.LabelDisplay]]:
    """Return a function describing one fake node the way a row would."""
    with stubbed_lookups({}):

        def _describe(node: FakeNode, is_root: bool = False) -> progress.LabelDisplay:
            return progress.describe_process(cast("ProcessNode", node), is_root=is_root)

        yield _describe


@pytest.fixture
def step_paths() -> Iterator[Callable[[FakeNode], dict[int, tuple[str, ...]]]]:
    """Return a function mapping each process of a fake tree to its step path."""
    registry: dict[int, FakeNode] = {}

    with stubbed_lookups(registry):

        def _paths(root: FakeNode) -> dict[int, tuple[str, ...]]:
            _register(root, registry)
            return progress.build_step_paths(cast("ProcessNode", root))

        yield _paths


def _wrapped_calcjob(state: str = "waiting", **kwargs: Any) -> FakeNode:
    """Build root → ``DFT initialization (nspin=1)`` → the one kcp.x call it wraps."""
    calcjob = FakeNode(
        link="dft_init",
        label="DFT initialization",
        kind="calcjob",
        executable="kcp.x",
        state=state,
        **kwargs,
    )
    wrapper = FakeNode(
        link="dft_init_nspin1", label="DFT initialization (nspin=1)", children=[calcjob]
    )
    return FakeNode(
        process_label="WorkGraph<KoopmansDSCFWorkflow>", label="Koopmans ΔSCF", children=[wrapper]
    )


class TestStableRows:
    """A row that has appeared is never withdrawn."""

    def test_the_wrapped_calcjob_never_gets_its_own_row(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """The duplicate is hidden while running, not only once it is done."""
        rows = render(_wrapped_calcjob(state="running"))

        assert [row.label for row in rows] == [
            "Koopmans ΔSCF",
            "DFT initialization (nspin=1)",
        ]

    def test_the_collapsed_row_shows_the_code_of_the_call_it_stands_for(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """A container that is one calculation reports that calculation's binary."""
        rows = render(_wrapped_calcjob(state="running"))

        assert [row.code for row in rows] == [None, "kcp.x"]

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
            assert [row.label for row in rows] == [
                "Koopmans ΔSCF",
                "DFT initialization (nspin=1)",
            ]
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
        calcjob = FakeNode(link="kcp-dft_init", kind="calcjob", state="created")
        wrapper = FakeNode(
            link="dft_init_nspin1",
            label="DFT initialization (nspin=1)",
            state="created",
            children=[calcjob],
        )
        root = FakeNode(
            process_label="WorkGraph<KoopmansDSCFWorkflow>",
            label="Koopmans ΔSCF",
            children=[wrapper],
        )

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

    def test_process_functions_and_their_subtrees_stay_hidden(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """Plumbing tasks add no rows, nor do the processes they call."""
        buried = FakeNode(link="scf", label="SCF", kind="calcjob", executable="pw.x")
        helper = FakeNode(link="build_iter_source", is_pyfunction=True, children=[buried])
        root = FakeNode(
            process_label="WorkGraph<KoopmansDSCFWorkflow>",
            label="Koopmans ΔSCF",
            children=[helper],
        )

        assert [row.label for row in render(root)] == ["Koopmans ΔSCF"]


class TestCollapsingAndTransparency:
    """Which containers keep a row, decided by what is inside them."""

    def _scf_nscf(self, scf_attempts: int = 1) -> FakeNode:
        """Build the ``scf_nscf`` sub-graph, with a restartable SCF beneath it."""
        scf = FakeNode(
            link="scf",
            label="SCF",
            children=[
                FakeNode(link="scf", label="SCF", kind="calcjob", executable="pw.x", seconds=index)
                for index in range(scf_attempts)
            ],
        )
        nscf = FakeNode(
            link="nscf",
            label="NSCF",
            seconds=10.0,
            children=[FakeNode(link="nscf", label="NSCF", kind="calcjob", executable="pw.x")],
        )
        ground_state = FakeNode(link="scf_nscf", label="Ground state", children=[scf, nscf])
        return FakeNode(
            process_label="WorkGraph<SinglepointDFPTWorkflow>",
            label="Koopmans DFPT",
            children=[ground_state],
        )

    def test_a_step_is_not_deleted_by_what_its_parent_is_called(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """``scf`` survives under ``scf_nscf``, which shares its opening word.

        Suppressing a row whose parent's label starts with it deleted the
        SCF from every route that groups an scf and an nscf together —
        along with the pw.x call under it, which carries the same label.
        """
        rows = render(self._scf_nscf())

        assert [(row.label, row.depth, row.code) for row in rows] == [
            ("Koopmans DFPT", 0, None),
            ("Ground state", 1, None),
            ("SCF", 2, "pw.x"),
            ("NSCF", 2, "pw.x"),
        ]

    def test_a_restarting_workchain_keeps_its_attempts_visible(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """Two calculations under one step are two rows, not a silent collapse.

        The collapse rule folds a container into the single calculation it
        contains. A ``PwBaseWorkChain`` that has restarted contains two,
        and that is exactly when a watcher needs to see them.
        """
        rows = render(self._scf_nscf(scf_attempts=2))

        assert [(row.label, row.depth, row.code) for row in rows] == [
            ("Koopmans DFPT", 0, None),
            ("Ground state", 1, None),
            ("SCF", 2, None),
            ("SCF", 3, "pw.x"),
            ("SCF", 3, "pw.x"),
            ("NSCF", 2, "pw.x"),
        ]

    def test_a_transparent_wrapper_lifts_its_children_to_its_own_depth(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """``PwBandsWorkChain`` adds nothing the root does not already say."""
        wrapper = FakeNode(
            link="PwBandsWorkChain",
            label="DFT band structure",
            children=[
                FakeNode(
                    link="scf",
                    label="SCF",
                    children=[FakeNode(link="scf", label="SCF", kind="calcjob", executable="pw.x")],
                ),
                FakeNode(
                    link="bands",
                    label="Band structure",
                    seconds=1.0,
                    children=[
                        FakeNode(
                            link="bands", label="Band structure", kind="calcjob", executable="pw.x"
                        )
                    ],
                ),
            ],
        )
        root = FakeNode(
            process_label="WorkGraph<RunPwBands>", label="DFT band structure", children=[wrapper]
        )

        rows = render(root)

        assert [(row.label, row.depth) for row in rows] == [
            ("DFT band structure", 0),
            ("SCF", 1),
            ("Band structure", 1),
        ]

    def test_a_transparent_wrapper_still_lends_its_state(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """A paused wrapper with no row of its own still reaches the display."""
        wrapper = FakeNode(
            link="PwBandsWorkChain",
            label="DFT band structure",
            paused=True,
            children=[FakeNode(link="scf", label="SCF", state="running")],
        )
        root = FakeNode(
            process_label="WorkGraph<RunPwBands>", label="DFT band structure", children=[wrapper]
        )

        assert render(root)[0].state == "paused"

    def test_the_wannier90_workchain_is_transparent_but_its_minimization_is_not(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """One link label, two meanings, told apart by the process behind it.

        Both rows read as their link labels: ``Wannier90WorkChain``
        submits its -pp and minimization runs with a metadata dict of its
        own, so no label given from outside reaches either.
        """
        workchain = FakeNode(
            link="wannier90",
            process_label="Wannier90WorkChain",
            children=[
                FakeNode(
                    link="wannier90_pp",
                    process_label="Wannier90BaseWorkChain",
                    children=[
                        FakeNode(link="wannier90_pp", kind="calcjob", executable="wannier90.x")
                    ],
                ),
                FakeNode(
                    link="wannier90",
                    seconds=1.0,
                    process_label="Wannier90BaseWorkChain",
                    children=[FakeNode(link="wannier90", kind="calcjob", executable="wannier90.x")],
                ),
            ],
        )
        root = FakeNode(
            process_label="WorkGraph<Wannierize>", label="Wannierization", children=[workchain]
        )

        rows = render(root)

        assert [(row.label, row.depth, row.code) for row in rows] == [
            ("Wannierization", 0, None),
            ("wannier90_pp", 1, "wannier90.x"),
            ("wannier90", 1, "wannier90.x"),
        ]


class TestScreeningIterationNumbering:
    """Iterations are counted by position, however deeply the recursion nests."""

    def _iteration(self, label: str, seconds: float) -> FakeNode:
        """Build one screening iteration, with a trial KI inside it."""
        return FakeNode(
            link=label,
            label="Iteration",
            seconds=seconds,
            children=[
                FakeNode(link="ki_trial", label="Trial KI", kind="calcjob", executable="kcp.x"),
                FakeNode(
                    link="compute_orbital_screening_parameters",
                    label="Orbital screening",
                    seconds=seconds + 0.1,
                ),
            ],
        )

    def test_every_iteration_gets_its_own_number_at_one_depth(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """The refinement recursion is transparent, so its iterations are siblings.

        ``refine_screening_parameters`` calls itself once per refinement,
        each nesting the next iteration one level deeper and naming it the
        same. Left visible, the table repeated ``Iteration 1`` three times
        at three different indents.
        """
        innermost = FakeNode(
            link="refine_screening_parameters",
            seconds=3.0,
            children=[self._iteration("screening_iteration", 3.1)],
        )
        refine = FakeNode(
            link="refine_screening_parameters",
            seconds=2.0,
            children=[self._iteration("screening_iteration", 2.1), innermost],
        )
        compute = FakeNode(
            link="ComputeScreeningParameters",
            label="Screening parameters",
            children=[self._iteration("ScreeningIteration", 1.0), refine],
        )
        root = FakeNode(
            process_label="WorkGraph<KoopmansDSCFWorkflow>",
            label="Koopmans ΔSCF",
            children=[compute],
        )

        rows = render(root)

        assert [(row.label, row.depth) for row in rows if row.label.startswith("Iteration")] == [
            ("Iteration 1", 2),
            ("Iteration 2", 2),
            ("Iteration 3", 2),
        ]

    def test_a_single_iteration_is_still_numbered(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """A converged-first-time run reads ``Iteration 1``, not a bare ``Iteration``."""
        compute = FakeNode(
            link="ComputeScreeningParameters",
            label="Screening parameters",
            children=[self._iteration("ScreeningIteration", 1.0)],
        )
        root = FakeNode(
            process_label="WorkGraph<KoopmansDSCFWorkflow>",
            label="Koopmans ΔSCF",
            children=[compute],
        )

        assert [row.label for row in render(root)][1:3] == ["Screening parameters", "Iteration 1"]

    def _lone_iteration(self) -> tuple[FakeNode, FakeNode, FakeNode, FakeNode]:
        """Build a screening whose one iteration holds a single calculation.

        The iteration keeps a row of its own and so does the calculation
        beneath it: an iteration is a pass of the loop, and collapsing its
        one calculation into it would erase that distinction.
        """
        calcjob = FakeNode(link="ki_trial", label="Trial KI", kind="calcjob", executable="kcp.x")
        iteration = FakeNode(link="ScreeningIteration", label="Iteration", children=[calcjob])
        compute = FakeNode(
            link="ComputeScreeningParameters", label="Screening parameters", children=[iteration]
        )
        root = FakeNode(
            process_label="WorkGraph<KoopmansDSCFWorkflow>",
            label="Koopmans ΔSCF",
            children=[compute],
        )
        return root, compute, iteration, calcjob

    def test_a_numbered_row_is_not_collapsed_away(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """A number is a position among siblings, and no other row records it.

        Collapsing the row that carries one deletes the only statement of
        which pass of the loop this was — and neither does the calculation
        it ran collapse into it: the reader wants to see ``Trial KI``
        running beneath ``Iteration 1``, not folded into it.
        """
        root, _, _, _ = self._lone_iteration()

        rows = render(root)

        assert [(row.label, row.depth, row.code) for row in rows] == [
            ("Koopmans ΔSCF", 0, None),
            ("Screening parameters", 1, None),
            ("Iteration 1", 2, None),
            ("Trial KI", 3, "kcp.x"),
        ]

    def test_the_numbered_row_places_its_failures_under_itself(
        self, step_paths: Callable[[FakeNode], dict[int, tuple[str, ...]]]
    ) -> None:
        """A failure in that pass says which pass, rather than naming the loop."""
        root, compute, iteration, calcjob = self._lone_iteration()

        paths = step_paths(root)

        assert paths[compute.pk] == ("Screening parameters",)
        assert paths[iteration.pk] == ("Screening parameters", "Iteration 1")
        assert paths[calcjob.pk] == ("Screening parameters", "Iteration 1", "Trial KI")


class TestSiblingOrder:
    """Siblings read in the order a user counts them."""

    def _fan_out(self, indices: list[int], seconds: list[float]) -> FakeNode:
        """Build a per-orbital fan-out whose creation order is not its index order."""
        children = [
            FakeNode(link=f"compute_alpha_orb_{index}", label=f"Orbital {index}", seconds=second)
            for index, second in zip(indices, seconds, strict=True)
        ]
        return FakeNode(
            process_label="ComputeScreeningParameters",
            label="Screening parameters",
            children=children,
        )

    def test_indices_sort_numerically(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """``Orbital 10`` sorts after ``Orbital 2``, whatever order they were created in."""
        root = self._fan_out([10, 1, 2], [0.0, 0.1, 0.2])

        rows = render(root)

        assert [row.label for row in rows[1:]] == ["Orbital 1", "Orbital 2", "Orbital 10"]

    def test_distinct_steps_keep_execution_order(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """Sequential steps are not alphabetized: the table still reads as the run did."""
        root = FakeNode(
            process_label="WorkGraph<KoopmansDSCFWorkflow>",
            children=[
                FakeNode(link="wannierize", label="Wannierization", seconds=1.0),
                FakeNode(link="bands", label="Band structure", seconds=2.0),
            ],
        )

        rows = render(root)

        assert [row.label for row in rows[1:]] == ["Wannierization", "Band structure"]

    def test_an_interleaved_family_keeps_its_creation_order(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """A family split by another step is left exactly as it ran.

        The spin initialization is one such family: the staging nspin=2
        step lays out the restart files the real nspin=2 step reads, so
        sorting the two nspin rows together would put the reader above
        the writer.
        """
        root = FakeNode(
            process_label="WorkGraph<KoopmansDSCFWorkflow>",
            children=[
                FakeNode(link="dft_init_nspin1", label="DFT initialization (nspin=1)", seconds=1.0),
                FakeNode(
                    link="dft_init_nspin2_dummy",
                    label="DFT initialization (nspin=2, staging)",
                    seconds=2.0,
                ),
                FakeNode(link="dft_init_nspin2", label="DFT initialization (nspin=2)", seconds=3.0),
            ],
        )

        rows = render(root)

        assert [row.label for row in rows[1:]] == [
            "DFT initialization (nspin=1)",
            "DFT initialization (nspin=2, staging)",
            "DFT initialization (nspin=2)",
        ]

    def test_an_interleaved_family_is_not_sorted_across_the_row_splitting_it(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """Interleaved members hold their rows even when out of index order.

        Sorting a split family among the positions it already occupies
        would reorder it across the row in between, which is the
        reordering the contiguity requirement exists to prevent.
        """
        root = FakeNode(
            process_label="WorkGraph<KoopmansDSCFWorkflow>",
            children=[
                FakeNode(link="compute_alpha_orb_2", label="Orbital 2", seconds=1.0),
                FakeNode(link="ki_final", label="Final KI", seconds=2.0),
                FakeNode(link="compute_alpha_orb_1", label="Orbital 1", seconds=3.0),
            ],
        )

        rows = render(root)

        assert [row.label for row in rows[1:]] == ["Orbital 2", "Final KI", "Orbital 1"]

    def test_a_contiguous_family_sorts_amid_unmoved_rows(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """The fan-out counts properly; the rows around it do not move."""
        root = FakeNode(
            process_label="WorkGraph<KoopmansDSCFWorkflow>",
            children=[
                FakeNode(link="ki_trial", label="Trial KI", seconds=1.0),
                FakeNode(link="compute_alpha_orb_1", label="Orbital 1", seconds=2.0),
                FakeNode(link="compute_alpha_orb_10", label="Orbital 10", seconds=3.0),
                FakeNode(link="compute_alpha_orb_2", label="Orbital 2", seconds=4.0),
                FakeNode(link="ki_final", label="Final KI", seconds=5.0),
            ],
        )

        rows = render(root)

        assert [row.label for row in rows[1:]] == [
            "Trial KI",
            "Orbital 1",
            "Orbital 2",
            "Orbital 10",
            "Final KI",
        ]

    def test_simultaneous_namesakes_break_by_pk(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """Two identical labels created in the same instant still get a fixed order."""
        later = FakeNode(link="pw-scf", kind="calcjob", pk=99, seconds=1.0)
        earlier = FakeNode(link="pw-scf", kind="calcjob", pk=7, seconds=1.0)
        root = FakeNode(
            process_label="WorkGraph<Wannierize>", label="Wannierization", children=[later, earlier]
        )

        rows = render(root)

        assert [row.depth for row in rows] == [0, 1, 1]
        assert progress._ordered_children(cast("ProcessNode", root))[0][1].pk == 7

    def test_the_order_applies_at_every_level(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """A fan-out nested under a step is ordered like a top-level one."""
        fan_out = self._fan_out([2, 10, 1], [0.0, 0.1, 0.2])
        fan_out.link = "compute_orbital_screening_parameters"
        fan_out.label = "Orbital screening"
        root = FakeNode(
            process_label="WorkGraph<KoopmansDSCFWorkflow>",
            label="Koopmans ΔSCF",
            children=[fan_out],
        )

        rows = render(root)

        assert [(row.label, row.depth) for row in rows] == [
            ("Koopmans ΔSCF", 0),
            ("Orbital screening", 1),
            ("Orbital 1", 2),
            ("Orbital 2", 2),
            ("Orbital 10", 2),
        ]


class TestDescribeProcess:
    """A step is named by its own process; the rest is shown as recorded."""

    def test_the_name_comes_from_the_process(self, describe: _Describe) -> None:
        """``aiida-koopmans`` sets it; the display shows what it says."""
        node = FakeNode(link="dfpt", label="DFPT screening (spin down)")

        assert describe(node).text == "DFPT screening (spin down)"

    def test_a_process_carrying_no_name_shows_its_identifier(self, describe: _Describe) -> None:
        """Every run made before the plugin named its steps reads this way.

        The identifier is what the database holds for that step, so it is
        what the reader is shown; nothing guesses at the name it was
        meant to have.
        """
        assert describe(FakeNode(link="dft_init_nspin1")).text == "dft_init_nspin1"

    def test_the_root_is_named_by_the_run_it_is(self, describe: _Describe) -> None:
        """The root has no call link to read, so its name comes from the run."""
        node = FakeNode(process_label="WorkGraph<KoopmansDSCFWorkflow>", label="Koopmans ΔSCF")

        assert describe(node, is_root=True).text == "Koopmans ΔSCF"

    def test_an_unnamed_root_shows_the_graph_that_ran(self, describe: _Describe) -> None:
        """A run from before the routes named themselves keeps the graph's own name."""
        node = FakeNode(process_label="WorkGraph<KoopmansDSCFWorkflow>")

        assert describe(node, is_root=True).text == "KoopmansDSCFWorkflow"

    def test_the_binary_is_read_off_the_code_that_ran(self, describe: _Describe) -> None:
        """A code registered under a name of its own still answers with its binary.

        ``decompose`` is a second pw2wannier90.x, registered separately so
        a build with the decompose mode can be pointed at explicitly.
        """
        node = FakeNode(
            link="decompose_occ_1",
            label="Decomposition (occupied block 1)",
            kind="calcjob",
            executable="pw2wannier90.x",
        )

        assert describe(node).code == "pw2wannier90.x"

    def test_a_workflow_reports_no_binary(self, describe: _Describe) -> None:
        """A calculation runs one program; a workflow runs whatever its steps run."""
        assert describe(FakeNode(link="dfpt", label="DFPT screening")).code is None

    def test_a_role_is_read_from_the_step_and_not_from_its_name(self, describe: _Describe) -> None:
        """Which processes get rows is the display's own question, not the plugin's."""
        transparent = FakeNode(link="wannier90", process_label="Wannier90WorkChain")

        assert describe(transparent).transparent
        assert describe(FakeNode(link="ScreeningIteration", label="Iteration")).numbered


class TestIdentifiersShownAsRecorded:
    """Nothing is invented for a process that arrives carrying no name."""

    @pytest.mark.parametrize(
        "raw",
        [
            "scf_nscf",
            "nscf",
            "wann2kc",
            "fold_occ_1",
            "emit_wannier90_parameters",
            "SomeNewWorkChain",
        ],
    )
    def test_an_identifier_is_shown_exactly_as_written(self, raw: str) -> None:
        """A guess looks like a shipped typo; an internal name looks internal."""
        assert progress.prettify_label(raw) == raw

    @pytest.mark.parametrize(
        ("raw", "no_longer"),
        [
            ("wannier90_pp", "Wannier 90 Pp"),
            ("pw2wannier90", "Pw 2 Wannier 90"),
            ("nscf", "Nscf"),
            ("dfpt", "Dfpt"),
            ("wann2kc", "Wann 2 KC"),
        ],
    )
    def test_the_tokenizer_spellings_are_gone(self, raw: str, no_longer: str) -> None:
        """Splitting a code name into words produced spellings nobody types.

        ``Wannier 90 Pp`` and ``Dfpt`` read as typos the project shipped.
        """
        assert progress.prettify_label(raw) != no_longer

    def test_the_graph_envelope_is_peeled(self) -> None:
        """``WorkGraph<...>`` repeats what the row's position already says."""
        assert progress.prettify_label("WorkGraph<KoopmansDSCFWorkflow>") == "KoopmansDSCFWorkflow"

    def test_the_two_unnameable_wannier90_steps_show_their_identifiers(self) -> None:
        """``Wannier90WorkChain`` submits both with metadata of its own.

        It assigns ``inputs["metadata"] = {"call_link_label": ...}`` for
        the -pp and the minimization runs, discarding any label given
        from outside, where the four steps around them mutate
        ``metadata.call_link_label`` alone and keep theirs. Until that is
        levelled upstream, these two rows read as their link labels.
        """
        assert progress.prettify_label("wannier90_pp") == "wannier90_pp"
        assert progress.prettify_label("wannier90") == "wannier90"

    def test_a_role_is_read_together_with_the_process_behind_it(self) -> None:
        """``wannier90`` names the whole workchain in one place and one run in another."""
        assert progress.describe_label("wannier90", "Wannier90WorkChain").transparent
        assert not progress.describe_label("wannier90", "Wannier90BaseWorkChain").transparent


# --- the whole table, route by route ----------------------------------
#
# The tree shapes come from the built WorkGraphs (top level) plus the
# ``call_link_label`` map of the nested ``@task.graph`` bodies in
# ``aiida-koopmans2``, and each node's ``label`` is the name that package
# sets on it. The two wannier90.x steps of an upstream
# ``Wannier90WorkChain`` carry none: it replaces their metadata before
# submitting them, so the lookup still names those two.


def _graph(link: str, display: str, *children: FakeNode) -> FakeNode:
    """Build a ``@task.graph`` call: a container with a link label of its own."""
    return FakeNode(link=link, label=display, kind="workgraph", children=list(children))


def _chain(link: str, display: str, *children: FakeNode, process_label: str = "") -> FakeNode:
    """Build a WorkChain call, optionally carrying the class name that names it."""
    return FakeNode(
        link=link,
        label=display,
        kind="workchain",
        children=list(children),
        process_label=process_label or None,
    )


def _calc(link: str, display: str, executable: str) -> FakeNode:
    """Build a CalcJob: a leaf that runs one executable."""
    return FakeNode(link=link, label=display, executable=executable, kind="calcjob")


def _func(link: str) -> FakeNode:
    """Build a PyFunction: plumbing, never displayed."""
    return FakeNode(link=link, kind="calcfunc", is_pyfunction=True)


def _ground_state() -> FakeNode:
    return _graph(
        "scf_nscf",
        "Ground state",
        _chain("scf", "SCF", _calc("scf", "SCF", "pw.x")),
        _chain("nscf", "NSCF", _calc("nscf", "NSCF", "pw.x")),
    )


def _wannier90_workchain() -> FakeNode:
    return _chain(
        "wannier90",
        "",
        _chain("wannier90_pp", "", _calc("wannier90_pp", "", "wannier90.x")),
        _chain("pw2wannier90", "Overlaps", _calc("pw2wannier90", "Overlaps", "pw2wannier90.x")),
        _chain("wannier90", "", _calc("wannier90", "", "wannier90.x")),
        process_label="Wannier90WorkChain",
    )


def _wannierize_block(label: str, block: str) -> FakeNode:
    return _graph(
        f"wannierize_{label}",
        f"Wannierization ({block})",
        _func("emit_wannier90_parameters"),
        _wannier90_workchain(),
        _func("extract_wannier_output_files"),
    )


def _wannierize_blocks(*blocks: tuple[str, str], ground_state: bool = True) -> list[FakeNode]:
    """Build the per-block Wannierization's children.

    ``ground_state`` is what a caller that hands over an nscf scratch of
    its own leaves out: the DFPT route runs the ground state once and
    passes it in, so only the routes that own the Wannierization run it
    here.
    """
    return [
        *([_ground_state()] if ground_state else []),
        _chain("bands", "Band structure", _calc("bands", "Band structure", "pw.x")),
        _graph(
            "projwfc",
            "Atomic projections",
            _chain(
                "projwfc",
                "Atomic projections",
                _calc("projwfc", "Atomic projections", "projwfc.x"),
            ),
        ),
        *[_wannierize_block(label, block) for label, block in blocks],
        _func("collect_wannier_functions"),
    ]


def _alpha_filled(key: str, orbital: str) -> FakeNode:
    return _graph(
        f"compute_alpha_{key}",
        orbital,
        _calc("dft_n_minus_1", "DFT (N-1)", "kcp.x"),
        _func("compute_alpha"),
    )


def _alpha_empty(key: str, orbital: str) -> FakeNode:
    return _graph(
        f"compute_alpha_{key}",
        orbital,
        _calc("dft_n_plus_1_dummy", "DFT (N+1, staging)", "kcp.x"),
        _calc("pz_print", "PZ staging", "kcp.x"),
        _calc("dft_n_plus_1", "DFT (N+1)", "kcp.x"),
        _func("compute_alpha"),
    )


def _screening_iteration(link: str) -> FakeNode:
    return _graph(
        link,
        "Iteration",
        _calc("ki_trial", "Trial KI", "kcp.x"),
        _func("extract_self_hartree_from_kcp"),
        _graph(
            "compute_orbital_screening_parameters",
            "Orbital screening",
            _alpha_filled("orb_1", "Orbital 1"),
            _alpha_filled("orb_2", "Orbital 2"),
            _alpha_empty("orb_5", "Orbital 5"),
            _func("assemble_alpha_screening"),
        ),
        _func("max_alpha_error"),
    )


def _screening_parameters() -> FakeNode:
    return _graph(
        "ComputeScreeningParameters",
        "Screening parameters",
        _func("generate_alphas"),
        _screening_iteration("ScreeningIteration"),
        _graph(
            "refine_screening_parameters",
            "",
            _func("echo_alpha_screening"),
            _screening_iteration("screening_iteration"),
            _graph(
                "refine_screening_parameters",
                "",
                _screening_iteration("screening_iteration"),
            ),
        ),
    )


def _dscf_body(*, mlwf: bool) -> list[FakeNode]:
    head: list[FakeNode] = [_func("count_electrons_task")]
    if mlwf:
        head = [
            _func("make_supercell"),
            *head,
            _graph(
                "wannier_initialization",
                "Wannier initialization",
                _graph(
                    "wannierize",
                    "Wannierization",
                    *_wannierize_blocks(("occ_1", "occupied block 1"), ("emp_1", "empty block 1")),
                ),
                _graph(
                    "fold_to_supercell",
                    "Supercell folding",
                    _func("extract_occ_1"),
                    _calc(
                        "fold_occ_1",
                        "Supercell Wannier functions (occupied block 1)",
                        "wann2kcp.x",
                    ),
                    _func("extract_emp_1"),
                    _calc(
                        "fold_emp_1",
                        "Supercell Wannier functions (empty block 1)",
                        "wann2kcp.x",
                    ),
                    _calc(
                        "merge_evc_occupied1",
                        "Merged Wannier manifold (occupied, spin 1)",
                        "merge_evc.x",
                    ),
                    _calc(
                        "merge_evc0_empty1",
                        "Merged Wannier manifold (empty, spin 1)",
                        "merge_evc.x",
                    ),
                ),
                _calc("dft_dummy", "DFT staging", "kcp.x"),
                _calc("dft_init", "DFT initialization", "kcp.x"),
                _func("merge_groups"),
            ),
        ]
    else:
        head = [
            *head,
            # The three init sub-steps differ by the graph that wraps each;
            # the kcp.x call inside every one is the same calculation, so
            # they share its name and are told apart by their rows.
            _graph(
                "dft_init_nspin1",
                "DFT initialization (nspin=1)",
                _calc("dft_init", "DFT initialization", "kcp.x"),
            ),
            _graph(
                "dft_init_nspin2_dummy",
                "DFT initialization (nspin=2, staging)",
                _calc("dft_init", "DFT initialization", "kcp.x"),
            ),
            _func("convert_spin1_to_spin2"),
            _graph(
                "dft_init_nspin2",
                "DFT initialization (nspin=2)",
                _calc("dft_init", "DFT initialization", "kcp.x"),
            ),
        ]
    return [
        *head,
        _screening_parameters(),
        _graph("RunFinalKI", "Final KI", _calc("ki_final", "Final KI", "kcp.x")),
    ]


def _dfpt(suffix: str, channel: str) -> FakeNode:
    return _graph(
        f"dfpt{suffix}",
        f"DFPT screening{channel}",
        _func("prepare_kcw_wannier_files"),
        _calc("wann2kc", "Wannier gauge", "kcw.x"),
        _func("assign_orbital_groups"),
        _graph(
            "grouped_screen",
            "Orbital screening",
            _calc("screen_orb_1", "Orbital 1", "kcw.x"),
            _func("alpha_orb_1"),
            _calc("screen_orb_5", "Orbital 5", "kcw.x"),
            _func("alpha_orb_5"),
            _func("alphas_in_orbital_order"),
        ),
        _calc("ham", "Koopmans Hamiltonian", "kcw.x"),
    )


ROUTES: dict[str, FakeNode] = {
    "dft_bands": FakeNode(
        process_label="WorkGraph<RunPwBands>",
        label="DFT band structure",
        kind="workgraph",
        children=[
            _chain(
                "PwBandsWorkChain",
                "",
                _func("seekpath"),
                _chain("scf", "SCF", _calc("scf", "SCF", "pw.x")),
                _chain("bands", "Band structure", _calc("bands", "Band structure", "pw.x")),
            )
        ],
    ),
    "dft_eps": FakeNode(
        process_label="WorkGraph<DielectricTask>",
        label="Dielectric constant",
        kind="workgraph",
        children=[
            _chain("scf", "SCF", _calc("scf", "SCF", "pw.x")),
            _chain("ph", "Dielectric response", _calc("ph", "Dielectric response", "ph.x")),
            _func("extract_dielectric_constant"),
        ],
    ),
    "wannierize (per-block)": FakeNode(
        process_label="WorkGraph<WannierizeBlocks>",
        label="Wannierization",
        kind="workgraph",
        children=_wannierize_blocks(("occ_1", "occupied block 1"), ("emp_1", "empty block 1")),
    ),
    "wannierize (split)": FakeNode(
        process_label="WorkGraph<WannierizeBlocks>",
        label="Wannierization",
        kind="workgraph",
        children=[
            _ground_state(),
            _chain("bands", "Band structure", _calc("bands", "Band structure", "pw.x")),
            _graph(
                "projwfc",
                "Atomic projections",
                _chain(
                    "projwfc",
                    "Atomic projections",
                    _calc("projwfc", "Atomic projections", "projwfc.x"),
                ),
            ),
            _func("detect_band_groups"),
            _graph(
                "wannierize_split_block_1",
                "Split Wannierization (block 1)",
                _graph(
                    "wannierize_whole_block",
                    "Whole-block Wannierization",
                    _func("emit_wannier90_parameters"),
                    _wannier90_workchain(),
                ),
                _func("extract_win_file"),
                _calc("split_wannierization", "Parallel-transport split", "julia"),
                _graph(
                    "rewannierize_split_blocks",
                    "Per-group Wannierization",
                    _calc("wannier90_split_block_0", "Minimization (group 1)", "wannier90.x"),
                    _calc("wannier90_split_block_1", "Minimization (group 2)", "wannier90.x"),
                    _func("merge_split_block_products"),
                ),
            ),
            _func("collect_wannier_functions"),
        ],
    ),
    "wannierize (whole manifold)": FakeNode(
        process_label="WorkGraph<Wannierize>",
        label="Wannierization",
        kind="workgraph",
        children=[
            _chain(
                "Wannier90WorkChain",
                "",
                _chain("scf", "SCF", _calc("scf", "SCF", "pw.x")),
                _chain("nscf", "NSCF", _calc("nscf", "NSCF", "pw.x")),
                _chain(
                    "projwfc",
                    "Atomic projections",
                    _calc("projwfc", "Atomic projections", "projwfc.x"),
                ),
                _chain("wannier90_pp", "", _calc("wannier90_pp", "", "wannier90.x")),
                _chain(
                    "pw2wannier90", "Overlaps", _calc("pw2wannier90", "Overlaps", "pw2wannier90.x")
                ),
                _chain("wannier90", "", _calc("wannier90", "", "wannier90.x")),
            ),
            _chain("bands", "Band structure", _calc("bands", "Band structure", "pw.x")),
            _graph(
                "projwfc",
                "Atomic projections",
                _chain(
                    "projwfc",
                    "Atomic projections",
                    _calc("projwfc", "Atomic projections", "projwfc.x"),
                ),
            ),
        ],
    ),
    "singlepoint / DSCF (molecular)": FakeNode(
        process_label="WorkGraph<KoopmansDSCFWorkflow>",
        label="Koopmans ΔSCF",
        kind="workgraph",
        children=_dscf_body(mlwf=False),
    ),
    "singlepoint / DSCF (periodic, mlwfs)": FakeNode(
        process_label="WorkGraph<KoopmansDSCFWorkflow>",
        label="Koopmans ΔSCF",
        kind="workgraph",
        children=_dscf_body(mlwf=True),
    ),
    # ``eps_inf: auto`` adds the ph.x dielectric run; the manifolds are
    # Wannierized as several blocks each, so every block is numbered.
    "singlepoint / DFPT (eps_inf auto, multi-block manifolds)": FakeNode(
        process_label="WorkGraph<SinglepointDFPTWorkflow>",
        label="Koopmans DFPT",
        kind="workgraph",
        children=[
            _graph(
                "dielectric",
                "Dielectric constant",
                _chain("scf", "SCF", _calc("scf", "SCF", "pw.x")),
                _chain("ph", "Dielectric response", _calc("ph", "Dielectric response", "ph.x")),
            ),
            _ground_state(),
            _graph(
                "wannierize",
                "Wannierization",
                *_wannierize_blocks(
                    ("occ_1", "occupied block 1"),
                    ("emp_1", "empty block 1"),
                    ground_state=False,
                ),
            ),
            _dfpt("", ""),
        ],
    ),
    # A manifold Wannierized as one block is labelled ``occ`` / ``emp``,
    # without an index — the shape every unsplit DFPT run has.
    "singlepoint / DFPT (one block per manifold)": FakeNode(
        process_label="WorkGraph<SinglepointDFPTWorkflow>",
        label="Koopmans DFPT",
        kind="workgraph",
        children=[
            _ground_state(),
            _graph(
                "wannierize",
                "Wannierization",
                *_wannierize_blocks(
                    ("occ", "occupied block"), ("emp", "empty block"), ground_state=False
                ),
            ),
            _dfpt("", ""),
        ],
    ),
    "singlepoint / DFPT (collinear)": FakeNode(
        process_label="WorkGraph<SinglepointDFPTWorkflow>",
        label="Koopmans DFPT",
        kind="workgraph",
        children=[
            _ground_state(),
            _graph(
                "wannierize_up",
                "Wannierization (spin up)",
                *_wannierize_blocks(
                    ("occ_up_1", "occupied block 1, spin up"),
                    ("emp_up_1", "empty block 1, spin up"),
                    ground_state=False,
                ),
            ),
            _dfpt("_up", " (spin up)"),
            _graph(
                "wannierize_down",
                "Wannierization (spin down)",
                *_wannierize_blocks(
                    ("occ_down_1", "occupied block 1, spin down"),
                    ("emp_down_1", "empty block 1, spin down"),
                    ground_state=False,
                ),
            ),
            _dfpt("_down", " (spin down)"),
        ],
    ),
    "trajectory (ml train)": FakeNode(
        process_label="WorkGraph<TrajectoryWorkflow>",
        label="Trajectory",
        kind="workgraph",
        children=[
            _graph("dscf_snapshot_1", "Snapshot 1", *_dscf_body(mlwf=False)),
            _graph(
                "descriptors_snapshot_1",
                "Descriptors (snapshot 1)",
                _calc("decompose_occ_1", "Decomposition (occupied block 1)", "pw2wannier90.x"),
                _func("descriptors_occ_1"),
            ),
            _graph("dscf_snapshot_2", "Snapshot 2", *_dscf_body(mlwf=False)),
            _graph(
                "descriptors_snapshot_2",
                "Descriptors (snapshot 2)",
                _calc("decompose_occ_1", "Decomposition (occupied block 1)", "pw2wannier90.x"),
                _func("descriptors_occ_1"),
            ),
            _func("train_screening_model"),
        ],
    ),
    "trajectory (ml test, power_spectrum)": FakeNode(
        process_label="WorkGraph<TrajectoryWorkflow>",
        label="Trajectory",
        kind="workgraph",
        children=[
            _graph(
                "dscf_snapshot_1",
                "Snapshot 1",
                _graph(
                    "PredictScreeningParameters",
                    "Predicted screening parameters",
                    _calc("ki_trial", "Trial KI", "kcp.x"),
                    _graph(
                        "predicted_descriptors",
                        "Descriptors",
                        _calc(
                            "decompose_occ_1",
                            "Decomposition (occupied block 1)",
                            "pw2wannier90.x",
                        ),
                        _func("descriptors_occ_1"),
                    ),
                    _func("predict_alphas"),
                ),
                _graph(
                    "run_final_ki_predicted",
                    "Final KI (predicted alphas)",
                    _calc("ki_final", "Final KI", "kcp.x"),
                ),
            ),
            _func("evaluate_screening_model"),
        ],
    ),
}


def _unnamed(node: FakeNode) -> FakeNode:
    """Return a copy of ``node``'s tree with every process's label removed.

    What a run made before ``aiida-koopmans`` named its steps looks like.
    Fresh pks throughout, so the copy and the original can be resolved
    side by side.
    """
    return replace(
        node,
        label="",
        pk=next(_pk_counter),
        children=[_unnamed(child) for child in node.children],
    )


def _as_a_pre_fix_run(node: FakeNode, is_root: bool = False) -> FakeNode:
    """Return a copy of ``node``'s tree labelled the way a pre-fix run is.

    Before aiida-workgraph ``5b140d4``, ``WorkGraphEngine.on_create``
    replaced the label of every process launched for a ``@task.graph``
    with that graph's task name — the call link label, or the graph
    function's name for the run as a whole — so the names the plugin gave
    those never reached the database. Everything else keeps its label.
    """
    label = node.label
    if node.kind == "workgraph":
        envelope = re.fullmatch(r"WorkGraph<(.+)>", node.process_label or "")
        label = envelope.group(1) if is_root and envelope else node.link
    return replace(
        node,
        label=label,
        pk=next(_pk_counter),
        children=[_as_a_pre_fix_run(child) for child in node.children],
    )


def render_route_tables() -> str:
    """Return every route's progress table as the text of ``TABLES_FILE``."""
    registry: dict[int, FakeNode] = {}
    blocks = []
    with stubbed_lookups(registry):
        for name, root in ROUTES.items():
            _register(root, registry)
            lines = [f"########## {name}"]
            for row in progress.build_progress_rows(cast("ProcessNode", root)):
                lines.append(f"{'  ' * row.depth}{row.label}".ljust(56) + (row.code or ""))
            blocks.append("\n".join(line.rstrip() for line in lines))
    return "\n\n".join(blocks) + "\n"


class TestEveryRouteTable:
    """The whole display, for every route the CLI can run."""

    def test_the_tables_match_the_recorded_rendering(self) -> None:
        """A change to any name, code, or indent shows up as a diff here.

        The recorded file is the reviewable artefact: it is the table a
        user watching a run of each route sees, checked in so it can be
        read without running anything.
        """
        assert render_route_tables() == TABLES_FILE.read_text(encoding="utf-8")

    def test_a_run_whose_steps_are_unnamed_shows_their_identifiers(self) -> None:
        """A database made before the plugin named its steps still reads.

        Strip every label and each row falls back to the identifier
        provenance recorded for that step. The shape is untouched — same
        rows, same depths, same executables — because which steps get a
        row is read from the run rather than from what it is called; only
        the names degrade, and they degrade to what the database holds.
        """
        registry: dict[int, FakeNode] = {}
        with stubbed_lookups(registry):
            for name, root in ROUTES.items():
                unnamed = _unnamed(root)
                _register(root, registry)
                _register(unnamed, registry)
                named = progress.build_progress_rows(cast("ProcessNode", root))
                bare = progress.build_progress_rows(cast("ProcessNode", unnamed))

                assert [(r.depth, r.code) for r in bare] == [(r.depth, r.code) for r in named], name
                assert all(row.label for row in bare), name

    def test_a_run_recorded_before_the_label_fix_shows_its_identifiers(self) -> None:
        """A database written before aiida-workgraph ``5b140d4`` still reads.

        That engine replaced a ``@task.graph``'s label with the graph's
        task name, so every sub-graph of such a run carries an identifier
        where a name should be, while the calculations under them kept
        theirs. The mixture renders with the same shape as a run made
        today.
        """
        registry: dict[int, FakeNode] = {}
        with stubbed_lookups(registry):
            for name, root in ROUTES.items():
                as_run = _as_a_pre_fix_run(root, is_root=True)
                _register(root, registry)
                _register(as_run, registry)
                named = progress.build_progress_rows(cast("ProcessNode", root))
                old = progress.build_progress_rows(cast("ProcessNode", as_run))

                assert [(r.depth, r.code) for r in old] == [(r.depth, r.code) for r in named], name
                assert all(row.label for row in old), name

    def test_an_unnamed_dfpt_run_reads_as_its_link_labels(self) -> None:
        """What that degradation actually looks like, on one route.

        Worth stating outright rather than only as a shape: a reader
        replaying an old database sees the step names the database has,
        which are the internal ones.
        """
        registry: dict[int, FakeNode] = {}
        with stubbed_lookups(registry):
            root = _unnamed(ROUTES["singlepoint / DFPT (one block per manifold)"])
            _register(root, registry)
            rows = progress.build_progress_rows(cast("ProcessNode", root))

        assert [row.label for row in rows][:4] == [
            "SinglepointDFPTWorkflow",
            "scf_nscf",
            "scf",
            "nscf",
        ]

    def test_no_route_shows_a_python_class_name(self) -> None:
        """``Pw Bands Work Chain`` and friends never reach a user.

        The tables only. The failure summary reads process labels rather
        than call link labels, and most ``@task.graph`` names have no
        entry, so it still prints identifiers like
        ``RewannierizeSplitBlocks`` verbatim.
        """
        rendered = render_route_tables()
        for leaked in ("WorkChain", "Work Chain", "Calculation", "Workflow"):
            assert leaked not in rendered
