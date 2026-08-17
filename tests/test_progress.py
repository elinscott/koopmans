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

    python -c "import tests.test_progress as t; print(t.render_route_tables())" \
        > tests/data/progress_tables.txt
"""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from dataclasses import dataclass, field
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


@contextmanager
def stubbed_lookups(registry: dict[int, FakeNode]) -> Iterator[None]:
    """Point the assembler's three AiiDA lookups at a fake node tree."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(progress, "get_node_label", lambda node, include_code=True: node.label)
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


def _wrapped_calcjob(state: str = "waiting", **kwargs: Any) -> FakeNode:
    """Build root → ``DFT initialization (nspin=1)`` → the one kcp.x call it wraps."""
    calcjob = FakeNode(label="kcp-dft_init", kind="calcjob", state=state, **kwargs)
    wrapper = FakeNode(label="dft_init_nspin1", children=[calcjob])
    return FakeNode(process_label="WorkGraph<KoopmansDSCFWorkflow>", children=[wrapper])


class TestStableRows:
    """A row that has appeared is never withdrawn."""

    def test_the_wrapped_calcjob_never_gets_its_own_row(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """The duplicate is hidden while running, not only once it is done."""
        rows = render(_wrapped_calcjob(state="running"))

        assert [row.label for row in rows] == [
            "Koopmans Delta-SCF",
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
                "Koopmans Delta-SCF",
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
        calcjob = FakeNode(label="kcp-dft_init", kind="calcjob", state="created")
        wrapper = FakeNode(label="dft_init_nspin1", state="created", children=[calcjob])
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

    def test_process_functions_and_their_subtrees_stay_hidden(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """Plumbing tasks add no rows, nor do the processes they call."""
        buried = FakeNode(label="pw-scf", kind="calcjob")
        helper = FakeNode(label="build_iter_source", is_pyfunction=True, children=[buried])
        root = FakeNode(process_label="WorkGraph<KoopmansDSCFWorkflow>", children=[helper])

        assert [row.label for row in render(root)] == ["Koopmans Delta-SCF"]


class TestCollapsingAndTransparency:
    """Which containers keep a row, decided by what is inside them."""

    def _scf_nscf(self, scf_attempts: int = 1) -> FakeNode:
        """Build the ``scf_nscf`` sub-graph, with a restartable SCF beneath it."""
        scf = FakeNode(
            label="scf",
            children=[
                FakeNode(label="pw-scf", kind="calcjob", seconds=index)
                for index in range(scf_attempts)
            ],
        )
        nscf = FakeNode(
            label="nscf", seconds=10.0, children=[FakeNode(label="pw-nscf", kind="calcjob")]
        )
        ground_state = FakeNode(label="scf_nscf", children=[scf, nscf])
        return FakeNode(process_label="WorkGraph<SinglepointDFPTWorkflow>", children=[ground_state])

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
            label="PwBandsWorkChain",
            children=[
                FakeNode(label="scf", children=[FakeNode(label="pw-scf", kind="calcjob")]),
                FakeNode(
                    label="bands",
                    seconds=1.0,
                    children=[FakeNode(label="pw-bands", kind="calcjob")],
                ),
            ],
        )
        root = FakeNode(process_label="WorkGraph<RunPwBands>", children=[wrapper])

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
            label="PwBandsWorkChain",
            paused=True,
            children=[FakeNode(label="scf", state="running")],
        )
        root = FakeNode(process_label="WorkGraph<RunPwBands>", children=[wrapper])

        assert render(root)[0].state == "paused"

    def test_the_wannier90_workchain_is_transparent_but_its_minimization_is_not(
        self, render: Callable[[FakeNode], list[progress.ProcessRow]]
    ) -> None:
        """One link label, two meanings, told apart by the process behind it."""
        workchain = FakeNode(
            label="wannier90",
            process_label="Wannier90WorkChain",
            children=[
                FakeNode(
                    label="wannier90_pp",
                    process_label="Wannier90BaseWorkChain",
                    children=[FakeNode(label="wannier90-wannier90_pp", kind="calcjob")],
                ),
                FakeNode(
                    label="wannier90",
                    seconds=1.0,
                    process_label="Wannier90BaseWorkChain",
                    children=[FakeNode(label="wannier90-wannier90", kind="calcjob")],
                ),
            ],
        )
        root = FakeNode(process_label="WorkGraph<Wannierize>", children=[workchain])

        rows = render(root)

        assert [(row.label, row.depth, row.code) for row in rows] == [
            ("Wannierization", 0, None),
            ("Preprocessing", 1, "wannier90.x"),
            ("Minimization", 1, "wannier90.x"),
        ]


class TestScreeningIterationNumbering:
    """Iterations are counted by position, however deeply the recursion nests."""

    def _iteration(self, label: str, seconds: float) -> FakeNode:
        """Build one screening iteration, with a trial KI inside it."""
        return FakeNode(
            label=label,
            seconds=seconds,
            children=[
                FakeNode(label="kcp-ki_trial", kind="calcjob"),
                FakeNode(label="compute_orbital_screening_parameters", seconds=seconds + 0.1),
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
            label="refine_screening_parameters",
            seconds=3.0,
            children=[self._iteration("screening_iteration", 3.1)],
        )
        refine = FakeNode(
            label="refine_screening_parameters",
            seconds=2.0,
            children=[self._iteration("screening_iteration", 2.1), innermost],
        )
        compute = FakeNode(
            label="ComputeScreeningParameters",
            children=[self._iteration("ScreeningIteration", 1.0), refine],
        )
        root = FakeNode(process_label="WorkGraph<KoopmansDSCFWorkflow>", children=[compute])

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
            label="ComputeScreeningParameters",
            children=[self._iteration("ScreeningIteration", 1.0)],
        )
        root = FakeNode(process_label="WorkGraph<KoopmansDSCFWorkflow>", children=[compute])

        assert [row.label for row in render(root)][1:3] == ["Screening parameters", "Iteration 1"]


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
                FakeNode(label="wannierize", seconds=1.0),
                FakeNode(label="bands", seconds=2.0),
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
                FakeNode(label="dft_init_nspin1", seconds=1.0),
                FakeNode(label="dft_init_nspin2_dummy", seconds=2.0),
                FakeNode(label="dft_init_nspin2", seconds=3.0),
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
                FakeNode(label="compute_alpha_orb_2", seconds=1.0),
                FakeNode(label="ki_final", seconds=2.0),
                FakeNode(label="compute_alpha_orb_1", seconds=3.0),
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
                FakeNode(label="ki_trial", seconds=1.0),
                FakeNode(label="compute_alpha_orb_1", seconds=2.0),
                FakeNode(label="compute_alpha_orb_10", seconds=3.0),
                FakeNode(label="compute_alpha_orb_2", seconds=4.0),
                FakeNode(label="ki_final", seconds=5.0),
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
        fan_out.label = "compute_orbital_screening_parameters"
        root = FakeNode(process_label="WorkGraph<KoopmansDSCFWorkflow>", children=[fan_out])

        rows = render(root)

        assert [(row.label, row.depth) for row in rows] == [
            ("Koopmans Delta-SCF", 0),
            ("Orbital screening", 1),
            ("Orbital 1", 2),
            ("Orbital 2", 2),
            ("Orbital 10", 2),
        ]


class TestDescribeLabel:
    """Every name is looked up; nothing is guessed at."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("scf_nscf", "Ground state"),
            ("nscf", "NSCF"),
            ("kcw-wann2kc", "Wannier gauge"),
            ("merge_evc-merge_evc0_empty1", "Supercell wavefunctions (empty, spin 1)"),
            ("wann2kcp-fold_occ_1", "Folded Wannier functions (occupied block 1)"),
            ("wannierize_occ_up_1", "Wannierization (occupied block 1, spin up)"),
            ("wannier90-wannier90_split_block_0", "Minimization (group 1)"),
            ("kcw-screen_up_orb_2", "Orbital 2 (spin up)"),
            ("dfpt_down", "DFPT screening (spin down)"),
            ("dscf_snapshot_3", "Snapshot 3"),
        ],
    )
    def test_labels_read_as_the_step_they_stand_for(self, raw: str, expected: str) -> None:
        """A step is named after what it is, not after the program that runs it."""
        assert progress.prettify_label(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("kcp-dft_n_plus_1_dummy", "DFT (N+1, staging)"),
            ("kcp-pz_print", "PZ staging"),
            ("dft_init_nspin2_dummy", "DFT initialization (nspin=2, staging)"),
        ],
    )
    def test_a_run_that_only_writes_files_for_the_next_one_says_staging(
        self, raw: str, expected: str
    ) -> None:
        """``dummy`` and ``print`` are our words for the same thing."""
        assert progress.prettify_label(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "no_longer"),
        [
            ("wannier90_pp", "Wannier 90 Pp"),
            ("pw2wannier90", "Pw 2 Wannier 90"),
            ("nscf", "Nscf"),
            ("dfpt", "Dfpt"),
            ("projwfc", "Projwfc"),
            ("kcw-wann2kc", "Wann 2 KC"),
        ],
    )
    def test_the_tokenizer_spellings_are_gone(self, raw: str, no_longer: str) -> None:
        """Splitting a code name into words produced spellings nobody types.

        ``Wannier 90 Pp`` and ``Dfpt`` read as typos the project shipped.
        A name is now either in the display table or shown as written.
        """
        assert progress.prettify_label(raw) != no_longer

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("wannier90_pp", "Preprocessing"),
            ("pw2wannier90", "Overlaps"),
            ("wannier90", "Minimization"),
        ],
    )
    def test_the_wannier90_protocol_reads_as_the_mechanism_it_is(
        self, raw: str, expected: str
    ) -> None:
        """Three calls, one Wannierization: the rows name the steps of the protocol."""
        assert progress.prettify_label(raw) == expected

    def test_an_unmapped_label_is_shown_exactly_as_written(self) -> None:
        """A guess looks like a shipped typo; an internal name looks internal."""
        assert progress.prettify_label("emit_wannier90_parameters") == "emit_wannier90_parameters"
        assert progress.prettify_label("SomeNewWorkChain") == "SomeNewWorkChain"

    def test_a_label_is_read_together_with_the_process_behind_it(self) -> None:
        """``wannier90`` names the whole workchain in one place and one run in another."""
        assert progress.describe_label("wannier90", "Wannier90WorkChain").transparent
        assert not progress.describe_label("wannier90", "Wannier90BaseWorkChain").transparent
        assert progress.describe_label("wannier90", "Wannier90BaseWorkChain").text == "Minimization"

    def test_the_failure_summary_names_the_binary_that_failed(self) -> None:
        """A class name is not something a user has ever typed."""
        assert progress.prettify_label("PwCalculation") == "pw.x"
        assert progress.prettify_label("Wann2kcCalculation") == "kcw.x"
        assert progress.prettify_label("Pw2wannier90Calculation") == "pw2wannier90.x"

    @pytest.mark.parametrize(
        "raw", ["orb_1_filled_orbital_screening", "orb_10_empty_orbital_screening"]
    )
    def test_the_map_zone_rewrites_are_gone(self, raw: str) -> None:
        """Nothing has emitted these since the Map zones became for-loops."""
        assert progress.prettify_label(raw) == raw

    def test_a_code_prefix_becomes_the_executable_and_leaves_the_name_alone(self) -> None:
        """The code travels in its own column, spelled the way its authors spell it."""
        assert progress.describe_label("pw-scf") == progress.LabelDisplay("SCF", "pw.x")
        assert progress.describe_label("wannierjl-split_wannierization").code == "wannier.jl"

    def test_the_two_wannier90_rows_carry_the_same_executable(self) -> None:
        """No mode flag in the code column: the labels tell the rows apart."""
        assert progress.describe_label("wannier90-wannier90_pp").code == "wannier90.x"
        assert progress.describe_label("wannier90-wannier90").code == "wannier90.x"


# --- the whole table, route by route ----------------------------------
#
# The tree shapes come from the built WorkGraphs (top level) plus the
# ``call_link_label`` map of the nested ``@task.graph`` bodies in
# ``aiida-koopmans2``.


def _graph(label: str, *children: FakeNode) -> FakeNode:
    """Build a ``@task.graph`` call: a container with a link label of its own."""
    return FakeNode(label=label, kind="workgraph", children=list(children))


def _chain(label: str, *children: FakeNode, process_label: str = "") -> FakeNode:
    """Build a WorkChain call, optionally carrying the class name that names it."""
    return FakeNode(
        label=label,
        kind="workchain",
        children=list(children),
        process_label=process_label or None,
    )


def _calc(label: str) -> FakeNode:
    """Build a CalcJob: a leaf that runs one executable."""
    return FakeNode(label=label, kind="calcjob")


def _func(label: str) -> FakeNode:
    """Build a PyFunction: plumbing, never displayed."""
    return FakeNode(label=label, kind="calcfunc", is_pyfunction=True)


def _ground_state() -> FakeNode:
    return _graph("scf_nscf", _chain("scf", _calc("pw-scf")), _chain("nscf", _calc("pw-nscf")))


def _wannier90_workchain() -> FakeNode:
    return _chain(
        "wannier90",
        _chain("wannier90_pp", _calc("wannier90-wannier90_pp")),
        _chain("pw2wannier90", _calc("pw2wannier90-pw2wannier90")),
        _chain("wannier90", _calc("wannier90-wannier90")),
        process_label="Wannier90WorkChain",
    )


def _wannierize_block(label: str) -> FakeNode:
    return _graph(
        f"wannierize_{label}",
        _func("emit_wannier90_parameters"),
        _wannier90_workchain(),
        _func("extract_wannier_output_files"),
    )


def _wannierize_blocks(*block_labels: str) -> list[FakeNode]:
    return [
        _ground_state(),
        _chain("bands", _calc("pw-bands")),
        _graph("projwfc", _chain("projwfc", _calc("projwfc-projwfc"))),
        *[_wannierize_block(label) for label in block_labels],
        _func("collect_wannier_functions"),
    ]


def _alpha_filled(key: str) -> FakeNode:
    return _graph(f"compute_alpha_{key}", _calc("kcp-dft_n_minus_1"), _func("compute_alpha"))


def _alpha_empty(key: str) -> FakeNode:
    return _graph(
        f"compute_alpha_{key}",
        _calc("kcp-dft_n_plus_1_dummy"),
        _calc("kcp-pz_print"),
        _calc("kcp-dft_n_plus_1"),
        _func("compute_alpha"),
    )


def _screening_iteration(label: str) -> FakeNode:
    return _graph(
        label,
        _calc("kcp-ki_trial"),
        _func("extract_self_hartree_from_kcp"),
        _graph(
            "compute_orbital_screening_parameters",
            _alpha_filled("orb_1"),
            _alpha_filled("orb_2"),
            _alpha_empty("orb_5"),
            _func("assemble_alpha_screening"),
        ),
        _func("max_alpha_error"),
    )


def _screening_parameters() -> FakeNode:
    return _graph(
        "ComputeScreeningParameters",
        _func("generate_alphas"),
        _screening_iteration("ScreeningIteration"),
        _graph(
            "refine_screening_parameters",
            _func("echo_alpha_screening"),
            _screening_iteration("screening_iteration"),
            _graph(
                "refine_screening_parameters",
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
                _graph("wannierize", *_wannierize_blocks("occ_1", "emp_1")),
                _graph(
                    "fold_to_supercell",
                    _func("extract_occ_1"),
                    _calc("wann2kcp-fold_occ_1"),
                    _func("extract_emp_1"),
                    _calc("wann2kcp-fold_emp_1"),
                    _calc("merge_evc-merge_evc_occupied1"),
                    _calc("merge_evc-merge_evc0_empty1"),
                ),
                _calc("kcp-dft_dummy"),
                _calc("kcp-dft_init"),
                _func("merge_groups"),
            ),
        ]
    else:
        head = [
            *head,
            _graph("dft_init_nspin1", _calc("kcp-dft_init")),
            _graph("dft_init_nspin2_dummy", _calc("kcp-dft_init")),
            _func("convert_spin1_to_spin2"),
            _graph("dft_init_nspin2", _calc("kcp-dft_init")),
        ]
    return [*head, _screening_parameters(), _graph("RunFinalKI", _calc("kcp-ki_final"))]


def _dfpt(suffix: str) -> FakeNode:
    return _graph(
        f"dfpt{suffix}",
        _func("prepare_kcw_wannier_files"),
        _calc("kcw-wann2kc"),
        _func("assign_orbital_groups"),
        _graph(
            "grouped_screen",
            _calc("kcw-screen_orb_1"),
            _func("alpha_orb_1"),
            _calc("kcw-screen_orb_5"),
            _func("alpha_orb_5"),
            _func("alphas_in_orbital_order"),
        ),
        _calc("kcw-ham"),
    )


ROUTES: dict[str, FakeNode] = {
    "dft_bands": FakeNode(
        process_label="WorkGraph<RunPwBands>",
        kind="workgraph",
        children=[
            _chain(
                "PwBandsWorkChain",
                _func("seekpath"),
                _chain("scf", _calc("pw-scf")),
                _chain("bands", _calc("pw-bands")),
            )
        ],
    ),
    "dft_eps": FakeNode(
        process_label="WorkGraph<DielectricTask>",
        kind="workgraph",
        children=[
            _chain("scf", _calc("pw-scf")),
            _chain("ph", _calc("ph-ph")),
            _func("extract_dielectric_constant"),
        ],
    ),
    "wannierize (per-block)": FakeNode(
        process_label="WorkGraph<WannierizeBlocks>",
        kind="workgraph",
        children=_wannierize_blocks("occ_1", "emp_1"),
    ),
    "wannierize (split)": FakeNode(
        process_label="WorkGraph<WannierizeBlocks>",
        kind="workgraph",
        children=[
            _ground_state(),
            _chain("bands", _calc("pw-bands")),
            _graph("projwfc", _chain("projwfc", _calc("projwfc-projwfc"))),
            _func("detect_band_groups"),
            _graph(
                "wannierize_split_block_1",
                _graph(
                    "wannierize_whole_block",
                    _func("emit_wannier90_parameters"),
                    _wannier90_workchain(),
                ),
                _func("extract_win_file"),
                _calc("wannierjl-split_wannierization"),
                _graph(
                    "rewannierize_split_blocks",
                    _calc("wannier90-wannier90_split_block_0"),
                    _calc("wannier90-wannier90_split_block_1"),
                    _func("merge_split_block_products"),
                ),
            ),
            _func("collect_wannier_functions"),
        ],
    ),
    "wannierize (whole manifold)": FakeNode(
        process_label="WorkGraph<Wannierize>",
        kind="workgraph",
        children=[
            _chain(
                "Wannier90WorkChain",
                _chain("scf", _calc("pw-scf")),
                _chain("nscf", _calc("pw-nscf")),
                _chain("projwfc", _calc("projwfc-projwfc")),
                _chain("wannier90_pp", _calc("wannier90-wannier90_pp")),
                _chain("pw2wannier90", _calc("pw2wannier90-pw2wannier90")),
                _chain("wannier90", _calc("wannier90-wannier90")),
            ),
            _chain("bands", _calc("pw-bands")),
            _graph("projwfc", _chain("projwfc", _calc("projwfc-projwfc"))),
        ],
    ),
    "singlepoint / DSCF (molecular)": FakeNode(
        process_label="WorkGraph<KoopmansDSCFWorkflow>",
        kind="workgraph",
        children=_dscf_body(mlwf=False),
    ),
    "singlepoint / DSCF (periodic, mlwfs)": FakeNode(
        process_label="WorkGraph<KoopmansDSCFWorkflow>",
        kind="workgraph",
        children=_dscf_body(mlwf=True),
    ),
    "singlepoint / DFPT": FakeNode(
        process_label="WorkGraph<SinglepointDFPTWorkflow>",
        kind="workgraph",
        children=[
            _graph("dielectric", _chain("scf", _calc("pw-scf")), _chain("ph", _calc("ph-ph"))),
            _ground_state(),
            _graph("wannierize", *_wannierize_blocks("occ_1", "emp_1")),
            _dfpt(""),
        ],
    ),
    "singlepoint / DFPT (collinear)": FakeNode(
        process_label="WorkGraph<SinglepointDFPTWorkflow>",
        kind="workgraph",
        children=[
            _ground_state(),
            _graph("wannierize_up", *_wannierize_blocks("occ_up_1", "emp_up_1")),
            _dfpt("_up"),
            _graph("wannierize_down", *_wannierize_blocks("occ_down_1", "emp_down_1")),
            _dfpt("_down"),
        ],
    ),
    "trajectory (ml train)": FakeNode(
        process_label="WorkGraph<TrajectoryWorkflow>",
        kind="workgraph",
        children=[
            _graph("dscf_snapshot_1", *_dscf_body(mlwf=False)),
            _graph("dscf_snapshot_2", *_dscf_body(mlwf=False)),
            _func("train_screening_model"),
        ],
    ),
    "trajectory (ml test, power_spectrum)": FakeNode(
        process_label="WorkGraph<TrajectoryWorkflow>",
        kind="workgraph",
        children=[
            _graph(
                "dscf_snapshot_1",
                _graph(
                    "PredictScreeningParameters",
                    _calc("kcp-ki_trial"),
                    _graph(
                        "predicted_descriptors",
                        _calc("decompose-decompose_occ_1"),
                        _func("descriptors_occ_1"),
                    ),
                    _func("predict_alphas"),
                ),
                _graph("run_final_ki_predicted", _calc("kcp-ki_final")),
            ),
            _func("evaluate_screening_model"),
        ],
    ),
}


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
        assert render_route_tables() == TABLES_FILE.read_text()

    def test_no_route_shows_a_python_class_name(self) -> None:
        """``Pw Bands Work Chain`` and friends never reach a user."""
        rendered = render_route_tables()
        for leaked in ("WorkChain", "Work Chain", "Calculation", "Workflow"):
            assert leaked not in rendered
