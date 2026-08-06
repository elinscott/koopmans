"""Unit tests for the dump folder-name simplification."""

import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent
from typing import Any, ClassVar

import pytest

from koopmans.aiida.dumping import (
    _NODE_METADATA_FILE,
    _hoist_lone_calculations,
    _link_duplicate_files,
    _prune_source_only_step_folders,
    _prune_workflow_metadata,
    _renumber_step_folders,
    _simplify_folder_names,
    _strip_process_label_suffixes,
    _tidy_dumped_tree,
)

# The two ``node_type`` values a dumped step folder can carry: a python
# task or a CalcJob on one side, a workgraph or a WorkChain on the other.
_CALCULATION_NODE_TYPE = "process.calculation.calcfunction.CalcFunctionNode."
_WORKFLOW_NODE_TYPE = "process.workflow.workgraph.WorkGraphNode."


def _make_tree(root: Path, layout: Sequence[str | tuple[str, str]]) -> None:
    """Create the listed paths under ``root``.

    An entry ending in "/" becomes an empty folder. A plain entry becomes
    a file holding its own path, so that no two files collide by
    accident; a ``(path, content)`` entry becomes a file holding that
    content, which is how two files are given identical bytes.

    :param root: Folder the entries are created under.
    :param layout: Paths relative to ``root``, optionally with content.
    """
    for entry in layout:
        path, content = entry if isinstance(entry, tuple) else (entry, entry)
        target = root / path
        if path.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)


def _folders(root: Path) -> list[str]:
    """Return every folder under ``root``, relative and sorted."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir())


def _render(root: Path) -> str:
    """Return the tree under ``root``, drawn the way the tutorials quote it.

    A symlink is marked rather than spelled out; where a link points is
    the business of :class:`TestLinkDuplicateFiles`.
    """
    lines = [root.name]

    def draw(folder: Path, prefix: str) -> None:
        """Append a line per entry of ``folder``, then descend into each."""
        entries = sorted(folder.iterdir(), key=lambda p: p.name)
        for index, entry in enumerate(entries):
            last = index == len(entries) - 1
            arrow = " -> (link)" if entry.is_symlink() else ""
            lines.append(prefix + ("└── " if last else "├── ") + entry.name + arrow)
            if entry.is_dir() and not entry.is_symlink():
                draw(entry, prefix + ("    " if last else "│   "))

    draw(root, "")
    return "\n".join(lines)


def _metadata(name: str, node_type: str = _CALCULATION_NODE_TYPE) -> tuple[str, str]:
    """Return the ``aiida_node_metadata.yaml`` entry of the folder ``name``.

    Every dumped process folder carries one. The label makes each file's
    bytes its own, as a real dump's pk and uuid do, so that the symlink
    pass has nothing to collapse.
    """
    path = f"{name}/{_NODE_METADATA_FILE}" if name else _NODE_METADATA_FILE
    return (path, f"---\nNode data:\n  label: {name}\n  node_type: {node_type}\n")


def _calculation(name: str) -> list[str | tuple[str, str]]:
    """Return the layout entries of a calculation folder called ``name``."""
    return [_metadata(name), f"{name}/inputs/aiida.cpi", f"{name}/outputs/aiida.cpo"]


def _bookkeeping(name: str) -> list[str | tuple[str, str]]:
    """Return the layout entries of a python step that dumped only its source.

    Its ``source_file`` is the task's own code, which the installed
    package holds, and its metadata file names the node; neither is
    output, so nothing goes with the folder.
    """
    return [_metadata(name), f"{name}/inputs/source_file"]


def _lambdas(name: str, content: str = "trial-hamiltonian") -> list[tuple[str, str]]:
    """Return a python step's serialized ``ArrayData`` input.

    aiida-core dumps an ``ArrayData`` only as its consumer's
    ``function_inputs``, never under the task that produced it, so this
    is the tree's one copy of that array unless a sibling holds the same
    bytes. Every caller is the same task, so they share a ``source_file``
    too.
    """
    return [
        _metadata(name),
        (f"{name}/inputs/source_file", "def compute_alpha_from_dscf():\n    return alpha\n"),
        (f"{name}/inputs/function_inputs/trial_lambdas/lambdas.npy", content),
    ]


@pytest.mark.parametrize(
    ("dumped", "simplified"),
    [
        # pyfunction: the process label equals the link label, so the dump
        # never appends it — only the pk goes
        ("01-resolve_pseudo_family_task-4711", "01-resolve_pseudo_family_task"),
        # sub-workgraph: the WorkGraph<...> process label always repeats the
        # link label, so it goes along with the pk
        ("03-dft_init_nspin1-WorkGraph<dft_init_nspin1>-4712", "03-dft_init_nspin1"),
        # CalcJob: only the pk goes here, the class name being the
        # business of the later _strip_process_label_suffixes pass
        ("01-dft_init-KcpCalculation-4713", "01-dft_init-KcpCalculation"),
    ],
    ids=["pyfunction", "sub-workgraph", "calcjob"],
)
def test_simplify_folder_names(tmp_path: Path, dumped: str, simplified: str) -> None:
    """Strip the pk and the WorkGraph process label; keep other suffixes."""
    (tmp_path / dumped).mkdir()

    _simplify_folder_names(tmp_path)

    assert [d.name for d in tmp_path.iterdir()] == [simplified]


def test_simplify_folder_names_renames_nested_folders(tmp_path: Path) -> None:
    """Both a sub-workgraph folder and the folders inside it are renamed."""
    parent = "04-compute_alpha_orb_1-WorkGraph<compute_alpha_orb_1>-4714"
    child = "01-dft_n_minus_1-KcpCalculation-4715"
    (tmp_path / parent / child).mkdir(parents=True)

    _simplify_folder_names(tmp_path)

    all_dirs = sorted(str(d.relative_to(tmp_path)) for d in tmp_path.rglob("*") if d.is_dir())
    assert all_dirs == [
        "04-compute_alpha_orb_1",
        "04-compute_alpha_orb_1/01-dft_n_minus_1-KcpCalculation",
    ]


def test_simplify_folder_names_keeps_taken_names(tmp_path: Path) -> None:
    """A folder is left untouched if its simplified name already exists."""
    (tmp_path / "02-scf").mkdir()
    (tmp_path / "02-scf-WorkGraph<scf>-99").mkdir()

    _simplify_folder_names(tmp_path)

    all_dirs = sorted(d.name for d in tmp_path.iterdir())
    assert all_dirs == ["02-scf", "02-scf-WorkGraph<scf>-99"]


class TestPruneSourceOnlyStepFolders:
    """Only a folder holding nothing but a task's own code goes."""

    def test_a_source_only_step_goes_and_a_calculation_stays(self, tmp_path: Path) -> None:
        """A task's own code is not data the run produced."""
        _make_tree(
            tmp_path,
            [*_bookkeeping("01-count_electrons_task"), *_calculation("02-dft_init")],
        )

        _prune_source_only_step_folders(tmp_path)

        assert _folders(tmp_path) == ["02-dft_init", "02-dft_init/inputs", "02-dft_init/outputs"]

    def test_a_metadata_file_does_not_keep_a_calculation_free_step(self, tmp_path: Path) -> None:
        """The metadata file names the folder's node; it is not output.

        Every dumped step carries one, so counting it as content would
        stop the pass removing anything at all.
        """
        _make_tree(tmp_path, [_metadata("01-assign_orbital_groups")])

        _prune_source_only_step_folders(tmp_path)

        assert _folders(tmp_path) == []

    def test_an_empty_step_folder_goes(self, tmp_path: Path) -> None:
        """A step AiiDA dumped nothing at all for is the same case."""
        _make_tree(tmp_path, ["01-resolve_pseudo_family_task/"])

        _prune_source_only_step_folders(tmp_path)

        assert _folders(tmp_path) == []

    def test_a_step_holding_only_such_steps_goes_too(self, tmp_path: Path) -> None:
        """Holding nothing of one's own propagates outwards, however deep."""
        _make_tree(tmp_path, ["01-outer/01-middle/", *_bookkeeping("01-outer/02-inner")])

        _prune_source_only_step_folders(tmp_path)

        assert _folders(tmp_path) == []

    def test_a_step_holding_a_unique_payload_survives(self, tmp_path: Path) -> None:
        """A serialized array reaches disk only under the task that consumed it.

        ``compute_alpha_from_dscf`` records no outputs and looks exactly
        like a bookkeeping step, but its ``function_inputs`` hold the KI
        trial Hamiltonian, which appears nowhere else in the tree.
        """
        _make_tree(tmp_path, _lambdas("01-compute_alpha_from_dscf"))

        _prune_source_only_step_folders(tmp_path)

        assert _folders(tmp_path) == [
            "01-compute_alpha_from_dscf",
            "01-compute_alpha_from_dscf/inputs",
            "01-compute_alpha_from_dscf/inputs/function_inputs",
            "01-compute_alpha_from_dscf/inputs/function_inputs/trial_lambdas",
        ]

    def test_a_killed_calculation_survives(self, tmp_path: Path) -> None:
        """A calculation killed before retrieval has inputs and no outputs.

        Dumping it at all is the point of ``dump_unsealed``, so its input
        file has to outlive the pass.
        """
        _make_tree(tmp_path, ["01-dft_init/inputs/aiida.cpi"])

        _prune_source_only_step_folders(tmp_path)

        assert (tmp_path / "01-dft_init/inputs/aiida.cpi").is_file()

    def test_every_folder_with_a_duplicated_payload_survives(self, tmp_path: Path) -> None:
        """Repetition is no reason to drop a step: the reader still counts them.

        All three orbitals are handed the same trial Hamiltonian.
        Deleting two of the folders would leave the reader wondering what
        happened to those orbitals; the repeated bytes are the symlink
        pass's business, not this one's.
        """
        _make_tree(
            tmp_path,
            [
                *_lambdas("01-orb_1/01-compute_alpha_from_dscf"),
                *_lambdas("02-orb_2/01-compute_alpha_from_dscf"),
                *_lambdas("03-orb_3/01-compute_alpha_from_dscf"),
            ],
        )

        _prune_source_only_step_folders(tmp_path)

        survivors = [f for f in _folders(tmp_path) if f.endswith("compute_alpha_from_dscf")]
        assert survivors == [
            "01-orb_1/01-compute_alpha_from_dscf",
            "02-orb_2/01-compute_alpha_from_dscf",
            "03-orb_3/01-compute_alpha_from_dscf",
        ]

    def test_an_unnumbered_folder_is_never_pruned(self, tmp_path: Path) -> None:
        """Only "<NN>-<label>" step folders are candidates.

        A calculation's own ``inputs`` is not a step, and has to survive
        the pass whatever it holds.
        """
        _make_tree(tmp_path, _calculation("01-dft_init"))

        _prune_source_only_step_folders(tmp_path)

        assert _folders(tmp_path) == ["01-dft_init", "01-dft_init/inputs", "01-dft_init/outputs"]


class TestLinkDuplicateFiles:
    """Repeated bytes become one real file and links to it."""

    def test_a_staged_copy_points_at_the_step_that_produced_it(self, tmp_path: Path) -> None:
        """The producer keeps the file; its consumer links to it.

        A merge step writes ``hr.dat`` and the next step is handed it, so
        the copy under ``outputs`` is the one to keep whatever the tree
        order says.
        """
        _make_tree(
            tmp_path,
            [
                ("02-prepare_kcw_wannier_files/outputs/hr.dat", "merged-hamiltonian"),
                ("03-wann2kc/inputs/hr.dat", "merged-hamiltonian"),
            ],
        )

        assert _link_duplicate_files(tmp_path) == 1

        producer = tmp_path / "02-prepare_kcw_wannier_files/outputs/hr.dat"
        consumer = tmp_path / "03-wann2kc/inputs/hr.dat"
        assert not producer.is_symlink()
        assert consumer.is_symlink()
        assert consumer.resolve() == producer.resolve()
        assert consumer.read_text() == "merged-hamiltonian"

    def test_a_step_that_writes_and_reads_a_file_keeps_the_output_copy(
        self, tmp_path: Path
    ) -> None:
        """One step's own staged input points at its own output.

        The fold step stages the matrices it then writes back out, so
        both copies live under the one folder. ``inputs`` sorts before
        ``outputs``, so tree order alone would keep the staged copy; only
        the preference for a producing ``outputs`` copy picks the other.
        """
        _make_tree(
            tmp_path,
            [
                ("01-fold/inputs/wannier_files/aiida_u.mat", "unitary-matrix"),
                ("01-fold/outputs/wannier_files/aiida_u.mat", "unitary-matrix"),
            ],
        )

        assert _link_duplicate_files(tmp_path) == 1

        assert not (tmp_path / "01-fold/outputs/wannier_files/aiida_u.mat").is_symlink()
        assert (tmp_path / "01-fold/inputs/wannier_files/aiida_u.mat").is_symlink()

    def test_a_symlink_already_in_the_tree_is_left_alone(self, tmp_path: Path) -> None:
        """A tree that already holds links is not linked into a loop.

        Ranking an existing link as the copy to keep would replace the
        real file with a link to it, leaving the two pointing at each
        other and the bytes gone.
        """
        _make_tree(tmp_path, [("01-step/inputs/data.mat", "payload")])
        real = tmp_path / "01-step/inputs/data.mat"
        link = tmp_path / "01-step/outputs/data.mat"
        link.parent.mkdir(parents=True)
        link.symlink_to(os.path.relpath(real, link.parent))

        assert _link_duplicate_files(tmp_path) == 0

        assert not real.is_symlink()
        assert real.read_text() == "payload"
        assert link.read_text() == "payload"

    def test_empty_files_are_never_linked(self, tmp_path: Path) -> None:
        """A successful run leaves an empty scheduler log under every step.

        They are byte-identical to one another, so the rule would tie
        every step to one arbitrary sibling for no bytes saved.
        """
        _make_tree(
            tmp_path,
            [
                ("01-scf/outputs/_scheduler-stderr.txt", ""),
                ("02-nscf/outputs/_scheduler-stderr.txt", ""),
                ("03-bands/outputs/_scheduler-stderr.txt", ""),
            ],
        )

        assert _link_duplicate_files(tmp_path) == 0

        assert not any(p.is_symlink() for p in tmp_path.rglob("*"))

    def test_the_smallest_real_payload_still_links(self, tmp_path: Path) -> None:
        """Only length zero is exempt, so a 95-byte input still collapses.

        That is the smallest duplicated file either tutorial dump holds;
        exempting it would be exempting payload.
        """
        content = "x" * 95
        _make_tree(
            tmp_path,
            [("01-scf/outputs/aiida.in", content), ("02-nscf/inputs/aiida.in", content)],
        )

        assert _link_duplicate_files(tmp_path) == 1

        assert (tmp_path / "02-nscf/inputs/aiida.in").read_text() == content

    def test_a_unique_file_is_left_alone(self, tmp_path: Path) -> None:
        """Only repeated bytes are replaced."""
        _make_tree(tmp_path, [("01-scf/outputs/aiida.out", "one of a kind")])

        assert _link_duplicate_files(tmp_path) == 0

        assert not (tmp_path / "01-scf/outputs/aiida.out").is_symlink()

    def test_the_links_are_relative_and_survive_a_move(self, tmp_path: Path) -> None:
        """A dump handed to someone else still resolves."""
        _make_tree(
            tmp_path,
            [
                ("dump/01-scf/outputs/pseudo.upf", "pseudopotential"),
                ("dump/02-nscf/inputs/pseudo.upf", "pseudopotential"),
            ],
        )
        _link_duplicate_files(tmp_path / "dump")

        moved = tmp_path / "moved"
        shutil.move(str(tmp_path / "dump"), str(moved))

        link = moved / "02-nscf/inputs/pseudo.upf"
        assert not Path(os.readlink(link)).is_absolute()
        assert link.read_text() == "pseudopotential"

    def test_repeated_task_sources_collapse_to_one_copy(self, tmp_path: Path) -> None:
        """One task run once per orbital dumps its code under each folder."""
        source = "def compute_alpha_from_dscf():\n    return alpha\n"
        _make_tree(
            tmp_path,
            [
                ("01-orb_1/inputs/source_file", source),
                ("01-orb_1/inputs/function_inputs/lambdas.npy", "first payload"),
                ("02-orb_2/inputs/source_file", source),
                ("02-orb_2/inputs/function_inputs/lambdas.npy", "second payload"),
            ],
        )

        assert _link_duplicate_files(tmp_path) == 1

        first = tmp_path / "01-orb_1/inputs/source_file"
        second = tmp_path / "02-orb_2/inputs/source_file"
        assert not first.is_symlink()
        assert second.is_symlink()
        assert second.read_text() == source

    def test_a_task_source_is_never_linked_to_a_data_file(self, tmp_path: Path) -> None:
        """A task's source and a data file with its bytes both stay real files.

        Sources collapse among themselves, but never across the boundary:
        no data file is made to depend on a folder that carries only
        code. Both folders here survive the prune, so the two would
        otherwise be linked whichever way the ordering fell.
        """
        shared = "def compute():\n    return 1\n"
        _make_tree(
            tmp_path,
            [
                ("01-compute_alpha/inputs/function_inputs/payload.npy", shared),
                ("02-other_task/inputs/source_file", shared),
                ("02-other_task/inputs/function_inputs/kept.npy", "a payload of its own"),
            ],
        )

        _prune_source_only_step_folders(tmp_path)
        _link_duplicate_files(tmp_path)

        payload = tmp_path / "01-compute_alpha/inputs/function_inputs/payload.npy"
        source = tmp_path / "02-other_task/inputs/source_file"
        assert not payload.is_symlink()
        assert not source.is_symlink()
        assert payload.read_text() == source.read_text() == shared


class TestPruneWorkflowMetadata:
    """Which folders keep the file naming the node they came from."""

    def test_a_calculation_keeps_its_metadata(self, tmp_path: Path) -> None:
        """That file is the way back from a step's files to its node."""
        _make_tree(tmp_path, _calculation("01-scf"))

        _prune_workflow_metadata(tmp_path)

        assert (tmp_path / "01-scf" / _NODE_METADATA_FILE).is_file()

    def test_a_workflow_layer_loses_its_metadata(self, tmp_path: Path) -> None:
        """A workgraph folder runs nothing of its own; its children do."""
        _make_tree(tmp_path, [_metadata("01-scf_nscf", _WORKFLOW_NODE_TYPE)])

        _prune_workflow_metadata(tmp_path)

        assert not (tmp_path / "01-scf_nscf" / _NODE_METADATA_FILE).exists()

    def test_the_root_keeps_its_own(self, tmp_path: Path) -> None:
        """The root is a workgraph too, and its pk names the whole run."""
        _make_tree(tmp_path, [_metadata("", _WORKFLOW_NODE_TYPE)])

        _prune_workflow_metadata(tmp_path)

        assert (tmp_path / _NODE_METADATA_FILE).is_file()

    def test_a_file_of_an_unrecognized_shape_is_kept(self, tmp_path: Path) -> None:
        """Only an explicit workflow ``node_type`` deletes the file.

        A format this cannot read is kept, so an aiida-core change costs
        the reader a folder listing rather than the node it names.
        """
        _make_tree(tmp_path, [("01-step/" + _NODE_METADATA_FILE, "---\nsomething: else\n")])

        _prune_workflow_metadata(tmp_path)

        assert (tmp_path / "01-step" / _NODE_METADATA_FILE).is_file()

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(b"---\nNode data:\n  label: 'dft_ini", id="truncated-mid-write"),
            pytest.param(b"---\nNode data:\n\tlabel: dft_init\n", id="indented-with-a-tab"),
            pytest.param(b"\xff\xfe\x00\x01", id="not-text-at-all"),
        ],
    )
    def test_a_file_that_cannot_be_read_is_kept_and_the_sweep_goes_on(
        self, tmp_path: Path, content: bytes
    ) -> None:
        """A file this cannot parse costs its own folder's listing, nothing else.

        A dump interrupted part-way through writing one leaves exactly
        this. The passes that make the tree readable — the tidying, the
        ``README``, the ``model.json`` — all run after this one, so a
        file that stopped it would cost the reader the whole tree.
        """
        _make_tree(tmp_path, [_metadata("01-scf_nscf", _WORKFLOW_NODE_TYPE)])
        (tmp_path / "02-dft_init").mkdir()
        (tmp_path / "02-dft_init" / _NODE_METADATA_FILE).write_bytes(content)

        _prune_workflow_metadata(tmp_path)

        assert (tmp_path / "02-dft_init" / _NODE_METADATA_FILE).is_file()
        assert not (tmp_path / "01-scf_nscf" / _NODE_METADATA_FILE).exists()


class TestRenumberStepFolders:
    """Pruning leaves holes in the numbering; renumbering closes them."""

    def test_holes_close(self, tmp_path: Path) -> None:
        """The numbers are dump-side prose, so they count from one again."""
        _make_tree(tmp_path, ["03-dft_init_nspin1/a", "06-dft_init_nspin2/a", "08-RunFinalKI/a"])

        _renumber_step_folders(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "01-dft_init_nspin1",
            "02-dft_init_nspin2",
            "03-RunFinalKI",
        ]

    def test_an_indexed_family_is_numbered_in_counting_order(self, tmp_path: Path) -> None:
        """``orb_10`` follows ``orb_2``, though the dump wrote it between 1 and 2.

        The dump numbers a fan-out in creation order, which is
        lexicographic by map key. This is also the case that makes two
        folders swap numbers, which the renaming has to survive.
        """
        _make_tree(
            tmp_path,
            [
                "01-compute_alpha_orb_1/a",
                "02-compute_alpha_orb_10/a",
                "03-compute_alpha_orb_2/a",
            ],
        )

        _renumber_step_folders(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "01-compute_alpha_orb_1",
            "02-compute_alpha_orb_2",
            "03-compute_alpha_orb_10",
        ]

    def test_distinct_steps_keep_execution_order(self, tmp_path: Path) -> None:
        """Sequential steps are not alphabetized: the tree still reads as the run did."""
        _make_tree(tmp_path, ["01-wannierize/a", "02-dft_bands/a"])

        _renumber_step_folders(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["01-wannierize", "02-dft_bands"]

    def test_an_interleaved_family_keeps_its_run_order(self, tmp_path: Path) -> None:
        """A family split by another step is left exactly as it ran.

        The spin initialization is one such family: the dummy nspin=2
        step lays out the restart files the real nspin=2 step reads, so
        sorting ``nspin1``, ``nspin2`` and the dummy together would put
        the reader before the writer.
        """
        _make_tree(
            tmp_path,
            [
                "03-dft_init_nspin1/a",
                "04-dft_init_nspin2_dummy/a",
                "06-dft_init_nspin2/a",
            ],
        )

        _renumber_step_folders(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "01-dft_init_nspin1",
            "02-dft_init_nspin2_dummy",
            "03-dft_init_nspin2",
        ]

    def test_an_interleaved_family_is_not_sorted_across_the_step_splitting_it(
        self, tmp_path: Path
    ) -> None:
        """Interleaved members hold their positions even when out of index order.

        Sorting a split family among the positions it already occupies
        would reorder it across the step in between, which is the
        reordering the contiguity requirement exists to prevent.
        """
        _make_tree(
            tmp_path,
            ["01-compute_alpha_orb_2/a", "02-final_scf/a", "03-compute_alpha_orb_1/a"],
        )

        _renumber_step_folders(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "01-compute_alpha_orb_2",
            "02-final_scf",
            "03-compute_alpha_orb_1",
        ]

    def test_a_contiguous_family_sorts_amid_untouched_siblings(self, tmp_path: Path) -> None:
        """The fan-out counts properly; the steps around it do not move."""
        _make_tree(
            tmp_path,
            [
                "01-ki_trial/a",
                "02-compute_alpha_orb_1/a",
                "03-compute_alpha_orb_10/a",
                "04-compute_alpha_orb_2/a",
                "05-final_scf/a",
            ],
        )

        _renumber_step_folders(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "01-ki_trial",
            "02-compute_alpha_orb_1",
            "03-compute_alpha_orb_2",
            "04-compute_alpha_orb_10",
            "05-final_scf",
        ]

    def test_renumbering_reaches_every_level(self, tmp_path: Path) -> None:
        """Each folder's children are numbered from one independently."""
        _make_tree(tmp_path, ["07-outer/02-ScreeningIteration/05-max_alpha_error/a"])

        _renumber_step_folders(tmp_path)

        assert _folders(tmp_path) == [
            "01-outer",
            "01-outer/01-ScreeningIteration",
            "01-outer/01-ScreeningIteration/01-max_alpha_error",
        ]

    def test_the_original_zero_padding_is_kept(self, tmp_path: Path) -> None:
        """Padding comes from the dump, so nine steps do not become "1-"."""
        _make_tree(tmp_path, ["002-first/a", "007-second/a"])

        _renumber_step_folders(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["001-first", "002-second"]

    def test_a_ten_step_folder_keeps_two_digits(self, tmp_path: Path) -> None:
        """The widest surviving number sets the width for all of them."""
        _make_tree(tmp_path, [f"{number:02d}-orb_{number}/a" for number in range(1, 11)])

        _renumber_step_folders(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir())[-1] == "10-orb_10"


class TestStripProcessLabelSuffixes:
    """A step folder does not need to name the process class that ran."""

    def test_the_class_name_goes_from_a_calculation_folder(self, tmp_path: Path) -> None:
        """The step-local name stays; the appended class name goes."""
        _make_tree(tmp_path, _calculation("01-dft_n_plus_1-KcpCalculation"))

        _strip_process_label_suffixes(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["01-dft_n_plus_1"]

    def test_several_calculations_in_one_step_keep_their_own_names(self, tmp_path: Path) -> None:
        """Stripping is per folder, so a multi-calculation step stays legible."""
        _make_tree(
            tmp_path,
            [
                *_calculation("01-step/01-dft_n_plus_1_dummy-KcpCalculation"),
                *_calculation("01-step/02-pz_print-KcpCalculation"),
                *_calculation("01-step/03-scf-PwCalculation"),
            ],
        )

        _strip_process_label_suffixes(tmp_path)

        assert sorted(p.name for p in (tmp_path / "01-step").iterdir()) == [
            "01-dft_n_plus_1_dummy",
            "02-pz_print",
            "03-scf",
        ]

    def test_a_wrapped_workchain_layer_loses_its_suffix_too(self, tmp_path: Path) -> None:
        """The WorkChain wrapping a calculation is stripped like the calculation."""
        _make_tree(
            tmp_path,
            _calculation("01-scf_nscf/01-scf-PwBaseWorkChain/02-iteration_01-PwCalculation"),
        )

        _strip_process_label_suffixes(tmp_path)

        assert _folders(tmp_path)[:3] == [
            "01-scf_nscf",
            "01-scf_nscf/01-scf",
            "01-scf_nscf/01-scf/02-iteration_01",
        ]

    def test_a_capitalised_link_label_is_not_mistaken_for_a_process_label(
        self, tmp_path: Path
    ) -> None:
        """Stripping needs a "<NN>-<label>" to be left over.

        A link label is a python identifier and holds no dash, so a
        one-dash folder carries no appended process label however
        capitalised its label is.
        """
        _make_tree(tmp_path, [*_calculation("05-RunFinalKI"), "06-ScreeningIteration/a"])

        _strip_process_label_suffixes(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "05-RunFinalKI",
            "06-ScreeningIteration",
        ]

    def test_a_taken_name_keeps_the_suffix(self, tmp_path: Path) -> None:
        """Nothing is overwritten if two folders would strip to one name."""
        _make_tree(
            tmp_path,
            ["01-step/01-scf/", *_calculation("01-step/01-scf-PwCalculation")],
        )

        _strip_process_label_suffixes(tmp_path)

        assert sorted(p.name for p in (tmp_path / "01-step").iterdir()) == [
            "01-scf",
            "01-scf-PwCalculation",
        ]


class TestHoistLoneCalculations:
    """One calculation alone in a step needs no folder of its own."""

    def test_a_lone_calculation_moves_up_and_the_step_keeps_its_name(self, tmp_path: Path) -> None:
        """The step name is the informative one; the calculation layer goes."""
        _make_tree(tmp_path, _calculation("01-dft_init_nspin1/01-dft_init"))

        _hoist_lone_calculations(tmp_path)

        assert _folders(tmp_path) == [
            "01-dft_init_nspin1",
            "01-dft_init_nspin1/inputs",
            "01-dft_init_nspin1/outputs",
        ]
        assert (tmp_path / "01-dft_init_nspin1/outputs/aiida.cpo").is_file()

    def test_a_step_with_two_calculations_keeps_both_folders(self, tmp_path: Path) -> None:
        """Hoisting either one would lose the distinction between them."""
        _make_tree(
            tmp_path,
            [*_calculation("01-step/01-pz_print"), *_calculation("01-step/02-dft_n_plus_1")],
        )

        _hoist_lone_calculations(tmp_path)

        assert sorted(p.name for p in (tmp_path / "01-step").iterdir()) == [
            "01-pz_print",
            "02-dft_n_plus_1",
        ]

    def test_a_calculation_beside_anything_else_stays_put(self, tmp_path: Path) -> None:
        """Hoist only a step whose sole entry is the calculation folder."""
        _make_tree(tmp_path, [*_calculation("01-step/01-scf"), "01-step/notes.txt"])

        _hoist_lone_calculations(tmp_path)

        assert sorted(p.name for p in (tmp_path / "01-step").iterdir()) == ["01-scf", "notes.txt"]

    def test_a_chain_of_single_child_steps_collapses_by_one_layer(self, tmp_path: Path) -> None:
        """Every step name on the way to the calculation survives."""
        _make_tree(tmp_path, _calculation("01-outer/01-inner/01-dft_n_minus_1"))

        _hoist_lone_calculations(tmp_path)

        assert _folders(tmp_path) == [
            "01-outer",
            "01-outer/01-inner",
            "01-outer/01-inner/inputs",
            "01-outer/01-inner/outputs",
        ]


class TestTidyDumpedTree:
    """The four passes composed, on a tree shaped like the ozone tutorial's."""

    SCREENING = "07-ComputeScreeningParameters/02-ScreeningIteration"
    ORBITALS = f"{SCREENING}/04-compute_orbital_screening_parameters"

    OZONE_DUMP: ClassVar[list[str | tuple[str, str]]] = [
        # What the sweep before the tidying leaves: every calculation's
        # metadata file, and the root workgraph's own.
        _metadata("", _WORKFLOW_NODE_TYPE),
        "01-resolve_pseudo_family_task/",
        *_bookkeeping("02-count_electrons_task"),
        *_calculation("03-dft_init_nspin1/01-dft_init-KcpCalculation"),
        *_calculation("04-dft_init_nspin2_dummy/01-dft_init-KcpCalculation"),
        *_bookkeeping("05-convert_spin1_to_spin2"),
        *_calculation("06-dft_init_nspin2/01-dft_init-KcpCalculation"),
        *_bookkeeping("07-ComputeScreeningParameters/01-generate_alphas"),
        *_calculation(f"{SCREENING}/01-ki_trial-KcpCalculation"),
        *_bookkeeping(f"{SCREENING}/02-extract_self_hartree_from_kcp"),
        *_bookkeeping(f"{SCREENING}/03-assign_orbital_groups"),
        *_calculation(f"{ORBITALS}/01-compute_alpha_orb_1/01-dft_n_minus_1-KcpCalculation"),
        # Every orbital is handed the same KI trial Hamiltonian, which
        # reaches disk only here — so all but the last copy of it go.
        *_lambdas(f"{ORBITALS}/01-compute_alpha_orb_1/02-compute_alpha_from_dscf"),
        # The dump numbers the fan-out lexicographically by map key, so
        # orb_10 sits between orb_1 and orb_2 until the renumbering.
        *_calculation(f"{ORBITALS}/02-compute_alpha_orb_10/01-dft_n_plus_1_dummy-KcpCalculation"),
        *_calculation(f"{ORBITALS}/02-compute_alpha_orb_10/02-pz_print-KcpCalculation"),
        *_calculation(f"{ORBITALS}/02-compute_alpha_orb_10/03-dft_n_plus_1-KcpCalculation"),
        *_lambdas(f"{ORBITALS}/02-compute_alpha_orb_10/04-compute_alpha_from_dscf"),
        *_calculation(f"{ORBITALS}/03-compute_alpha_orb_2/01-dft_n_minus_1-KcpCalculation"),
        *_lambdas(f"{ORBITALS}/03-compute_alpha_orb_2/02-compute_alpha_from_dscf"),
        *_bookkeeping(f"{ORBITALS}/11-expand_alphas_by_group"),
        *_bookkeeping(f"{SCREENING}/05-max_alpha_error"),
        *_calculation("08-RunFinalKI/01-ki_final-KcpCalculation"),
    ]

    TIDIED = """\
        ozone
        ├── 01-dft_init_nspin1
        │   ├── aiida_node_metadata.yaml
        │   ├── inputs
        │   │   └── aiida.cpi
        │   └── outputs
        │       └── aiida.cpo
        ├── 02-dft_init_nspin2_dummy
        │   ├── aiida_node_metadata.yaml
        │   ├── inputs
        │   │   └── aiida.cpi
        │   └── outputs
        │       └── aiida.cpo
        ├── 03-dft_init_nspin2
        │   ├── aiida_node_metadata.yaml
        │   ├── inputs
        │   │   └── aiida.cpi
        │   └── outputs
        │       └── aiida.cpo
        ├── 04-ComputeScreeningParameters
        │   └── 01-ScreeningIteration
        │       ├── 01-ki_trial
        │       │   ├── aiida_node_metadata.yaml
        │       │   ├── inputs
        │       │   │   └── aiida.cpi
        │       │   └── outputs
        │       │       └── aiida.cpo
        │       └── 02-compute_orbital_screening_parameters
        │           ├── 01-compute_alpha_orb_1
        │           │   ├── 01-dft_n_minus_1
        │           │   │   ├── aiida_node_metadata.yaml
        │           │   │   ├── inputs
        │           │   │   │   └── aiida.cpi
        │           │   │   └── outputs
        │           │   │       └── aiida.cpo
        │           │   └── 02-compute_alpha_from_dscf
        │           │       ├── aiida_node_metadata.yaml
        │           │       └── inputs
        │           │           ├── function_inputs
        │           │           │   └── trial_lambdas
        │           │           │       └── lambdas.npy
        │           │           └── source_file
        │           ├── 02-compute_alpha_orb_2
        │           │   ├── 01-dft_n_minus_1
        │           │   │   ├── aiida_node_metadata.yaml
        │           │   │   ├── inputs
        │           │   │   │   └── aiida.cpi
        │           │   │   └── outputs
        │           │   │       └── aiida.cpo
        │           │   └── 02-compute_alpha_from_dscf
        │           │       ├── aiida_node_metadata.yaml
        │           │       └── inputs
        │           │           ├── function_inputs
        │           │           │   └── trial_lambdas
        │           │           │       └── lambdas.npy -> (link)
        │           │           └── source_file -> (link)
        │           └── 03-compute_alpha_orb_10
        │               ├── 01-dft_n_plus_1_dummy
        │               │   ├── aiida_node_metadata.yaml
        │               │   ├── inputs
        │               │   │   └── aiida.cpi
        │               │   └── outputs
        │               │       └── aiida.cpo
        │               ├── 02-pz_print
        │               │   ├── aiida_node_metadata.yaml
        │               │   ├── inputs
        │               │   │   └── aiida.cpi
        │               │   └── outputs
        │               │       └── aiida.cpo
        │               ├── 03-dft_n_plus_1
        │               │   ├── aiida_node_metadata.yaml
        │               │   ├── inputs
        │               │   │   └── aiida.cpi
        │               │   └── outputs
        │               │       └── aiida.cpo
        │               └── 04-compute_alpha_from_dscf
        │                   ├── aiida_node_metadata.yaml
        │                   └── inputs
        │                       ├── function_inputs
        │                       │   └── trial_lambdas
        │                       │       └── lambdas.npy -> (link)
        │                       └── source_file -> (link)
        ├── 05-RunFinalKI
        │   ├── aiida_node_metadata.yaml
        │   ├── inputs
        │   │   └── aiida.cpi
        │   └── outputs
        │       └── aiida.cpo
        ├── README
        └── aiida_node_metadata.yaml"""

    def test_the_ozone_dump_tidies_to_the_documented_tree(self, tmp_path: Path) -> None:
        """Every pass is visible in the result, and they compose in one order.

        The bookkeeping steps are gone, the survivors count from one, no
        folder names a CalcJob class, and each step that ran a single
        calculation holds that calculation's own ``inputs``/``outputs``.
        """
        root = tmp_path / "ozone"
        _make_tree(root, self.OZONE_DUMP)

        _tidy_dumped_tree(root)

        assert _render(root) == dedent(self.TIDIED)

    def test_a_wrapped_workchain_reduces_to_its_link_label(self, tmp_path: Path) -> None:
        """A wrapped WorkChain sheds both its class name and its own layer.

        The zno shape: a ``PwBaseWorkChain`` whose kpoint step recorded
        no outputs and whose one iteration is the calculation itself.
        """
        root = tmp_path / "zno"
        _make_tree(
            root,
            [
                *_bookkeeping("01-scf_nscf/01-scf-PwBaseWorkChain/01-create_kpoints_from_distance"),
                *_calculation("01-scf_nscf/01-scf-PwBaseWorkChain/02-iteration_01-PwCalculation"),
            ],
        )

        _tidy_dumped_tree(root)

        assert _folders(root) == [
            "01-scf_nscf",
            "01-scf_nscf/01-scf",
            "01-scf_nscf/01-scf/inputs",
            "01-scf_nscf/01-scf/outputs",
        ]


class TestDumpModelJson:
    """A trained model's Dict output gets a ``model.json`` convenience copy."""

    @staticmethod
    def _run_train_task(aiida_profile_clean: object) -> object:
        """Run the training task on a two-row dataset; return its process node."""
        from aiida_koopmans.functionals import Correction
        from aiida_koopmans.ml import MLDescriptor
        from aiida_koopmans.variational_orbitals import VariationalOrbitalType
        from aiida_koopmans.workgraphs.ml import train_screening_model
        from aiida_workgraph import WorkGraph

        wg = WorkGraph("train_for_dump")
        wg.add_task(
            train_screening_model,
            name="train",
            datasets={
                "snapshot_1": {
                    "descriptors": [[-1.0], [-2.0]],
                    "alpha_targets": [0.5, 0.6],
                    "filled": [True, False],
                    "labels": ["orb_1", "orb_2"],
                }
            },
            estimator="linear_regression",
            occ_and_emp_together=True,
            descriptor=MLDescriptor.SELF_HARTREE,
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
        )
        wg.run()
        children = [link.node for link in wg.process.base.links.get_outgoing().all()]
        return next(node for node in children if hasattr(node, "is_finished_ok"))

    def test_model_json_written_from_the_model_output(
        self, aiida_profile_clean: object, tmp_path: Path
    ) -> None:
        """The full stamped model dict lands in ``model.json``."""
        import json

        from koopmans.aiida.dumping import _dump_model_json

        train: Any = self._run_train_task(aiida_profile_clean)
        assert train.is_finished_ok, train.exception

        _dump_model_json(train, tmp_path)

        written = json.loads((tmp_path / "model.json").read_text())
        assert written == train.outputs.model.get_dict()
        assert written["correction"] == "ki"

    def test_process_without_model_output_writes_nothing(
        self, aiida_profile_clean: object, tmp_path: Path
    ) -> None:
        """A process without a ``model`` Dict output dumps no file."""
        from koopmans.aiida.dumping import _dump_model_json

        train: Any = self._run_train_task(aiida_profile_clean)
        # The surrounding WorkGraph node exposes no top-level ``model``
        # output, so it stands in for any modelless process.
        from aiida import orm

        workgraph_node = next(
            link.node
            for link in train.base.links.get_incoming().all()
            if isinstance(link.node, orm.ProcessNode)
        )

        _dump_model_json(workgraph_node, tmp_path)

        assert not (tmp_path / "model.json").exists()


class TestDumpedNodeMetadata:
    """What a real dump leaves behind for finding a step's node again."""

    @staticmethod
    def _dump_a_run(output_path: Path) -> tuple[Path, Any]:
        """Dump a workgraph of one bookkeeping task and one that keeps a file.

        Returns the dumped tree and the root workgraph node. The graph is
        the smallest one carrying both cases the sweep separates: a step
        whose only trace is its own source, and one whose output reaches
        disk.
        """
        from aiida import orm
        from aiida_workgraph import WorkGraph, task

        from koopmans.aiida.dumping import dump_workgraph

        @task  # type: ignore[untyped-decorator]
        def count_electrons(charge: int) -> int:
            """Return an electron count, leaving nothing on disk but this code."""
            return 8 - charge

        @task  # type: ignore[untyped-decorator]
        def write_note(text: str) -> orm.SinglefileData:
            """Return ``text`` as the one file this run puts on disk."""
            import io

            return orm.SinglefileData(io.BytesIO(text.encode()), filename="note.txt")

        @task.graph  # type: ignore[untyped-decorator]
        def run(text: str) -> orm.SinglefileData:
            """Run both tasks, so the dump holds one step of each kind."""
            count_electrons(charge=0)
            note: orm.SinglefileData = write_note(text=text).result
            return note

        wg = WorkGraph("dump_metadata")
        wg.add_task(run, name="run", text="hello")
        wg.run()

        return dump_workgraph(wg.process, output_path), wg.process

    def test_the_root_names_the_process_the_run_was(
        self, aiida_profile_clean: object, tmp_path: Path
    ) -> None:
        """The one pk a reader needs to query the whole run is on disk."""
        import yaml

        dumped, process = self._dump_a_run(tmp_path / "dump")

        metadata = yaml.safe_load((dumped / _NODE_METADATA_FILE).read_text())
        assert metadata["Node data"]["pk"] == process.pk
        assert metadata["Node data"]["uuid"] == process.uuid

    def test_a_calculation_keeps_the_file_naming_its_node(
        self, aiida_profile_clean: object, tmp_path: Path
    ) -> None:
        """The step whose output survives carries the pk that produced it."""
        import yaml
        from aiida import orm

        dumped, _ = self._dump_a_run(tmp_path / "dump")

        metadata = yaml.safe_load((dumped / "01-run" / _NODE_METADATA_FILE).read_text())
        assert (dumped / "01-run/outputs/result/note.txt").read_text() == "hello"
        assert orm.load_node(metadata["Node data"]["pk"]).uuid == metadata["Node data"]["uuid"]

    def test_no_step_below_the_root_carries_a_workflow_node_metadata(
        self, aiida_profile_clean: object, tmp_path: Path
    ) -> None:
        """Only the root workgraph keeps its own; the layers inside do not."""
        import yaml

        dumped, _ = self._dump_a_run(tmp_path / "dump")

        below = [p for p in dumped.rglob(_NODE_METADATA_FILE) if p.parent != dumped]
        assert below
        for path in below:
            node_type = yaml.safe_load(path.read_text())["Node data"]["node_type"]
            assert node_type.startswith("process.calculation."), path

    def test_the_metadata_carries_no_node_attributes(
        self, aiida_profile_clean: object, tmp_path: Path
    ) -> None:
        """``include_attributes=False`` keeps the file to the node's identity.

        A workgraph's attributes hold the serialized graph, which grows
        with the graph; without them every file stays well under a
        kilobyte.
        """
        import yaml

        dumped, _ = self._dump_a_run(tmp_path / "dump")

        written = list(dumped.rglob(_NODE_METADATA_FILE))
        assert written
        for path in written:
            assert "Node attributes" not in yaml.safe_load(path.read_text())
            assert path.stat().st_size < 1024, path

    def test_a_step_that_ran_no_calculation_is_still_pruned(
        self, aiida_profile_clean: object, tmp_path: Path
    ) -> None:
        """The bookkeeping task's metadata file does not keep its folder.

        ``count_electrons`` leaves nothing but its own source, so the
        tree that reaches the reader holds one step, not two.
        """
        dumped, _ = self._dump_a_run(tmp_path / "dump")

        assert not any(p.name.endswith("count_electrons") for p in dumped.rglob("*"))
        assert [p.name for p in dumped.iterdir() if p.is_dir()] == ["01-run"]
