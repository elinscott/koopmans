"""Unit tests for the dump folder-name simplification."""

import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent
from typing import Any, ClassVar

import pytest

from koopmans.aiida.dumping import (
    NODE_METADATA_FILE,
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
    path = f"{name}/{NODE_METADATA_FILE}" if name else NODE_METADATA_FILE
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

        assert (tmp_path / "01-scf" / NODE_METADATA_FILE).is_file()

    def test_a_workflow_layer_loses_its_metadata(self, tmp_path: Path) -> None:
        """A workgraph folder runs nothing of its own; its children do."""
        _make_tree(tmp_path, [_metadata("01-scf_nscf", _WORKFLOW_NODE_TYPE)])

        _prune_workflow_metadata(tmp_path)

        assert not (tmp_path / "01-scf_nscf" / NODE_METADATA_FILE).exists()

    def test_the_root_keeps_its_own(self, tmp_path: Path) -> None:
        """The root is a workgraph too, and its pk names the whole run."""
        _make_tree(tmp_path, [_metadata("", _WORKFLOW_NODE_TYPE)])

        _prune_workflow_metadata(tmp_path)

        assert (tmp_path / NODE_METADATA_FILE).is_file()

    def test_a_file_of_an_unrecognized_shape_is_kept(self, tmp_path: Path) -> None:
        """Only an explicit workflow ``node_type`` deletes the file.

        A format this cannot read is kept, so an aiida-core change costs
        the reader a folder listing rather than the node it names.
        """
        _make_tree(tmp_path, [("01-step/" + NODE_METADATA_FILE, "---\nsomething: else\n")])

        _prune_workflow_metadata(tmp_path)

        assert (tmp_path / "01-step" / NODE_METADATA_FILE).is_file()

    def test_a_workflow_node_type_named_elsewhere_in_the_file_decides_nothing(
        self, tmp_path: Path
    ) -> None:
        """``Node data.node_type`` alone says what the folder came from.

        The other fields are free text — a label, a description someone
        wrote — so matching on the text of the file rather than on that
        one key would delete a calculation's file for quoting a workflow.
        """
        _make_tree(
            tmp_path,
            [
                (
                    f"01-scf/{NODE_METADATA_FILE}",
                    "---\nNode data:\n"
                    f"  description: rerun of the {_WORKFLOW_NODE_TYPE} that stopped\n"
                    f"  node_type: {_CALCULATION_NODE_TYPE}\n",
                )
            ],
        )

        _prune_workflow_metadata(tmp_path)

        assert (tmp_path / "01-scf" / NODE_METADATA_FILE).is_file()

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
        """A file this cannot parse costs its own folder's listing and its hoist.

        A dump interrupted part-way through writing one leaves exactly
        this. Keeping the file costs the reader that folder's listing,
        and costs the layer its flattening: a step wrapping a lone
        calculation keeps both folders where a readable workflow file
        would have left one. What it must not cost is the rest of the
        tree — the tidying, the ``README`` and the ``model.json`` all
        run after this pass, so a file that stopped it would cost the
        reader everything.

        The file the sweep must still delete sits below the one it
        cannot read: ``rglob`` yields a folder's own matches before
        descending into it, while the order of two sibling folders is
        the filesystem's.
        """
        _make_tree(tmp_path, [_metadata("01-scf_nscf/01-scf", _WORKFLOW_NODE_TYPE)])
        (tmp_path / "01-scf_nscf" / NODE_METADATA_FILE).write_bytes(content)

        _prune_workflow_metadata(tmp_path)

        assert (tmp_path / "01-scf_nscf" / NODE_METADATA_FILE).is_file()
        assert not (tmp_path / "01-scf_nscf" / "01-scf" / NODE_METADATA_FILE).exists()


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

    def test_a_lone_top_level_calculation_keeps_its_folder(self, tmp_path: Path) -> None:
        """The root's own metadata file is one of those other entries.

        Hoisting into the root would write the calculation's file over
        the root's, so the pk naming the whole run would name one step of
        it, and the step's name would go with the folder.
        """
        _make_tree(tmp_path, [_metadata("", _WORKFLOW_NODE_TYPE), *_calculation("01-write_note")])

        _hoist_lone_calculations(tmp_path)

        assert (tmp_path / "01-write_note/outputs/aiida.cpo").is_file()
        assert _WORKFLOW_NODE_TYPE in (tmp_path / NODE_METADATA_FILE).read_text()

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
        """Dump a workgraph of three tasks, two of which leave a file behind.

        Returns the dumped tree and the root workgraph node. The graph is
        the smallest one carrying every case the metadata sweep tells
        apart: a step whose only trace is its own source, a step whose
        lone calculation is hoisted into it, and a second calculation to
        tell the first one's file from.
        """
        from aiida import orm
        from aiida_workgraph import WorkGraph, task

        from koopmans.aiida.dumping import dump_workgraph

        @task  # type: ignore[untyped-decorator]
        def count_electrons(charge: int) -> int:
            """Return an electron count, leaving nothing on disk but this code.

            A plain ``@task`` is a ``CalcFunctionNode`` — bookkeeping,
            never a genuine calculation (see
            ``koopmans.aiida.dumping._is_calcjob_step``) — so it gets no
            output listing for this ``int`` however real it is
            (see ``TestStepIoListing``).
            """
            return 8 - charge

        @task  # type: ignore[untyped-decorator]
        def write_note(text: str) -> orm.SinglefileData:
            """Return ``text`` as a file this run puts on disk."""
            import io

            return orm.SinglefileData(io.BytesIO(text.encode()), filename="note.txt")

        @task  # type: ignore[untyped-decorator]
        def write_summary(text: str) -> orm.SinglefileData:
            """Return ``text`` as the other file this run puts on disk."""
            import io

            return orm.SinglefileData(io.BytesIO(text.encode()), filename="summary.txt")

        @task.graph  # type: ignore[untyped-decorator]
        def run(text: str) -> orm.SinglefileData:
            """Run both tasks, so the dump holds one step of each kind."""
            count_electrons(charge=0)
            note: orm.SinglefileData = write_note(text=text).result
            return note

        wg = WorkGraph("dump_metadata")
        wg.add_task(run, name="run", text="hello")
        wg.add_task(write_summary, name="write_summary", text="three orbitals screened")
        wg.run()

        return dump_workgraph(wg.process, output_path), wg.process

    def test_the_root_names_the_process_the_run_was(
        self, aiida_profile_clean: object, tmp_path: Path
    ) -> None:
        """The one pk a reader needs to query the whole run is on disk."""
        import yaml

        dumped, process = self._dump_a_run(tmp_path / "dump")

        metadata = yaml.safe_load((dumped / NODE_METADATA_FILE).read_text())
        assert metadata["Node data"]["pk"] == process.pk
        assert metadata["Node data"]["uuid"] == process.uuid

    def test_each_calculation_keeps_the_file_naming_its_own_node(
        self, aiida_profile_clean: object, tmp_path: Path
    ) -> None:
        """A step's file names the process that ran there, not a sibling's.

        The pk a folder carries is the way from its files back to the
        database, so it has to be that folder's own. ``01-run`` names the
        calculation hoisted into it rather than the workflow layer the
        folder was.
        """
        import yaml
        from aiida import orm

        dumped, _ = self._dump_a_run(tmp_path / "dump")

        named = {}
        for path in dumped.rglob(NODE_METADATA_FILE):
            if path.parent == dumped:
                continue
            recorded = yaml.safe_load(path.read_text())["Node data"]
            node = orm.load_node(recorded["pk"])
            assert node.uuid == recorded["uuid"]
            named[str(path.parent.relative_to(dumped))] = node.process_label

        assert named == {"01-run": "write_note", "02-write_summary": "write_summary"}
        assert (dumped / "01-run/outputs/result.txt").read_text() == "hello"

    def test_no_step_below_the_root_carries_a_workflow_node_metadata(
        self, aiida_profile_clean: object, tmp_path: Path
    ) -> None:
        """Only the root workgraph keeps its own; the layers inside do not."""
        import yaml

        dumped, _ = self._dump_a_run(tmp_path / "dump")

        below = [p for p in dumped.rglob(NODE_METADATA_FILE) if p.parent != dumped]
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

        written = list(dumped.rglob(NODE_METADATA_FILE))
        assert written
        for path in written:
            assert "Node attributes" not in yaml.safe_load(path.read_text())
            assert path.stat().st_size < 1024, path

    def test_a_step_that_ran_no_calculation_is_still_pruned(
        self, aiida_profile_clean: object, tmp_path: Path
    ) -> None:
        """The bookkeeping task's metadata file does not keep its folder.

        ``count_electrons`` leaves nothing but its own source, so the
        tree that reaches the reader holds the two steps that wrote a
        file, not three.
        """
        dumped, _ = self._dump_a_run(tmp_path / "dump")

        assert not any(p.name.endswith("count_electrons") for p in dumped.rglob("*"))
        assert sorted(p.name for p in dumped.iterdir() if p.is_dir()) == [
            "01-run",
            "02-write_summary",
        ]


class TestStepIoListing:
    """A step lists every ``Data`` input and output as one entry apiece.

    Built on :func:`tests.fixtures.make_process`/:func:`tests.fixtures.attach`
    rather than a live ``WorkGraph`` run, so a ``CalcJobNode``, a
    ``CalcFunctionNode`` (a plain pyfunction) and a ``PythonJob``-typed
    ``CalcJobNode`` can be told apart directly — the distinction
    :func:`koopmans.aiida.dumping._is_calcjob_step` keys off. Every
    scenario wraps its calculation under a trivial workflow ``root``,
    matching how a real dump is never a bare CalcJob at its top level
    (see ``ARITHMETIC_ADD``'s own probe: dumping a CalcJob with no
    wrapper skips aiida-core's ``node_outputs`` → ``outputs`` merge,
    which only ever walks a root's *descendants*).
    """

    #: A real, always-registered CalcJob — enough to resolve ``process_class``.
    ARITHMETIC_ADD = "aiida.calculations:core.arithmetic.add"
    #: A domain CalcJob's own class each real one gets; PyFunction has no
    #: entry point of its own to resolve (it runs in-process, no
    #: scheduler/computer needed), so this string is never actually
    #: loaded — only ``isinstance(node, orm.CalcFunctionNode)`` decides.
    PYFUNCTION = "aiida_pythonjob.calculations:pyfunction.pyfunction"
    #: A real, always-registered PythonJob — a genuine CalcJobNode, but
    #: excluded by :func:`koopmans.aiida.dumping._is_calcjob_step`'s
    #: ``process_class`` comparison (mutant ``m4_no_pythonjob`` drops
    #: that comparison and treats it as a real CalcJob).
    PYTHONJOB = "aiida.calculations:pythonjob.pythonjob"

    def test_a_calcjobs_dict_output_lands_beside_its_own_file(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """A CalcJob's Dict CREATE output sits next to a real file it also created."""
        import io
        import json

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach, make_process

        root = make_process("aiida.workflows:workgraph.engine", label="root")
        calc = make_process(
            self.ARITHMETIC_ADD,
            caller=root,
            link_label="calc",
            calcjob=True,
            computer=aiida_localhost,
        )
        attach(calc, "output_parameters", orm.Dict({"homo_energy": -12.353, "lumo_energy": -4.02}))  # type: ignore[no-untyped-call]
        attach(calc, "report", orm.SinglefileData(io.BytesIO(b"hello"), filename="report.txt"))

        dumped = dump_workgraph(root, tmp_path / "dump")

        outputs = dumped / "01-calc" / "outputs"
        written = json.loads((outputs / "output_parameters.json").read_text())
        assert written == {"homo_energy": -12.353, "lumo_energy": -4.02}
        assert (outputs / "report.txt").read_text() == "hello"

    def test_a_pythonjob_helper_never_lists_its_outputs(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """A ``PythonJob`` is a real ``CalcJobNode`` but still bookkeeping.

        Distinguishes it from a domain CalcJob by ``process_class``, not
        node type — an ``isinstance(node, orm.CalcJobNode)``-only check
        would wrongly write ``result.json`` here.
        """
        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach, make_process

        root = make_process("aiida.workflows:workgraph.engine", label="root")
        helper = make_process(
            self.PYTHONJOB, caller=root, link_label="helper", calcjob=True, computer=aiida_localhost
        )
        attach(helper, "result", orm.Dict({"gap": 1.5}))  # type: ignore[no-untyped-call]

        dumped = dump_workgraph(root, tmp_path / "dump")

        # Nothing is listed, and with no other content either (no
        # repository file, no input listing), the folder is pruned away
        # entirely — a domain CalcJob returning the same Dict survives
        # with its listing (test_a_calcjobs_dict_output_lands_beside_its_own_file).
        assert not any(p.name.endswith("helper") for p in dumped.rglob("*"))

    def test_a_calcjob_with_no_json_representable_output_writes_no_json(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """A CalcJob that creates only a file gets no JSON entry."""
        import io

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach, make_process

        root = make_process("aiida.workflows:workgraph.engine", label="root")
        calc = make_process(
            self.ARITHMETIC_ADD,
            caller=root,
            link_label="calc",
            calcjob=True,
            computer=aiida_localhost,
        )
        attach(calc, "report", orm.SinglefileData(io.BytesIO(b"hello"), filename="report.txt"))

        dumped = dump_workgraph(root, tmp_path / "dump")

        outputs = dumped / "01-calc" / "outputs"
        assert list(outputs.glob("*.json")) == []
        assert (outputs / "report.txt").read_text() == "hello"

    def test_a_plain_pyfunctions_dict_input_alone_leaves_no_folder(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A bare input listing does not rescue a folder from the prune.

        A step's own input listing is non-content for the same reason a
        scalar-argument helper task always was: without this, every
        pyfunction taking a Data argument would gain a folder of its own
        just to echo it back.
        """
        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import make_process

        root = make_process("aiida.workflows:workgraph.engine", label="root")
        make_process(
            self.PYFUNCTION,
            caller=root,
            link_label="helper",
            calcfunction=True,
            inputs={"parameters": orm.Dict({"x": 1})},  # type: ignore[no-untyped-call]
        )

        dumped = dump_workgraph(root, tmp_path / "dump")

        assert not any(p.name.endswith("helper") for p in dumped.rglob("*"))

    def test_a_calcjobs_data_inputs_land_in_its_input_listing(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """A CalcJob's own Dict input is directly link-labelled, unlike a pyfunction's.

        Also gives the calculation a real file output, so its folder has
        content to survive the prune besides the input listing under test
        — a bare input listing does not by itself (see
        ``test_a_plain_pyfunctions_dict_input_alone_leaves_no_folder``).
        """
        import io
        import json

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach, make_process

        root = make_process("aiida.workflows:workgraph.engine", label="root")
        calc = make_process(
            self.ARITHMETIC_ADD,
            caller=root,
            link_label="calc",
            calcjob=True,
            computer=aiida_localhost,
            inputs={"parameters": orm.Dict({"x": 1})},  # type: ignore[no-untyped-call]
        )
        attach(calc, "report", orm.SinglefileData(io.BytesIO(b"hello"), filename="report.txt"))

        dumped = dump_workgraph(root, tmp_path / "dump")

        written = json.loads((dumped / "01-calc" / "inputs" / "parameters.json").read_text())
        assert written == {"x": 1}

    def test_a_calcjob_with_only_a_dict_input_and_no_output_leaves_no_folder(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """A bare CalcJob input listing, with no other content, does not survive.

        Mirrors ``test_a_plain_pyfunctions_dict_input_alone_leaves_no_folder``
        for a genuine CalcJob rather than a pyfunction, so the rule that
        an echoed input counts as nothing produced is pinned for both
        step kinds that can ever gain one.
        """
        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import make_process

        root = make_process("aiida.workflows:workgraph.engine", label="root")
        make_process(
            self.ARITHMETIC_ADD,
            caller=root,
            link_label="calc",
            calcjob=True,
            computer=aiida_localhost,
            inputs={"parameters": orm.Dict({"x": 1})},  # type: ignore[no-untyped-call]
        )

        dumped = dump_workgraph(root, tmp_path / "dump")

        assert not any(p.name.endswith("calc") for p in dumped.rglob("*"))

    def test_the_same_calcjob_with_a_dict_output_too_survives(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """Adding a real output to the same setup makes the folder survive, with both files."""
        import json

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach, make_process

        root = make_process("aiida.workflows:workgraph.engine", label="root")
        calc = make_process(
            self.ARITHMETIC_ADD,
            caller=root,
            link_label="calc",
            calcjob=True,
            computer=aiida_localhost,
            inputs={"parameters": orm.Dict({"x": 1})},  # type: ignore[no-untyped-call]
        )
        attach(calc, "output_parameters", orm.Dict({"y": 2}))  # type: ignore[no-untyped-call]

        dumped = dump_workgraph(root, tmp_path / "dump")

        calc_dir = dumped / "01-calc"
        assert json.loads((calc_dir / "inputs" / "parameters.json").read_text()) == {"x": 1}
        assert json.loads((calc_dir / "outputs" / "output_parameters.json").read_text()) == {"y": 2}

    def test_a_plain_pyfunction_lists_neither_its_inputs_nor_its_outputs(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A pyfunction is bookkeeping and lists no value, whatever it returns.

        A real file output still lands under ``outputs/`` as aiida-core's
        own dumper always wrote it — only the values this module adds are
        withheld.
        """
        import io

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach, make_process

        root = make_process("aiida.workflows:workgraph.engine", label="root")
        helper = make_process(
            self.PYFUNCTION,
            caller=root,
            link_label="helper",
            calcfunction=True,
            inputs={"parameters": orm.Dict({"x": 1})},  # type: ignore[no-untyped-call]
        )
        attach(helper, "output_parameters", orm.Dict({"homo_energy": -12.353}))  # type: ignore[no-untyped-call]
        attach(helper, "report", orm.SinglefileData(io.BytesIO(b"hello"), filename="report.txt"))

        dumped = dump_workgraph(root, tmp_path / "dump")

        helper_dir = dumped / "01-helper"
        assert list(helper_dir.rglob("*.json")) == []
        assert (helper_dir / "outputs" / "report.txt").read_text() == "hello"

    def test_a_workflow_steps_return_lands_in_its_own_output_listing(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A workflow's RETURN surfaces a pyfunction's value its own folder never shows.

        The pyfunction ``compute_alphas`` creates no file, so its own
        folder is pruned entirely — matching ``no per-iteration helper
        folders`` — and ``alphas`` reaches the reader only via ``root``'s
        own RETURN, exactly the real ``ComputeScreeningParameters`` case.
        """
        import json

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach, make_process

        root = make_process("aiida.workflows:workgraph.engine", label="root")
        compute_alphas = make_process(
            self.PYFUNCTION, caller=root, link_label="compute_alphas", calcfunction=True
        )
        from aiida.common.links import LinkType

        alphas_dict = orm.Dict({"up": [0.7019, 0.78], "down": [0.6896]})  # type: ignore[no-untyped-call]
        alphas = attach(compute_alphas, "result", alphas_dict)
        alphas.base.links.add_incoming(root, link_type=LinkType.RETURN, link_label="alphas")

        dumped = dump_workgraph(root, tmp_path / "dump")

        assert not any(p.name.endswith("compute_alphas") for p in dumped.rglob("*"))
        written = json.loads((dumped / "outputs" / "alphas.json").read_text())
        assert written == {"up": [0.7019, 0.78], "down": [0.6896]}

    def test_a_workflow_step_never_lists_its_own_inputs(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """``root``'s own INPUT_WORK Data is never listed under ``inputs``.

        Listing it would only risk colliding with a hoisted calculation's
        own input listing (see
        :func:`koopmans.aiida.dumping._write_step_io`).
        """
        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import make_process

        root = make_process(
            "aiida.workflows:workgraph.engine",
            label="root",
            inputs={"text": orm.Str("hello")},  # type: ignore[no-untyped-call]
        )

        dumped = dump_workgraph(root, tmp_path / "dump")

        assert not (dumped / "inputs").exists()


class TestFlatEntryNames:
    """Every entry under ``inputs``/``outputs`` is named for its link label.

    The rule that distinguishes this layout from aiida-core's own: a node
    is one entry, whether it holds a value or files, so the two kinds sit
    side by side in one listing instead of a JSON file beside a directory
    tree.
    """

    ARITHMETIC_ADD = "aiida.calculations:core.arithmetic.add"

    def _calc(self, aiida_localhost: Any) -> tuple[Any, Any]:
        """Return a ``(root, calc)`` pair, the calculation wrapped as a real dump has it."""
        from tests.fixtures import make_process

        root = make_process("aiida.workflows:workgraph.engine", label="root")
        calc = make_process(
            self.ARITHMETIC_ADD,
            caller=root,
            link_label="calc",
            calcjob=True,
            computer=aiida_localhost,
        )
        return root, calc

    def test_a_single_file_repository_takes_the_link_labels_name(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """Two nodes whose files share a name stay apart, keyed by their labels.

        The shape this replaces gave both files a directory of their own
        (``output_lambdas/lambdas.npy``, ``output_bare_lambdas/lambdas.npy``),
        so the reader had to open a directory to learn which was which.
        """
        import io

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach

        root, calc = self._calc(aiida_localhost)
        attach(calc, "output_lambdas", orm.SinglefileData(io.BytesIO(b"ki"), filename="l.npy"))
        attach(calc, "output_bare_lambdas", orm.SinglefileData(io.BytesIO(b"pz"), filename="l.npy"))

        outputs = dump_workgraph(root, tmp_path / "dump") / "01-calc" / "outputs"

        assert (outputs / "output_lambdas.npy").read_bytes() == b"ki"
        assert (outputs / "output_bare_lambdas.npy").read_bytes() == b"pz"

    def test_a_repository_of_several_files_keeps_a_directory(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """Only a lone file collapses into the listing; several stay under the label."""
        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach

        root, calc = self._calc(aiida_localhost)
        tree = tmp_path / "tree"
        tree.mkdir()
        (tree / "a.dat").write_text("one")
        (tree / "b.dat").write_text("two")
        attach(calc, "output_folder", orm.FolderData(tree=str(tree)))

        outputs = dump_workgraph(root, tmp_path / "dump") / "01-calc" / "outputs"

        assert (outputs / "output_folder" / "a.dat").read_text() == "one"
        assert (outputs / "output_folder" / "b.dat").read_text() == "two"

    def test_a_namespaced_label_nests_values_into_one_file(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """``alphas__filled`` and ``alphas__empty`` become one ``alphas.json``."""
        import json

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach

        root, calc = self._calc(aiida_localhost)
        attach(calc, "alphas__filled", orm.List([0.66, 0.79]))  # type: ignore[no-untyped-call]
        attach(calc, "alphas__empty", orm.List([0.73]))  # type: ignore[no-untyped-call]

        outputs = dump_workgraph(root, tmp_path / "dump") / "01-calc" / "outputs"

        assert json.loads((outputs / "alphas.json").read_text()) == {
            "filled": [0.66, 0.79],
            "empty": [0.73],
        }

    def test_a_namespaced_label_nests_files_under_one_directory(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """``pseudos__O``'s lone file becomes ``pseudos/O.upf``, not ``pseudos/O/O.upf``."""
        import io

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach

        root, calc = self._calc(aiida_localhost)
        attach(calc, "wf__O", orm.SinglefileData(io.BytesIO(b"pp"), filename="O.upf"))

        outputs = dump_workgraph(root, tmp_path / "dump") / "01-calc" / "outputs"

        assert (outputs / "wf" / "O.upf").read_bytes() == b"pp"

    def test_a_name_the_calculations_own_files_already_use_keeps_its_directory(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """A collision with a retrieved file falls back to the ``<label>/`` form.

        The retrieved files sit loose under ``outputs`` — the calculation
        wrote them itself — so a linked node whose flattened name is
        already taken keeps a directory rather than overwriting one.
        """
        import io

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach

        root, calc = self._calc(aiida_localhost)
        tree = tmp_path / "tree"
        tree.mkdir()
        (tree / "report.txt").write_text("stdout")
        attach(calc, "retrieved", orm.FolderData(tree=str(tree)))
        attach(calc, "report", orm.SinglefileData(io.BytesIO(b"linked"), filename="r.txt"))

        outputs = dump_workgraph(root, tmp_path / "dump") / "01-calc" / "outputs"

        assert (outputs / "report.txt").read_bytes() == b"stdout"
        assert (outputs / "report" / "r.txt").read_bytes() == b"linked"

    def test_a_compound_suffix_survives_whole(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """``bundle.tar.gz`` keeps both parts of its suffix, not just ``.gz``."""
        import io

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach

        root, calc = self._calc(aiida_localhost)
        attach(calc, "tarred", orm.SinglefileData(io.BytesIO(b"t"), filename="bundle.tar.gz"))

        outputs = dump_workgraph(root, tmp_path / "dump") / "01-calc" / "outputs"

        assert (outputs / "tarred.tar.gz").read_bytes() == b"t"

    def test_a_version_numbered_name_keeps_only_its_final_suffix(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """``H_ONCV_PBE-1.0.upf`` becomes ``<label>.upf``, not ``<label>.0.upf``.

        The digit before the real suffix is part of the pseudopotential's
        own version number, not a compound suffix like ``.tar.gz``.
        """
        import io

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach

        root, calc = self._calc(aiida_localhost)
        attach(
            calc,
            "pseudos__H",
            orm.SinglefileData(io.BytesIO(b"pp"), filename="H_ONCV_PBE-1.0.upf"),
        )

        outputs = dump_workgraph(root, tmp_path / "dump") / "01-calc" / "outputs"

        assert (outputs / "pseudos" / "H.upf").read_bytes() == b"pp"

    def test_a_retrieved_file_of_the_same_name_is_not_overwritten(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """A JSON-valued link whose name collides with a retrieved file backs off.

        The calculation retrieved ``results.json`` itself; a linked
        ``results`` output with a JSON form must not clobber it, so it
        falls back to ``results/results.json`` the way a repository-backed
        entry falls back to a directory on the same collision.
        """
        import json

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach

        root, calc = self._calc(aiida_localhost)
        tree = tmp_path / "tree"
        tree.mkdir()
        (tree / "results.json").write_text("RETRIEVED-ORIGINAL")
        attach(calc, "retrieved", orm.FolderData(tree=str(tree)))
        attach(calc, "results", orm.Dict({"a": 1}))  # type: ignore[no-untyped-call]

        outputs = dump_workgraph(root, tmp_path / "dump") / "01-calc" / "outputs"

        assert (outputs / "results.json").read_text() == "RETRIEVED-ORIGINAL"
        assert json.loads((outputs / "results" / "results.json").read_text()) == {"a": 1}


class TestHoistDropsRedundantWorkflowReturn:
    """A wrapper's RETURN echo of its own direct CalcJob's output is dropped.

    Mirrors the real ``RunFinalKI`` / ``ki_final`` shape: a workflow wraps
    exactly one CalcJob and re-exports one of its Dict outputs under a
    different link label. Before this class's fix, the wrapper's own
    output listing (holding that one redundant entry) counted as a second
    child alongside the calculation's folder, so
    ``_hoist_lone_calculations`` no longer saw a lone calculation to merge
    up — leaving the same underlying node listed twice, once per label.
    """

    ARITHMETIC_ADD = "aiida.calculations:core.arithmetic.add"

    def test_the_wrapper_is_hoisted_with_one_entry_keyed_by_the_calculation(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """The wrapper folder collapses into the CalcJob's, as it did before this PR."""
        import io
        import json

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach, make_process

        # The dump root's own metadata file always survives as a sibling
        # (see TestHoistLoneCalculations.test_a_lone_top_level_calculation_keeps_its_folder),
        # so ``wrapper`` has to sit one layer below the root to be
        # eligible for the hoist under test, exactly as ``RunFinalKI``
        # sits below the real top-level workgraph.
        true_root = make_process("aiida.workflows:workgraph.engine", label="true_root")
        wrapper = make_process(
            "aiida.workflows:workgraph.engine", caller=true_root, link_label="run_final"
        )
        ki_final = make_process(
            self.ARITHMETIC_ADD,
            caller=wrapper,
            link_label="ki_final",
            calcjob=True,
            computer=aiida_localhost,
        )
        from aiida.common.links import LinkType

        parameters = attach(
            ki_final,
            "output_parameters",
            orm.Dict({"homo_energy": -12.353, "lumo_energy": -0.4034}),  # type: ignore[no-untyped-call]
        )
        attach(
            ki_final,
            "eigenvalues",
            orm.SinglefileData(io.BytesIO(b"eigenvalues"), filename="eigs.dat"),
        )
        # The wrapper re-exports the same Dict node under a different label.
        parameters.base.links.add_incoming(
            wrapper, link_type=LinkType.RETURN, link_label="parameters"
        )

        dumped = dump_workgraph(true_root, tmp_path / "dump")

        assert not any(p.name.endswith("ki_final") for p in dumped.rglob("*"))
        outputs = dumped / "01-run_final" / "outputs"
        assert sorted(p.name for p in dumped.rglob("*.json")) == ["output_parameters.json"]

        written = json.loads((outputs / "output_parameters.json").read_text())
        assert written == {"homo_energy": -12.353, "lumo_energy": -0.4034}
        assert (outputs / "eigenvalues.dat").read_text() == "eigenvalues"


class TestWorkflowReturnKeepsPyfunctionCreatedValue:
    """A wrapper's RETURN echo of a pyfunction child's output is never dropped.

    The redundant-echo rule in :func:`koopmans.aiida.dumping._direct_calculation_created_pks`
    only excludes a direct *CalcJob* child's own output — a pyfunction
    child's folder is pruned regardless of what it returns
    (:func:`koopmans.aiida.dumping._is_calcjob_step`), so its value would
    vanish entirely from the dump if the wrapper's RETURN were dropped too.
    """

    PYFUNCTION = "aiida_pythonjob.calculations:pyfunction.pyfunction"

    def test_the_pyfunctions_folder_is_pruned_but_its_value_survives_via_the_wrapper(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """The helper leaves no folder; the wrapper's own listing still shows its value."""
        import json

        from aiida import orm

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach, make_process

        wrapper = make_process("aiida.workflows:workgraph.engine", label="wrapper")
        generate_alphas = make_process(
            self.PYFUNCTION, caller=wrapper, link_label="generate_alphas", calcfunction=True
        )
        from aiida.common.links import LinkType

        alphas = attach(generate_alphas, "result", orm.Dict({"up": [0.6], "down": [0.6]}))  # type: ignore[no-untyped-call]
        # The wrapper re-exports the same Dict node under a different label.
        alphas.base.links.add_incoming(wrapper, link_type=LinkType.RETURN, link_label="alphas")

        dumped = dump_workgraph(wrapper, tmp_path / "dump")

        assert not any(p.name.endswith("generate_alphas") for p in dumped.rglob("*"))
        written = json.loads((dumped / "outputs" / "alphas.json").read_text())
        assert written == {"up": [0.6], "down": [0.6]}


class TestWorkFunctionIsBookkeeping:
    """A ``@workfunction`` (``WorkFunctionNode``) is bookkeeping, exactly like a pyfunction.

    ``resolve_pseudo_family_task`` — the codebase's only workfunction —
    is one solely because it hands back *existing* ``UpfData`` nodes
    rather than creating new ones; nothing about that makes it a step a
    reader needs its own folder for.
    """

    ARITHMETIC_ADD = "aiida.calculations:core.arithmetic.add"
    WORKFUNCTION = "aiida_koopmans.utils.pseudos.resolve_pseudo_family_task"

    def test_a_workfunctions_dict_return_gets_no_folder_of_its_own(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """No folder for the workfunction; its siblings number contiguously; its RETURN survives."""
        import io
        import json

        from aiida import orm
        from aiida.common.links import LinkType

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach, make_process

        root = make_process("aiida.workflows:workgraph.engine", label="root")
        helper = make_process(
            self.WORKFUNCTION, caller=root, link_label="resolve_family", workfunction=True
        )
        family = attach(helper, "family_name", orm.Str("SG15"))  # type: ignore[no-untyped-call]
        family.base.links.add_incoming(root, link_type=LinkType.RETURN, link_label="family_name")

        scf = make_process(
            self.ARITHMETIC_ADD,
            caller=root,
            link_label="scf",
            calcjob=True,
            computer=aiida_localhost,
        )
        attach(scf, "report", orm.SinglefileData(io.BytesIO(b"x"), filename="r.txt"))
        nscf = make_process(
            self.ARITHMETIC_ADD,
            caller=root,
            link_label="nscf",
            calcjob=True,
            computer=aiida_localhost,
        )
        attach(nscf, "report", orm.SinglefileData(io.BytesIO(b"y"), filename="s.txt"))

        dumped = dump_workgraph(root, tmp_path / "dump")

        # No folder for the workfunction.
        assert not any(p.name.endswith("resolve_family") for p in dumped.rglob("*"))
        # Its two real-calculation siblings number contiguously from one,
        # with no gap left where the pruned workfunction would have sat.
        # (the root's own output listing sits beside them)
        assert sorted(p.name for p in dumped.iterdir() if p.is_dir()) == [
            "01-scf",
            "02-nscf",
            "outputs",
        ]
        # The enclosing graph's own RETURN of the workfunction's value survives.
        assert json.loads((dumped / "outputs" / "family_name.json").read_text()) == "SG15"


class TestWrapperKeepsItsOwnFolderBesideAKeptPyfunctionReturn:
    """A wrapper's own folder survives when it keeps a genuine RETURN value.

    Repeating a value up the tree is accepted (each graph states what it
    returns): only a *direct CalcJob child's* echo is dropped. A pyfunction
    child's value is never dropped, so a wrapper re-exporting one keeps a
    real output listing of its own — and the lone CalcJob it also wraps
    stays nested rather than being hoisted, since the wrapper's folder no
    longer holds *only* that one calculation.
    """

    ARITHMETIC_ADD = "aiida.calculations:core.arithmetic.add"
    PYFUNCTION = "aiida_pythonjob.calculations:pyfunction.pyfunction"

    def test_the_calcjob_stays_nested_and_the_wrapper_keeps_its_listing(
        self, aiida_profile: Any, aiida_localhost: Any, tmp_path: Path
    ) -> None:
        """Mirrors a workflow wrapping both a CalcJob and a bookkeeping helper."""
        import io
        import json

        from aiida import orm
        from aiida.common.links import LinkType

        from koopmans.aiida.dumping import dump_workgraph
        from tests.fixtures import attach, make_process

        root = make_process("aiida.workflows:workgraph.engine", label="root")
        wrapper = make_process(
            "aiida.workflows:workgraph.engine", caller=root, link_label="run_final"
        )
        calc = make_process(
            self.ARITHMETIC_ADD,
            caller=wrapper,
            link_label="ki_final",
            calcjob=True,
            computer=aiida_localhost,
        )
        attach(calc, "eigenvalues", orm.SinglefileData(io.BytesIO(b"eig"), filename="e.dat"))
        helper = make_process(
            self.PYFUNCTION, caller=wrapper, link_label="postprocess", calcfunction=True
        )
        gap = attach(helper, "result", orm.Dict({"value": 1.5}))  # type: ignore[no-untyped-call]
        gap.base.links.add_incoming(wrapper, link_type=LinkType.RETURN, link_label="gap")

        dumped = dump_workgraph(root, tmp_path / "dump")

        # The bookkeeping helper leaves no folder of its own.
        assert not any(p.name.endswith("postprocess") for p in dumped.rglob("*"))
        # The CalcJob is NOT hoisted: the wrapper's own output listing is
        # real content, so the wrapper is not left holding only one child.
        assert (dumped / "01-run_final" / "01-ki_final").is_dir()
        written = json.loads((dumped / "01-run_final" / "outputs" / "gap.json").read_text())
        assert written == {"value": 1.5}
