"""Unit tests for the dump folder-name simplification."""

from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent
from typing import Any, ClassVar

import pytest

from koopmans.aiida.dumping import (
    _hoist_lone_calculations,
    _prune_outputless_step_folders,
    _renumber_step_folders,
    _simplify_folder_names,
    _strip_process_label_suffixes,
    _tidy_dumped_tree,
)


def _make_tree(root: Path, layout: Sequence[str]) -> None:
    """Create the listed paths under ``root``.

    An entry ending in "/" becomes an empty folder; any other entry
    becomes an empty file, with its parent folders created for it.

    :param root: Folder the entries are created under.
    :param layout: Slash-separated paths relative to ``root``.
    """
    for entry in layout:
        target = root / entry
        if entry.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()


def _folders(root: Path) -> list[str]:
    """Return every folder under ``root``, relative and sorted."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir())


def _render(root: Path) -> str:
    """Return the tree under ``root`` drawn the way the tutorials quote it."""
    lines = [root.name]

    def draw(folder: Path, prefix: str) -> None:
        """Append a line per entry of ``folder``, then descend into each."""
        entries = sorted(folder.iterdir(), key=lambda p: p.name)
        for index, entry in enumerate(entries):
            last = index == len(entries) - 1
            lines.append(prefix + ("└── " if last else "├── ") + entry.name)
            if entry.is_dir():
                draw(entry, prefix + ("    " if last else "│   "))

    draw(root, "")
    return "\n".join(lines)


def _calculation(name: str) -> list[str]:
    """Return the layout entries of a calculation folder called ``name``."""
    return [f"{name}/inputs/aiida.cpi", f"{name}/outputs/aiida.cpo"]


def _bookkeeping(name: str) -> list[str]:
    """Return the layout entries of a python step that recorded no outputs.

    AiiDA dumps such a step its own source, and a serialized copy of
    whichever inputs it was handed.
    """
    return [f"{name}/inputs/source_file"]


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


class TestPruneOutputlessStepFolders:
    """A step that recorded no outputs leaves no folder behind."""

    def test_a_bookkeeping_step_goes_and_a_calculation_stays(self, tmp_path: Path) -> None:
        """The source and inputs AiiDA dumps alongside are no reason to keep it."""
        _make_tree(
            tmp_path,
            [*_bookkeeping("01-count_electrons_task"), *_calculation("02-dft_init")],
        )

        _prune_outputless_step_folders(tmp_path)

        assert _folders(tmp_path) == ["02-dft_init", "02-dft_init/inputs", "02-dft_init/outputs"]

    def test_an_empty_step_folder_goes(self, tmp_path: Path) -> None:
        """A step AiiDA dumped nothing at all for is the same case."""
        _make_tree(tmp_path, ["01-resolve_pseudo_family_task/"])

        _prune_outputless_step_folders(tmp_path)

        assert _folders(tmp_path) == []

    def test_a_step_holding_only_outputless_steps_goes_too(self, tmp_path: Path) -> None:
        """Having no outputs propagates outwards, however deeply nested."""
        _make_tree(tmp_path, ["01-outer/01-middle/", *_bookkeeping("01-outer/02-inner")])

        _prune_outputless_step_folders(tmp_path)

        assert _folders(tmp_path) == []

    def test_a_python_step_that_wrote_files_survives(self, tmp_path: Path) -> None:
        """Outputs, not the kind of process, decide.

        ``prepare_kcw_wannier_files`` is a python step like the pruned
        ones, but it writes the merged Wannier files the next step reads.
        """
        _make_tree(tmp_path, _calculation("01-prepare_kcw_wannier_files"))

        _prune_outputless_step_folders(tmp_path)

        assert "01-prepare_kcw_wannier_files" in _folders(tmp_path)

    def test_an_unnumbered_folder_is_never_pruned(self, tmp_path: Path) -> None:
        """Only "<NN>-<label>" step folders are candidates.

        A calculation's own ``inputs`` is not a step that recorded no
        outputs, and has to survive the pass whatever it holds.
        """
        _make_tree(tmp_path, _calculation("01-dft_init"))

        _prune_outputless_step_folders(tmp_path)

        assert _folders(tmp_path) == ["01-dft_init", "01-dft_init/inputs", "01-dft_init/outputs"]


class TestRenumberStepFolders:
    """Pruning leaves holes in the numbering; renumbering closes them."""

    def test_holes_close_and_the_order_survives(self, tmp_path: Path) -> None:
        """The numbers are dump-side prose, the order they encode is not."""
        _make_tree(tmp_path, ["03-dft_init_nspin1/a", "06-dft_init_nspin2/a", "08-RunFinalKI/a"])

        _renumber_step_folders(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "01-dft_init_nspin1",
            "02-dft_init_nspin2",
            "03-RunFinalKI",
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

    OZONE_DUMP: ClassVar[list[str]] = [
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
        *_bookkeeping(f"{ORBITALS}/01-compute_alpha_orb_1/02-compute_alpha_from_dscf"),
        *_calculation(f"{ORBITALS}/02-compute_alpha_orb_10/01-dft_n_plus_1_dummy-KcpCalculation"),
        *_calculation(f"{ORBITALS}/02-compute_alpha_orb_10/02-pz_print-KcpCalculation"),
        *_calculation(f"{ORBITALS}/02-compute_alpha_orb_10/03-dft_n_plus_1-KcpCalculation"),
        *_bookkeeping(f"{ORBITALS}/02-compute_alpha_orb_10/04-compute_alpha_from_dscf"),
        *_bookkeeping(f"{ORBITALS}/11-expand_alphas_by_group"),
        *_bookkeeping(f"{SCREENING}/05-max_alpha_error"),
        *_calculation("08-RunFinalKI/01-ki_final-KcpCalculation"),
    ]

    TIDIED = """\
        ozone
        ├── 01-dft_init_nspin1
        │   ├── inputs
        │   │   └── aiida.cpi
        │   └── outputs
        │       └── aiida.cpo
        ├── 02-dft_init_nspin2_dummy
        │   ├── inputs
        │   │   └── aiida.cpi
        │   └── outputs
        │       └── aiida.cpo
        ├── 03-dft_init_nspin2
        │   ├── inputs
        │   │   └── aiida.cpi
        │   └── outputs
        │       └── aiida.cpo
        ├── 04-ComputeScreeningParameters
        │   └── 01-ScreeningIteration
        │       ├── 01-ki_trial
        │       │   ├── inputs
        │       │   │   └── aiida.cpi
        │       │   └── outputs
        │       │       └── aiida.cpo
        │       └── 02-compute_orbital_screening_parameters
        │           ├── 01-compute_alpha_orb_1
        │           │   ├── inputs
        │           │   │   └── aiida.cpi
        │           │   └── outputs
        │           │       └── aiida.cpo
        │           └── 02-compute_alpha_orb_10
        │               ├── 01-dft_n_plus_1_dummy
        │               │   ├── inputs
        │               │   │   └── aiida.cpi
        │               │   └── outputs
        │               │       └── aiida.cpo
        │               ├── 02-pz_print
        │               │   ├── inputs
        │               │   │   └── aiida.cpi
        │               │   └── outputs
        │               │       └── aiida.cpo
        │               └── 03-dft_n_plus_1
        │                   ├── inputs
        │                   │   └── aiida.cpi
        │                   └── outputs
        │                       └── aiida.cpo
        └── 05-RunFinalKI
            ├── inputs
            │   └── aiida.cpi
            └── outputs
                └── aiida.cpo"""

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
