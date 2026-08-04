"""Unit tests for the dump folder-name simplification."""

from pathlib import Path
from typing import Any

import pytest

from koopmans.aiida.dumping import _simplify_folder_names


@pytest.mark.parametrize(
    ("dumped", "simplified"),
    [
        # pyfunction: the process label equals the link label, so the dump
        # never appends it — only the pk goes
        ("01-resolve_pseudo_family_task-4711", "01-resolve_pseudo_family_task"),
        # sub-workgraph: the WorkGraph<...> process label always repeats the
        # link label, so it goes along with the pk
        ("03-dft_init_nspin1-WorkGraph<dft_init_nspin1>-4712", "03-dft_init_nspin1"),
        # CalcJob: the class name says which code ran, so it stays
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
