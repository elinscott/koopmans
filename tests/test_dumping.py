"""Unit tests for the dump folder-name simplification."""

from pathlib import Path

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
