"""Unit tests for the dump folder-name simplification."""

from pathlib import Path

from koopmans.aiida.dumping import _simplify_folder_names


def _mkdirs(root: Path, names: list[str]) -> None:
    for name in names:
        (root / name).mkdir(parents=True)


def test_simplify_folder_names(tmp_path: Path) -> None:
    """Strip the pk and the WorkGraph process label; keep other suffixes.

    The names mirror what AiiDA's dump produces for a workgraph: every
    folder ends with the pk, sub-workgraph folders additionally carry
    the ``WorkGraph<graph_name>`` process label, CalcJob folders carry
    their class name, and pyfunction folders carry no label at all.
    """
    _mkdirs(
        tmp_path,
        [
            "01-resolve_pseudo_family_task-4711",
            "03-dft_init_nspin1-WorkGraph<dft_init_nspin1>-4712/01-dft_init-KcpCalculation-4713",
            "04-compute_orbital_screening_parameters-WorkGraph<compute_orbital_screening_parameters>-4714"
            "/01-compute_alpha_orb_1-WorkGraph<compute_alpha_orb_1>-4715",
        ],
    )

    _simplify_folder_names(tmp_path)

    all_dirs = sorted(str(d.relative_to(tmp_path)) for d in tmp_path.rglob("*") if d.is_dir())
    assert all_dirs == [
        "01-resolve_pseudo_family_task",
        "03-dft_init_nspin1",
        "03-dft_init_nspin1/01-dft_init-KcpCalculation",
        "04-compute_orbital_screening_parameters",
        "04-compute_orbital_screening_parameters/01-compute_alpha_orb_1",
    ]


def test_simplify_folder_names_keeps_taken_names(tmp_path: Path) -> None:
    """A folder is left untouched if its simplified name already exists."""
    _mkdirs(tmp_path, ["02-scf", "02-scf-WorkGraph<scf>-99"])

    _simplify_folder_names(tmp_path)

    all_dirs = sorted(d.name for d in tmp_path.iterdir())
    assert all_dirs == ["02-scf", "02-scf-WorkGraph<scf>-99"]
