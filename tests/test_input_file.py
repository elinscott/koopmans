"""Tests for input file parsing."""

from pathlib import Path

import pytest

from koopmans.input_file import KoopmansInput, read_input_file
from koopmans.input_file.workflow import Task

TUTORIALS_DIR = Path(__file__).parent.parent / "docs" / "source" / "tutorials"


class TestInputFileParsing:
    """Test input file parsing."""

    def test_parse_si_tutorial(self) -> None:
        """Test that the silicon tutorial input file parses successfully."""
        input_file = TUTORIALS_DIR / "si.json"
        assert input_file.exists(), f"Tutorial file not found: {input_file}"

        koopmans_input = read_input_file(input_file)

        assert isinstance(koopmans_input, KoopmansInput)
        assert koopmans_input.workflow.task == Task.WANNIERIZE
        assert koopmans_input.workflow.pseudo_library == "PseudoDojo/0.4/LDA/SR/standard/upf"
        assert koopmans_input.calculator_parameters.ecutwfc == 60.0
        assert koopmans_input.kpoints.grid == (2, 2, 2)

    def test_parse_si_tutorial_via_classmethod(self) -> None:
        """Test parsing via the KoopmansInput.from_file classmethod."""
        input_file = TUTORIALS_DIR / "si.json"

        koopmans_input = KoopmansInput.from_file(input_file)

        assert isinstance(koopmans_input, KoopmansInput)
        assert koopmans_input.workflow.task == Task.WANNIERIZE

    def test_invalid_file_extension(self, tmp_path: Path) -> None:
        """Test that an invalid file extension raises an error."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("{}")

        with pytest.raises(ValueError, match="Unrecognized file type"):
            read_input_file(invalid_file)
