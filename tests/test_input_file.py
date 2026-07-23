"""Tests for input file parsing."""

import json
from pathlib import Path

import pytest

from koopmans.input_file import (
    INPUT_FILE_FORMAT_VERSION,
    KoopmansInput,
    migrate_input_dict,
    read_input_file,
)
from koopmans.input_file.workflow import Task


class TestInputFileParsing:
    """Test input file parsing."""

    def test_parse_si_tutorial(self, tutorials_dir: Path) -> None:
        """Test that the silicon tutorial input file parses successfully."""
        input_file = tutorials_dir / "si.json"
        assert input_file.exists(), f"Tutorial file not found: {input_file}"

        koopmans_input = read_input_file(input_file)

        assert isinstance(koopmans_input, KoopmansInput)
        assert koopmans_input.workflow.task == Task.WANNIERIZE
        assert koopmans_input.workflow.pseudo_library == "PseudoDojo/0.4/LDA/SR/standard/upf"
        assert koopmans_input.calculator_parameters.ecutwfc == 60.0
        assert koopmans_input.kpoints.grid == (2, 2, 2)

    def test_parse_si_tutorial_via_classmethod(self, tutorials_dir: Path) -> None:
        """Test parsing via the KoopmansInput.from_file classmethod."""
        input_file = tutorials_dir / "si.json"

        koopmans_input = KoopmansInput.from_file(input_file)

        assert isinstance(koopmans_input, KoopmansInput)
        assert koopmans_input.workflow.task == Task.WANNIERIZE

    def test_invalid_file_extension(self, tmp_path: Path) -> None:
        """Test that an invalid file extension raises an error."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("{}")

        with pytest.raises(ValueError, match="Unrecognized file type"):
            read_input_file(invalid_file)


class TestInputFileVersioning:
    """Test input file format versioning."""

    def test_missing_version_treated_as_version_1(self, tutorials_dir: Path) -> None:
        """Test that a file without a `version` key parses as the current version."""
        koopmans_input = read_input_file(tutorials_dir / "si.json")
        assert koopmans_input.version == INPUT_FILE_FORMAT_VERSION

    def test_explicit_current_version(self, tutorials_dir: Path, tmp_path: Path) -> None:
        """Test that a file with an explicit current `version` parses."""
        input_dict = json.loads((tutorials_dir / "si.json").read_text())
        input_dict["version"] = INPUT_FILE_FORMAT_VERSION
        input_file = tmp_path / "si.json"
        input_file.write_text(json.dumps(input_dict))

        koopmans_input = read_input_file(input_file)
        assert koopmans_input.version == INPUT_FILE_FORMAT_VERSION

    def test_future_version_raises(self, tutorials_dir: Path, tmp_path: Path) -> None:
        """Test that a file from a newer format version raises a clear error."""
        input_dict = json.loads((tutorials_dir / "si.json").read_text())
        input_dict["version"] = INPUT_FILE_FORMAT_VERSION + 1
        input_file = tmp_path / "si.json"
        input_file.write_text(json.dumps(input_dict))

        with pytest.raises(ValueError, match="Please upgrade `koopmans`"):
            read_input_file(input_file)

    @pytest.mark.parametrize("version", ["banana", 0, -1, 1.5, True])
    def test_invalid_version_raises(self, version: object) -> None:
        """Test that a non-positive-integer `version` raises a clear error."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            migrate_input_dict({"version": version})


class TestSchemaValidation:
    """Validation checks that used to fail late (or not at all) at conversion time."""

    def test_celldms_without_celldm1_rejected(self) -> None:
        """``celldms`` without celldm(1) has no length scale and must be rejected."""
        from koopmans.input_file.cell_parameters import CellParametersViaIbrav

        with pytest.raises(ValueError, match=r"celldm\(1\)"):
            CellParametersViaIbrav.model_validate({"ibrav": 2, "celldms": {2: 0.5}})

    def test_non_integer_nbnd_rejected(self) -> None:
        """A fractional band count must be rejected, not silently truncated."""
        from koopmans.input_file import CalculatorParametersInput

        with pytest.raises(ValueError, match="nbnd"):
            CalculatorParametersInput.model_validate({"nbnd": 10.7})

        assert CalculatorParametersInput.model_validate({"nbnd": 10.0}).nbnd == 10


def _parallelization_input(*, parallelization: object | None = None) -> dict[str, object]:
    """Return a minimal silicon input dict for parallelization-block tests."""
    d: dict[str, object] = {
        "workflow": {"task": "dft_bands", "pseudo_library": "X"},
        "atoms": {
            "cell_parameters": {"periodic": True, "ibrav": 2, "celldms": {"1": 10.26}},
            "atomic_positions": {
                "units": "crystal",
                "positions": [["Si", 0, 0, 0], ["Si", 0.25, 0.25, 0.25]],
            },
        },
        "kpoints": {"grid": [2, 2, 2]},
        "calculator_parameters": {"ecutwfc": 20.0},
    }
    if parallelization is not None:
        d["parallelization"] = parallelization
    return d


class TestParallelizationSchema:
    """The top-level per-code ``parallelization`` block."""

    def test_per_code_entries_parse(self) -> None:
        """Each configured code carries its own ntasks / npool."""
        inp = KoopmansInput.model_validate(
            _parallelization_input(
                parallelization={"pw": {"npool": 2, "ntasks": 8}, "kcw": {"npool": 4}}
            )
        )
        pw = inp.parallelization.pw
        kcw = inp.parallelization.kcw
        assert pw is not None and kcw is not None
        assert (pw.npool, pw.ntasks) == (2, 8)
        assert kcw.npool == 4

    def test_as_mapping_drops_unset_fields_and_codes(self) -> None:
        """as_mapping keeps only configured codes and their set fields."""
        inp = KoopmansInput.model_validate(
            _parallelization_input(parallelization={"pw": {"npool": 2}, "kcw": {"ntasks": 8}})
        )
        assert inp.parallelization.as_mapping() == {"pw": {"npool": 2}, "kcw": {"ntasks": 8}}

    def test_no_config_leaves_codes_unset(self) -> None:
        """Without a block, every code entry stays ``None`` and the mapping is empty."""
        inp = KoopmansInput.model_validate(_parallelization_input())
        assert inp.parallelization.pw is None
        assert inp.parallelization.as_mapping() == {}

    def test_npool_rejected_for_wannier90(self) -> None:
        """wannier90 has no k-point pools, so npool must be rejected."""
        with pytest.raises(ValueError, match=r"npool.*not valid for wannier90"):
            KoopmansInput.model_validate(
                _parallelization_input(parallelization={"wannier90": {"npool": 2}})
            )

    def test_wannier90_ntasks_allowed(self) -> None:
        """wannier90 accepts ntasks (only npool is rejected)."""
        inp = KoopmansInput.model_validate(
            _parallelization_input(parallelization={"wannier90": {"ntasks": 4}})
        )
        wannier90 = inp.parallelization.wannier90
        assert wannier90 is not None
        assert wannier90.ntasks == 4

    def test_unknown_code_rejected(self) -> None:
        """An unrecognised code name is not a valid parallelization key."""
        with pytest.raises(ValueError):
            KoopmansInput.model_validate(
                _parallelization_input(parallelization={"foo": {"npool": 2}})
            )

    @pytest.mark.parametrize("field", ["ntasks", "npool"])
    def test_positive_ints_only(self, field: str) -> None:
        """Both fields reject zero and negative values."""
        with pytest.raises(ValueError):
            KoopmansInput.model_validate(_parallelization_input(parallelization={"pw": {field: 0}}))
