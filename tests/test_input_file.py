"""Tests for input file parsing."""

import json
from pathlib import Path

import pytest
import yaml

from koopmans.input_file import (
    INPUT_FILE_FORMAT_VERSION,
    KoopmansInput,
    migrate_input_dict,
    read_input_file,
)
from koopmans.input_file.workflow import Task

# The silicon tutorial input file, relative to the tutorials directory.
SI_INPUT = "band_structures/silicon_finite_differences/si.yaml"


class TestInputFileParsing:
    """Test input file parsing."""

    def test_parse_si_tutorial(self, tutorials_dir: Path) -> None:
        """Test that the silicon tutorial input file parses successfully."""
        input_file = tutorials_dir / SI_INPUT
        assert input_file.exists(), f"Tutorial file not found: {input_file}"

        koopmans_input = read_input_file(input_file)

        assert isinstance(koopmans_input, KoopmansInput)
        assert koopmans_input.workflow.task == Task.WANNIERIZE
        assert koopmans_input.workflow.pseudo_library == "PseudoDojo/0.4/LDA/SR/standard/upf"
        assert koopmans_input.calculator_parameters.ecutwfc == 60.0
        assert koopmans_input.kpoints.grid == (2, 2, 2)

    def test_parse_si_tutorial_via_classmethod(self, tutorials_dir: Path) -> None:
        """Test parsing via the KoopmansInput.from_file classmethod."""
        input_file = tutorials_dir / SI_INPUT

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
        koopmans_input = read_input_file(tutorials_dir / SI_INPUT)
        assert koopmans_input.version == INPUT_FILE_FORMAT_VERSION

    def test_explicit_current_version(self, tutorials_dir: Path, tmp_path: Path) -> None:
        """Test that a file with an explicit current `version` parses."""
        input_dict = yaml.safe_load((tutorials_dir / SI_INPUT).read_text())
        input_dict["version"] = INPUT_FILE_FORMAT_VERSION
        input_file = tmp_path / "si.json"
        input_file.write_text(json.dumps(input_dict))

        koopmans_input = read_input_file(input_file)
        assert koopmans_input.version == INPUT_FILE_FORMAT_VERSION

    def test_future_version_raises(self, tutorials_dir: Path, tmp_path: Path) -> None:
        """Test that a file from a newer format version raises a clear error."""
        input_dict = yaml.safe_load((tutorials_dir / SI_INPUT).read_text())
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


def _minimal_si_input() -> dict[str, object]:
    """Return a minimal, valid silicon ``KoopmansInput`` dict."""
    return {
        "workflow": {"task": "singlepoint", "pseudo_library": "X"},
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


_REMOVED_KEYWORDS = [
    ("workflow", "converge"),
    ("workflow", "automated_wannierization"),
    ("ml", "train_on_the_fly"),
    ("ml", "alphas_from_file"),
    ("calculator_parameters.wannier90.up", "auto_projections"),
    ("calculator_parameters.wannier90.down", "auto_projections"),
]


def _set_keyword(d: dict[str, object], section: str, keyword: str, value: object = True) -> None:
    """Set ``keyword`` to ``value`` under the (dotted) ``section`` path of an input dict."""
    target = d
    for part in section.split("."):
        target = target.setdefault(part, {})  # type: ignore[assignment]
    target[keyword] = value


class TestRemovedKeywordsRejected:
    """Retired keywords must fail loudly at parse, not be silently ignored."""

    @pytest.mark.parametrize(("section", "keyword"), _REMOVED_KEYWORDS)
    def test_removed_keyword_rejected_by_model(self, section: str, keyword: str) -> None:
        """A retired keyword is an unknown field under ``extra='forbid'``."""
        from pydantic import ValidationError

        d = _minimal_si_input()
        _set_keyword(d, section, keyword)

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            KoopmansInput.model_validate(d)

    @pytest.mark.parametrize(("section", "keyword"), _REMOVED_KEYWORDS)
    def test_removed_keyword_gives_friendly_message(
        self, section: str, keyword: str, tmp_path: Path
    ) -> None:
        """``read_input_file`` reports the retired keyword as an invalid keyword."""
        d = _minimal_si_input()
        _set_keyword(d, section, keyword)
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(d))

        with pytest.raises(ValueError, match=rf"{section}\.{keyword}.*is not a valid keyword"):
            read_input_file(input_file)


_CUTOFF_KEYS = [
    ("calculator_parameters", "ecutwfc"),
]


class TestCutoffsMustBePositive:
    """A cutoff of zero or less is rejected at parse."""

    @pytest.mark.parametrize(("section", "keyword"), _CUTOFF_KEYS)
    @pytest.mark.parametrize("value", [0.0, -45.0])
    def test_a_non_positive_cutoff_is_rejected(
        self, section: str, keyword: str, value: float, tmp_path: Path
    ) -> None:
        """The message names the key the input file used, not a ratio arithmetic failure.

        Zero and a negative value fail differently downstream — one divides by
        zero building the off-ratio warning, the other reports a ratio of -36 —
        so both are asserted rather than one standing in for the pair.
        """
        d = _minimal_si_input()
        _set_keyword(d, section, keyword, value)
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(d))

        with pytest.raises(ValueError) as excinfo:
            read_input_file(input_file)

        message = str(excinfo.value)
        assert f"`{section}.{keyword}`" in message
        assert "must be greater than 0" in message


class TestCalculateBandsRemoved:
    """A band structure is asked for by ``kpoints.path``, not by a switch."""

    def test_the_removed_keyword_names_the_path_instead(self, tmp_path: Path) -> None:
        """The message points the reader at ``kpoints.path``, not at ``extra_forbidden``."""
        d = _minimal_si_input()
        _set_keyword(d, "workflow", "calculate_bands")
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(d))

        with pytest.raises(ValueError) as excinfo:
            read_input_file(input_file)

        message = str(excinfo.value)
        assert "`workflow.calculate_bands` no longer exists" in message
        assert "`kpoints.path`" in message
        assert "is not a valid keyword" not in message

    def test_the_removed_keyword_is_rejected_however_it_is_set(self) -> None:
        """``false`` is refused too: the keyword no longer states anything."""
        from pydantic import ValidationError

        d = _minimal_si_input()
        _set_keyword(d, "workflow", "calculate_bands", False)

        with pytest.raises(ValidationError, match="no longer exists"):
            KoopmansInput.model_validate(d)


class TestPeriodicIsOnePerCellVector:
    """``periodic`` is canonical after validation, whichever way it was written."""

    @pytest.mark.parametrize(
        ("written", "expected"),
        [
            (True, (True, True, True)),
            (False, (False, False, False)),
            ([True, True, False], (True, True, False)),
        ],
    )
    def test_a_bool_states_the_same_of_all_three(
        self, written: object, expected: tuple[bool, bool, bool]
    ) -> None:
        """A single bool expands; an explicit triple passes through."""
        d = _minimal_si_input()
        _set_keyword(d, "atoms.cell_parameters", "periodic", written)

        inp = KoopmansInput.model_validate(d)

        assert inp.atoms.cell_parameters.periodic == expected

    def test_a_wrong_length_is_rejected(self) -> None:
        """Two entries name no third cell vector."""
        from pydantic import ValidationError

        d = _minimal_si_input()
        _set_keyword(d, "atoms.cell_parameters", "periodic", [True, False])

        with pytest.raises(ValidationError):
            KoopmansInput.model_validate(d)

    def test_the_canonical_form_round_trips(self) -> None:
        """``model_dump`` emits the triple, and re-validating it changes nothing.

        koopmans re-validates dumped inputs, so a normalization only the raw
        file survived would drift on the second pass.
        """
        inp = KoopmansInput.model_validate(_minimal_si_input())

        dumped = inp.model_dump()
        periodic = dumped["atoms"]["cell_parameters"]["periodic"]

        assert periodic == (True, True, True)
        assert KoopmansInput.model_validate(dumped).atoms.cell_parameters.periodic == periodic


def _si_input_with(calculator_parameters: dict[str, object]) -> dict[str, object]:
    """Return the minimal silicon input, its ``calculator_parameters`` replaced."""
    d = _minimal_si_input()
    d["calculator_parameters"] = calculator_parameters
    return d


_REMOVED_PER_CALCULATOR_CUTOFFS = [
    ("pw", "ecutwfc"),
    ("pw", "ecutrho"),
    ("kcp", "ecutwfc"),
    ("kcp", "ecutrho"),
]


class TestPerCalculatorCutoffsRemoved:
    """pw.x and kcp.x always share one grid, stated once via ``ecutwfc``."""

    @pytest.mark.parametrize(
        ("calculator", "keyword"),
        _REMOVED_PER_CALCULATOR_CUTOFFS,
        ids=[f"{calc}.{kw}" for calc, kw in _REMOVED_PER_CALCULATOR_CUTOFFS],
    )
    def test_a_removed_key_names_its_replacement(
        self, calculator: str, keyword: str, tmp_path: Path
    ) -> None:
        """Each retired spelling points the reader at ``calculator_parameters.ecutwfc``."""
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(_si_input_with({calculator: {"system": {keyword: 45.0}}})))

        with pytest.raises(ValueError) as excinfo:
            read_input_file(input_file)

        message = str(excinfo.value)
        assert f"`calculator_parameters.{calculator}.system.{keyword}`" in message
        assert "`calculator_parameters.ecutwfc`" in message

    def test_a_single_ecutwfc_reaches_both_pw_and_kcp(self, tmp_path: Path) -> None:
        """One stated cutoff derives both codes' grids, at the norm-conserving ratio."""
        from koopmans.aiida.conversion import input_to_pw_parameters
        from koopmans.aiida.workflows.dscf import kcp_dscf_inputs

        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(_si_input_with({"ecutwfc": 45.0, "nbnd": 8})))
        inp = read_input_file(input_file)

        pw_system = input_to_pw_parameters(inp)["SYSTEM"]
        kcp = kcp_dscf_inputs(inp)
        assert (pw_system["ecutwfc"], pw_system["ecutrho"]) == pytest.approx((45.0, 180.0))
        assert (kcp["ecutwfc"], kcp["ecutrho"]) == pytest.approx((45.0, 180.0))


class TestKcpMagnetizationRemoved:
    """The magnetization is stated once, via ``calculator_parameters``."""

    def test_the_kcp_magnetization_names_the_shared_field(self, tmp_path: Path) -> None:
        """The retired ``kcp`` spelling points at ``calculator_parameters``' own field."""
        input_file = tmp_path / "input.json"
        input_file.write_text(
            json.dumps(_si_input_with({"kcp": {"system": {"tot_magnetization": 2.0}}}))
        )

        with pytest.raises(ValueError) as excinfo:
            read_input_file(input_file)

        message = str(excinfo.value)
        assert "`calculator_parameters.kcp.system.tot_magnetization`" in message
        assert "`calculator_parameters.tot_magnetization`" in message

    def test_the_shared_magnetization_still_reaches_kcp(self, tmp_path: Path) -> None:
        """The surviving spelling is what the kcp.x builders read."""
        from koopmans.aiida.workflows.dscf import kcp_dscf_inputs

        input_file = tmp_path / "input.json"
        input_file.write_text(
            json.dumps(_si_input_with({"ecutwfc": 45.0, "nbnd": 8, "tot_magnetization": 2.0}))
        )

        assert kcp_dscf_inputs(read_input_file(input_file))["tot_magnetization"] == 2


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
        """Each configured code carries its own ntasks / npool / pd."""
        inp = KoopmansInput.model_validate(
            _parallelization_input(
                parallelization={"pw": {"npool": 2, "ntasks": 8, "pd": True}, "kcw": {"npool": 4}}
            )
        )
        pw = inp.parallelization.pw
        kcw = inp.parallelization.kcw
        assert pw is not None and kcw is not None
        assert (pw.npool, pw.ntasks, pw.pd) == (2, 8, True)
        assert kcw.npool == 4

    def test_as_mapping_drops_unset_fields_and_codes(self) -> None:
        """as_mapping keeps only configured codes and their set fields."""
        inp = KoopmansInput.model_validate(
            _parallelization_input(
                parallelization={"pw": {"npool": 2, "pd": True}, "kcw": {"ntasks": 8}}
            )
        )
        assert inp.parallelization.as_mapping() == {
            "pw": {"npool": 2, "pd": True},
            "kcw": {"ntasks": 8},
        }

    def test_omp_parses_and_maps_through(self) -> None:
        """The omp thread count round-trips and reaches as_mapping."""
        inp = KoopmansInput.model_validate(
            _parallelization_input(parallelization={"pw": {"ntasks": 8, "omp": 4}})
        )
        pw = inp.parallelization.pw
        assert pw is not None and pw.omp == 4
        assert inp.parallelization.as_mapping() == {"pw": {"ntasks": 8, "omp": 4}}

    @pytest.mark.parametrize("code", ["kcp", "wann2kcp", "wannier90"])
    def test_omp_allowed_for_every_code(self, code: str) -> None:
        """The omp knob has no support matrix — even codes that reject npool/pd accept it."""
        inp = KoopmansInput.model_validate(
            _parallelization_input(parallelization={code: {"omp": 2}})
        )
        cfg = getattr(inp.parallelization, code)
        assert cfg is not None and cfg.omp == 2
        assert inp.parallelization.as_mapping() == {code: {"omp": 2}}

    def test_omp_rejects_zero(self) -> None:
        """The omp field is a positive integer."""
        with pytest.raises(ValueError):
            KoopmansInput.model_validate(_parallelization_input(parallelization={"pw": {"omp": 0}}))

    def test_no_config_leaves_codes_unset(self) -> None:
        """Without a block, every code entry stays ``None`` and the mapping is empty."""
        inp = KoopmansInput.model_validate(_parallelization_input())
        assert inp.parallelization.pw is None
        assert inp.parallelization.as_mapping() == {}

    @pytest.mark.parametrize("code", ["kcp", "wann2kcp", "wannier90"])
    def test_npool_rejected_for_non_pool_codes(self, code: str) -> None:
        """Only pw, ph, projwfc, pw2wannier90, and kcw parallelize over k-point pools."""
        with pytest.raises(ValueError, match=r"'npool' is not valid"):
            KoopmansInput.model_validate(
                _parallelization_input(parallelization={code: {"npool": 2}})
            )

    @pytest.mark.parametrize("code", ["kcp", "wann2kcp", "wannier90"])
    def test_pd_rejected_for_non_pd_codes(self, code: str) -> None:
        """Only pw, ph, projwfc, pw2wannier90, and kcw support pencil decomposition."""
        with pytest.raises(ValueError, match=r"'pd' \(pencil decomposition\) is not valid"):
            KoopmansInput.model_validate(
                _parallelization_input(parallelization={code: {"pd": True}})
            )

    @pytest.mark.parametrize("code", ["ph", "pw2wannier90"])
    def test_npool_and_pd_allowed_for_ph_and_pw2wannier90(self, code: str) -> None:
        """The ph and pw2wannier90 codes accept both flags (verified against QE source)."""
        inp = KoopmansInput.model_validate(
            _parallelization_input(parallelization={code: {"npool": 2, "pd": True}})
        )
        cfg = getattr(inp.parallelization, code)
        assert cfg is not None
        assert (cfg.npool, cfg.pd) == (2, True)

    def test_ntasks_allowed_for_any_code(self) -> None:
        """The ntasks (MPI ranks) field is universal — even wannier90 accepts it."""
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
        """Both integer fields reject zero and negative values."""
        with pytest.raises(ValueError):
            KoopmansInput.model_validate(_parallelization_input(parallelization={"pw": {field: 0}}))


class TestKpointsOffset:
    """The offset is a per-axis fraction of a grid step, as ``KpointsData`` reads it."""

    @pytest.mark.parametrize("offset", [(0.0, 0.0, 0.0), (0.5, 0.0, 0.5)])
    def test_the_two_expressible_shifts_accepted(self, offset: tuple[float, ...]) -> None:
        """No shift and a half-step shift both survive validation unchanged."""
        from koopmans.input_file import GridKpointsInput

        assert GridKpointsInput(grid=(2, 2, 2), offset=offset).offset == offset

    def test_a_whole_grid_step_rejected_with_the_syntax_explained(self) -> None:
        """A 1 is where a reader carries QE's flag convention over by mistake.

        Arithmetically it is the same as 0, so accepting it would hand back
        an unshifted mesh to someone who asked for a shifted one. The error
        has to say that and name the value that does work.
        """
        from koopmans.input_file import GridKpointsInput

        with pytest.raises(ValueError, match=r"whole grid step.*same as 0"):
            GridKpointsInput(grid=(2, 2, 2), offset=(1, 1, 1))
        with pytest.raises(ValueError, match=r"0\.5"):
            GridKpointsInput(grid=(2, 2, 2), offset=(1, 1, 1))

    @pytest.mark.parametrize("offset", [(0.25, 0.0, 0.0), (2.0, 0.0, 0.0), (-0.5, 0.0, 0.0)])
    def test_any_other_shift_rejected(self, offset: tuple[float, ...]) -> None:
        """A ``K_POINTS automatic`` card cannot carry any other shift."""
        from koopmans.input_file import GridKpointsInput

        with pytest.raises(ValueError, match=r"0 \(unshifted\) or 0\.5"):
            GridKpointsInput(grid=(2, 2, 2), offset=offset)

    def test_a_shifted_gamma_point_rejected(self) -> None:
        """Shifting a gamma-only calculation moves it off the point it samples."""
        from koopmans.input_file import GammaOnlyKpointsInput

        with pytest.raises(ValueError, match="samples Gamma itself"):
            GammaOnlyKpointsInput(offset=(0.5, 0.0, 0.0))


def _si_input_with_kpoints(**kpoints: object) -> dict[str, object]:
    """Return the minimal silicon input with its ``kpoints`` block replaced."""
    d = _minimal_si_input()
    d["kpoints"] = kpoints
    return d


class TestPerStepKpoints:
    """``kpoints.overrides`` states the mesh one step samples, absolutely."""

    def test_an_entry_parses_alongside_the_top_level_values(self) -> None:
        """The top-level values stay the default for the steps left out."""
        inp = KoopmansInput.model_validate(
            _si_input_with_kpoints(grid=[2, 2, 2], overrides={"scf": {"grid": [4, 4, 4]}})
        )
        assert inp.kpoints.grid == (2, 2, 2)
        assert inp.kpoints.overrides.scf is not None
        assert inp.kpoints.overrides.scf.grid == (4, 4, 4)
        assert inp.kpoints.overrides.scf.offset is None
        assert inp.kpoints.overrides.nscf is None

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({"bands": {"grid": [4, 4, 4]}}, r"overrides\.bands"),
            ({"scf": {"gird": [4, 4, 4]}}, r"overrides\.scf\.gird"),
        ],
    )
    def test_step_and_attribute_names_are_closed_sets(
        self, overrides: dict[str, object], expected: str, tmp_path: Path
    ) -> None:
        """A typo in either name is a parse error, not a silently dropped mesh."""
        input_file = tmp_path / "input.json"
        input_file.write_text(
            json.dumps(_si_input_with_kpoints(grid=[2, 2, 2], **{"overrides": overrides}))
        )

        with pytest.raises(ValueError, match=rf"{expected}.*is not a valid keyword"):
            read_input_file(input_file)

    def test_grid_and_grid_spacing_exclude_each_other(self) -> None:
        """Both state the same mesh, so accepting both would hide one of them."""
        with pytest.raises(ValueError, match="Give one of the two"):
            KoopmansInput.model_validate(
                _si_input_with_kpoints(
                    grid=[2, 2, 2], overrides={"scf": {"grid": [4, 4, 4], "grid_spacing": 0.15}}
                )
            )

    def test_the_nscf_mesh_must_state_its_dimensions(self) -> None:
        """A spacing cannot say what wannier90 reads as ``mp_grid``."""
        with pytest.raises(ValueError, match=r"nscf\.grid_spacing.*mp_grid"):
            KoopmansInput.model_validate(
                _si_input_with_kpoints(grid=[2, 2, 2], overrides={"nscf": {"grid_spacing": 0.15}})
            )

    def test_the_nscf_mesh_cannot_be_shifted(self) -> None:
        """Reject a shift the nscf mesh, built Gamma-centred, would never carry.

        Every route expands it with ``get_explicit_kpoints``, which returns
        the unshifted k-list, so accepting the keyword would leave the input
        file describing a calculation that never runs.
        """
        with pytest.raises(ValueError, match=r"`nscf\.offset` is not supported"):
            KoopmansInput.model_validate(
                _si_input_with_kpoints(
                    grid=[2, 2, 2], overrides={"nscf": {"offset": [0.5, 0.5, 0.5]}}
                )
            )

    def test_the_scf_mesh_can_still_be_shifted(self) -> None:
        """The rejection is the nscf entry's, not the attribute's.

        An scf converges faster on a shifted mesh, and that mesh reaches
        Quantum ESPRESSO as written.
        """
        inp = KoopmansInput.model_validate(
            _si_input_with_kpoints(grid=[2, 2, 2], overrides={"scf": {"offset": [0.5, 0.5, 0.5]}})
        )
        assert inp.kpoints.overrides.scf is not None
        assert inp.kpoints.overrides.scf.offset == (0.5, 0.5, 0.5)

    def test_wannier90_density_is_unset_without_being_stated(self) -> None:
        """Left out, the entry is unset, like ``scf``/``nscf``: no override to apply."""
        from koopmans.aiida.conversion import wannier90_path_density

        inp = KoopmansInput.model_validate(_si_input_with_kpoints(grid=[2, 2, 2]))
        assert inp.kpoints.overrides.wannier90 is None
        assert wannier90_path_density(inp.kpoints) == 50.0

    def test_wannier90_density_can_be_set_explicitly(self) -> None:
        """A stated ``overrides.wannier90.path_density`` overrides the default."""
        from koopmans.aiida.conversion import wannier90_path_density

        inp = KoopmansInput.model_validate(
            _si_input_with_kpoints(grid=[2, 2, 2], overrides={"wannier90": {"path_density": 80.0}})
        )
        assert inp.kpoints.overrides.wannier90 is not None
        assert inp.kpoints.overrides.wannier90.path_density == 80.0
        assert wannier90_path_density(inp.kpoints) == 80.0

    def test_wannier90_override_survives_a_dump_and_revalidate_round_trip(self) -> None:
        """An unset entry must not look "stated" after ``model_dump`` re-validates.

        ``model_dump`` serializes every field, default or not; if the unset
        sentinel were anything other than ``None`` the round trip used to
        build input variants throughout the test suite would make every
        input look like it explicitly gave ``overrides.wannier90``.
        """
        inp = KoopmansInput.model_validate(_si_input_with_kpoints(gamma_only=True))
        round_tripped = KoopmansInput.model_validate(inp.model_dump())
        assert round_tripped.kpoints.overrides.wannier90 is None

    def test_wannier90_entry_has_no_mesh_fields(self, tmp_path: Path) -> None:
        """The wannier90 entry states a density, not a mesh: it has no `grid` to give."""
        input_file = tmp_path / "input.json"
        input_file.write_text(
            json.dumps(
                _si_input_with_kpoints(grid=[2, 2, 2], overrides={"wannier90": {"grid": [4, 4, 4]}})
            )
        )
        with pytest.raises(ValueError, match=r"overrides\.wannier90\.grid.*is not a valid keyword"):
            read_input_file(input_file)

    def test_gamma_only_has_no_steps_to_override(self) -> None:
        """Every step of a gamma-only calculation samples the same one point."""
        with pytest.raises(ValueError, match=r"overrides\.scf.*gamma_only"):
            KoopmansInput.model_validate(
                _si_input_with_kpoints(gamma_only=True, overrides={"scf": {"grid": [4, 4, 4]}})
            )

    def test_gamma_only_accepts_an_unstated_wannier90_default(self) -> None:
        """A gamma-only input never mentioning `overrides.wannier90` still validates.

        `wannier90` is unset by default, like `scf`/`nscf`; only an input
        that states it itself has anything to reject.
        """
        inp = KoopmansInput.model_validate(_si_input_with_kpoints(gamma_only=True))
        assert inp.kpoints.overrides.wannier90 is None

    def test_gamma_only_rejects_a_stated_wannier90_density(self) -> None:
        """Gamma-only samples one point, so there is no band structure to interpolate."""
        with pytest.raises(ValueError, match=r"overrides\.wannier90.*gamma_only"):
            KoopmansInput.model_validate(
                _si_input_with_kpoints(
                    gamma_only=True, overrides={"wannier90": {"path_density": 80.0}}
                )
            )

    def test_gamma_only_rejects_a_mesh_built_as_a_model(self) -> None:
        """The rejection belongs to the field, not to one way of reaching it.

        A check that reads the raw input dict passes anything already
        validated straight through, so a gamma-only input assembled in code
        would carry a per-step mesh no route can honour.
        """
        from koopmans.input_file import (
            GammaOnlyKpointsInput,
            KpointsOverridesInput,
            StepKpointsOverridesInput,
        )

        with pytest.raises(ValueError, match=r"overrides\.scf.*gamma_only"):
            GammaOnlyKpointsInput(
                overrides=KpointsOverridesInput(scf=StepKpointsOverridesInput(grid=(4, 4, 4)))
            )

    @pytest.mark.parametrize("overrides", ["scf", [{"grid": [4, 4, 4]}], 4])
    def test_an_overrides_block_that_is_not_a_mapping_is_a_parse_error(
        self, overrides: object, tmp_path: Path
    ) -> None:
        """Reported with the file's other errors rather than as a traceback."""
        input_file = tmp_path / "input.json"
        input_file.write_text(
            json.dumps(_si_input_with_kpoints(grid=[2, 2, 2], overrides=overrides))
        )

        with pytest.raises(ValueError, match=r"overrides.*valid dictionary"):
            read_input_file(input_file)


class TestPathDensityRename:
    """``density`` sat beside ``grid`` and read as the density of a mesh."""

    def test_the_old_name_names_its_replacement(self, tmp_path: Path) -> None:
        """A silent alias would leave the input file saying the ambiguous thing."""
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(_si_input_with_kpoints(grid=[2, 2, 2], density=20.0)))

        with pytest.raises(ValueError, match=r"`density` has been renamed `path_density`"):
            read_input_file(input_file)

    def test_the_new_name_carries_the_value(self) -> None:
        """``path_density`` is the same number under a name that says what it counts."""
        inp = KoopmansInput.model_validate(
            _si_input_with_kpoints(grid=[2, 2, 2], path="GX", path_density=20.0)
        )
        assert inp.kpoints.path_density == 20.0


# (keyword, a substring of the explanation it is refused with).
_PH_ROUTE_OWNED_KEYS = [
    ("epsil", "dft_eps"),
    ("trans", "dft_eps"),
    ("verbosity", "AiiDA"),
    ("outdir", "AiiDA"),
]

# (keyword, the value the dft_eps route always forces).
_PH_ROUTE_FORCED_VALUES = [
    ("epsil", True),
    ("trans", False),
    ("verbosity", "high"),
]


class TestPhCalculatorParameters:
    """``calculator_parameters.ph`` mounts the ph.x ``INPUTPH`` namelist (koopmans2#162)."""

    @pytest.mark.parametrize(("keyword", "reason_snippet"), _PH_ROUTE_OWNED_KEYS)
    def test_route_owned_key_is_rejected(self, keyword: str, reason_snippet: str) -> None:
        """A user-set route-owned key fails at parse, naming the key and why it went."""
        d = _si_input_with({"ecutwfc": 20.0})
        _set_keyword(d, "calculator_parameters.ph", keyword, "custom")

        with pytest.raises(ValueError) as excinfo:
            KoopmansInput.model_validate(d)

        message = str(excinfo.value)
        assert f"`calculator_parameters.ph.{keyword}`" in message
        assert reason_snippet in message

    @pytest.mark.parametrize(("keyword", "value"), _PH_ROUTE_FORCED_VALUES)
    def test_restating_the_forced_value_is_rejected_too(self, keyword: str, value: object) -> None:
        """The keyword is gone, so agreeing with the route is refused like disagreeing.

        Accepting the agreeing spelling was how the old check let a stated
        value through: it compared against the field's declared default, so
        whichever value that was passed and then won the merge.
        """
        d = _si_input_with({"ecutwfc": 20.0, "ph": {keyword: value}})
        with pytest.raises(ValueError, match=rf"`calculator_parameters\.ph\.{keyword}`"):
            KoopmansInput.model_validate(d)

    def test_dump_and_revalidate_roundtrips(self) -> None:
        """``model_dump()`` -> ``model_validate()`` must not trip the owned-key checks."""
        inp = KoopmansInput.model_validate(_si_input_with({"ecutwfc": 20.0}))
        KoopmansInput.model_validate(inp.model_dump())

    def test_non_owned_keywords_pass_through(self) -> None:
        """A user-set expert keyword outside the owned set is accepted as-is."""
        d = _si_input_with({"ecutwfc": 20.0, "ph": {"tr2_ph": 1.0e-14, "nmix_ph": 6}})
        inp = KoopmansInput.model_validate(d)
        assert inp.calculator_parameters.ph.tr2_ph == pytest.approx(1.0e-14)
        assert inp.calculator_parameters.ph.nmix_ph == 6

    def test_defaults_leave_the_namelist_unset(self) -> None:
        """With no ``ph`` block, the namelist states nothing explicitly."""
        inp = KoopmansInput.model_validate(_si_input_with({"ecutwfc": 20.0}))
        assert inp.calculator_parameters.ph.model_fields_set == set()


def _collinear_input(**calculator_parameters: object) -> dict[str, object]:
    """Return the minimal silicon input at ``spin = 'collinear'``."""
    d = _si_input_with({"ecutwfc": 20.0, **calculator_parameters})
    _set_keyword(d, "workflow", "spin", "collinear")
    return d


class TestCollinearNeedsAMagnetization:
    """``spin = 'collinear'`` states its moment; koopmans does not pick one."""

    def test_an_input_without_one_is_refused(self) -> None:
        """The refusal names the field to set."""
        with pytest.raises(ValueError) as excinfo:
            KoopmansInput.model_validate(_collinear_input())

        assert "`calculator_parameters.tot_magnetization`" in str(excinfo.value)

    def test_zero_is_a_statement(self) -> None:
        """A closed-shell collinear run is a deliberate input, not a missing one."""
        inp = KoopmansInput.model_validate(_collinear_input(tot_magnetization=0))
        assert inp.calculator_parameters.tot_magnetization == 0

    @pytest.mark.parametrize("task", [task.value for task in Task])
    def test_every_task_is_held_to_it(self, task: str) -> None:
        """The rule belongs to the input file, so no task escapes it.

        The discriminator against a route-local check: the refusal used to
        sit inside individual route builders, so which task enforced it —
        and which path within a task — varied.
        """
        d = _collinear_input()
        _set_keyword(d, "workflow", "task", task)

        with pytest.raises(ValueError, match="tot_magnetization"):
            KoopmansInput.model_validate(d)

    @pytest.mark.parametrize("spin", ["none", "non_collinear", "spin_orbit"])
    def test_the_other_regimes_need_no_moment(self, spin: str) -> None:
        """Only collinear splits the electrons between two channels.

        ``none`` has one channel, and the two spinor regimes pin no moment
        along any axis, so a magnetization is neither needed nor meaningful.
        """
        d = _si_input_with({"ecutwfc": 20.0})
        _set_keyword(d, "workflow", "spin", spin)
        inp = KoopmansInput.model_validate(d)
        assert inp.calculator_parameters.tot_magnetization is None

    def test_the_pw_namelist_spelling_does_not_satisfy_it(self) -> None:
        """``pw.system.tot_magnetization`` reaches pw.x alone, and not on every route.

        The kcp.x and kcw.x steps read the shared field, so a moment given
        only to pw.x leaves those channels unstated.
        """
        d = _collinear_input(pw={"system": {"tot_magnetization": 2}})

        with pytest.raises(ValueError) as excinfo:
            KoopmansInput.model_validate(d)

        assert "`calculator_parameters.tot_magnetization`" in str(excinfo.value)

    def test_the_file_level_refusal_names_no_empty_location(self, tmp_path: Path) -> None:
        """A rule over the whole file has no field to point at, and says so.

        Every other refusal is reported against the field that carries it,
        which this one has none of; the reader should not be handed an empty
        pair of backticks.
        """
        input_file = tmp_path / "collinear.json"
        input_file.write_text(json.dumps(_collinear_input()))

        with pytest.raises(ValueError) as excinfo:
            read_input_file(input_file)

        assert "``" not in str(excinfo.value)
        assert "tot_magnetization" in str(excinfo.value)

    def test_a_dumped_input_is_still_refused(self) -> None:
        """The check reads the moment's value, not whether the key is present.

        ``model_dump()`` states every field, the moment nobody set included,
        so an input that reached the check as a dumped model would satisfy a
        key-presence test while stating nothing.
        """
        dumped = KoopmansInput.model_validate(_si_input_with({"ecutwfc": 20.0})).model_dump()
        _set_keyword(dumped, "workflow", "spin", "collinear")

        with pytest.raises(ValueError, match="tot_magnetization"):
            KoopmansInput.model_validate(dumped)


class TestTheMomentIsWholeElectrons:
    """A magnetization counts unpaired electrons, so it cannot be a fraction."""

    @pytest.mark.parametrize("magnetization", [0.5, 2.7, -1.5])
    def test_a_fraction_is_refused(self, magnetization: float) -> None:
        """Truncating would run a different calculation than the one asked for.

        ``0.5`` is the discriminating value: rounded down it becomes a
        closed-shell run, which pw.x accepts and nothing downstream records
        as a substitution.
        """
        with pytest.raises(ValueError) as excinfo:
            KoopmansInput.model_validate(_collinear_input(tot_magnetization=magnetization))

        assert "whole" in str(excinfo.value)

    @pytest.mark.parametrize("magnetization", [0, 2, 2.0, -1])
    def test_a_whole_number_passes_however_it_is_written(self, magnetization: float) -> None:
        """The rule is the value, not the JSON type: ``2.0`` states two electrons."""
        inp = KoopmansInput.model_validate(_collinear_input(tot_magnetization=magnetization))
        assert inp.calculator_parameters.tot_magnetization == pytest.approx(magnetization)

    @pytest.mark.parametrize("task", [task.value for task in Task])
    def test_every_task_is_held_to_it(self, task: str) -> None:
        """Every route fixes its occupations, so none of them can place half an electron.

        The discriminator against a route-local check: the coercion this
        replaces sat in the dispatcher, so a task reaching pw.x by another
        path silently rounded instead.
        """
        d = _si_input_with({"ecutwfc": 20.0, "tot_magnetization": 0.5})
        _set_keyword(d, "workflow", "task", task)

        with pytest.raises(ValueError, match="whole"):
            KoopmansInput.model_validate(d)

    def test_the_pw_namelist_spelling_is_refused(self) -> None:
        """``pw.system.tot_magnetization`` has no input-file spelling.

        The moment has exactly one spelling, ``calculator_parameters.
        tot_magnetization``; every route that runs pw.x under ``nspin = 2``
        writes the namelist keyword from there, so stating it directly
        would risk a second, disagreeing value.
        """
        d = _si_input_with(
            {
                "ecutwfc": 20.0,
                "pw": {
                    "system": {
                        "tot_magnetization": 0.5,
                        "occupations": "smearing",
                        "degauss": 0.01,
                    }
                },
            }
        )
        with pytest.raises(ValueError) as excinfo:
            KoopmansInput.model_validate(d)

        message = str(excinfo.value)
        assert "`calculator_parameters.pw.system.tot_magnetization`" in message
        assert "`calculator_parameters.tot_magnetization`" in message
