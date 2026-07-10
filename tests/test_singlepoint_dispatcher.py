"""Tests for the Koopmans singlepoint dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from koopmans.aiida.workflows import (
    _build_singlepoint_workgraph,
    _kcp_dscf_inputs,
    _KcpDscfInputs,
)
from koopmans.input_file import KoopmansInput, read_input_file
from koopmans.input_file.workflow import (
    CalculateScreeningMethod,
    Correction,
    Task,
    VariationalOrbitalType,
)


@pytest.fixture
def ozone_json(tutorials_dir: Path) -> Path:
    """Return the path to the ozone tutorial JSON."""
    return tutorials_dir / "ozone.json"


@pytest.fixture
def ozone_input(ozone_json: Path) -> KoopmansInput:
    """Return a freshly parsed ozone ``KoopmansInput``."""
    return read_input_file(ozone_json)


def _copy_with_calc_overrides(inp: KoopmansInput, **calc_param_updates: object) -> KoopmansInput:
    """Return a fresh ``KoopmansInput`` with patched ``calculator_parameters``.

    Each key in ``calc_param_updates`` is a dotted path into
    ``calculator_parameters`` (e.g. ``"kcp.system.ecutrho"``). Pydantic's
    ``model_copy`` is shallow, so we dump, mutate, and re-validate.
    """
    d = inp.model_dump()
    calc = d["calculator_parameters"]
    for dotted, value in calc_param_updates.items():
        target = calc
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return KoopmansInput.model_validate(d)


class TestOzoneInputParse:
    """Parsing checks for the ozone tutorial JSON."""

    def test_ozone_parses_with_expected_workflow_fields(
        self, ozone_json: Path, ozone_input: KoopmansInput
    ) -> None:
        """The ozone tutorial should parse with the expected workflow fields."""
        assert ozone_json.exists(), f"Tutorial file not found: {ozone_json}"

        assert isinstance(ozone_input, KoopmansInput)
        assert ozone_input.workflow.task == Task.SINGLEPOINT
        assert ozone_input.workflow.correction == Correction.KI
        assert ozone_input.workflow.screening_method == CalculateScreeningMethod.DSCF
        assert ozone_input.workflow.init_orbitals == VariationalOrbitalType.KOHN_SHAM
        assert ozone_input.workflow.alpha_numsteps == 1
        assert ozone_input.workflow.pseudo_library == "SG15/1.2/PBE/SR"

    def test_ozone_is_non_periodic(self, ozone_input: KoopmansInput) -> None:
        """The ozone tutorial should be non-periodic (molecule in a box)."""
        assert ozone_input.atoms.cell_parameters.periodic is False

    def test_ozone_has_three_oxygen_atoms(self, ozone_input: KoopmansInput) -> None:
        """The ozone tutorial should have three oxygens at the expected positions."""
        positions = ozone_input.atoms.atomic_positions.positions
        assert len(positions) == 3
        assert all(atom[0] == "O" for atom in positions)

        expected = [
            ("O", 7.0869, 6.0, 5.89),
            ("O", 8.1738, 6.0, 6.55),
            ("O", 6.0, 6.0, 6.55),
        ]
        for got, want in zip(positions, expected, strict=True):
            assert got[0] == want[0]
            assert got[1] == pytest.approx(want[1])
            assert got[2] == pytest.approx(want[2])
            assert got[3] == pytest.approx(want[3])


def _scalars(inputs: _KcpDscfInputs) -> tuple[float, float, int, int]:
    """Project the (ecutwfc, ecutrho, nbnd, nspin) corner of a kcp input bundle."""
    return (inputs["ecutwfc"], inputs["ecutrho"], inputs["nbnd"], inputs["nspin"])


class TestKcpDscfInputs:
    """Unit tests for ``_kcp_dscf_inputs``."""

    def test_ozone_default(self, ozone_input: KoopmansInput) -> None:
        """Ozone input should yield (65.0, 260.0, 10, 2)."""
        assert _scalars(_kcp_dscf_inputs(ozone_input)) == (65.0, 260.0, 10, 2)

    def test_ecutrho_defaults_to_four_times_ecutwfc(self, ozone_input: KoopmansInput) -> None:
        """With ecutrho unset, it should default to 4 * ecutwfc (4 * 65 = 260)."""
        inp = _copy_with_calc_overrides(ozone_input, **{"kcp.system.ecutrho": 0.0})
        assert _scalars(_kcp_dscf_inputs(inp)) == (65.0, 260.0, 10, 2)

    def test_ecutrho_default_with_custom_ecutwfc(self, ozone_input: KoopmansInput) -> None:
        """With ecutwfc=30 and no ecutrho, ecutrho should fall back to 120.0."""
        inp = _copy_with_calc_overrides(
            ozone_input,
            ecutwfc=30.0,
            **{"kcp.system.ecutrho": 0.0},
        )
        assert _scalars(_kcp_dscf_inputs(inp)) == (30.0, 120.0, 10, 2)

    def test_missing_ecutwfc_raises_valueerror(self, ozone_input: KoopmansInput) -> None:
        """Missing ecutwfc (both top-level and kcp.system) should raise ValueError."""
        inp = _copy_with_calc_overrides(
            ozone_input,
            ecutwfc=None,
            **{"kcp.system.ecutwfc": 0.0},
        )
        with pytest.raises(ValueError, match="ecutwfc is required"):
            _kcp_dscf_inputs(inp)

    def test_missing_nbnd_raises_valueerror(self, ozone_input: KoopmansInput) -> None:
        """Missing nbnd (both top-level and kcp.system) should raise ValueError."""
        inp = _copy_with_calc_overrides(
            ozone_input,
            nbnd=None,
            **{"kcp.system.nbnd": None},
        )
        with pytest.raises(ValueError, match="nbnd is required"):
            _kcp_dscf_inputs(inp)

    def test_workflow_fields_forwarded(self, ozone_input: KoopmansInput) -> None:
        """The workflow-level fields should land in the bundle unchanged."""
        inputs = _kcp_dscf_inputs(ozone_input)
        workflow = ozone_input.workflow
        assert inputs["pseudo_family"] == workflow.pseudo_library
        assert inputs["correction"] == workflow.correction
        assert inputs["init_orbitals"] == workflow.init_orbitals
        assert inputs["alpha_numsteps"] == workflow.alpha_numsteps
        assert inputs["initial_alpha"] == workflow.alpha_guess


class TestBuildSinglepointWorkgraphScopeGuards:
    """Scope-guard tests for ``_build_singlepoint_workgraph``.

    These fire BEFORE ``ensure_pseudo_family_installed``, so they are testable
    without an AiiDA profile or database.
    """

    @pytest.mark.parametrize(
        "correction_value",
        ["pkipz", "none", "all"],
    )
    def test_unsupported_correction_raises_notimplemented(
        self, ozone_input: KoopmansInput, correction_value: str
    ) -> None:
        """Only KI/KIPZ are implemented; other corrections should raise NotImplementedError."""
        d = ozone_input.model_dump()
        d["workflow"]["correction"] = correction_value
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match="correction="):
            _build_singlepoint_workgraph(inp, codes={})

    @pytest.mark.parametrize("correction_value", ["kipz", "pkipz", "none", "all"])
    def test_dfpt_rejects_non_ki_corrections(
        self, ozone_input: KoopmansInput, correction_value: str
    ) -> None:
        """The DFPT route (kcw.x) implements KI only; anything else must raise loudly.

        Guards against silently running KI physics for a requested KIPZ (etc.)
        correction: the DFPT branch dispatches on ``screening_method`` before
        the DSCF correction guard is reached.
        """
        d = ozone_input.model_dump()
        d["workflow"]["screening_method"] = "dfpt"
        d["workflow"]["correction"] = correction_value
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match="only implements the KI correction"):
            _build_singlepoint_workgraph(inp, codes={})
