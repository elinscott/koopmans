"""Tests for the Koopmans singlepoint dispatcher (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from koopmans.aiida.workflows import (
    _build_singlepoint_workgraph,
    _extract_kcp_scalar_inputs,
    build_workgraph,
    load_codes_for_task,
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


class TestExtractKcpScalarInputs:
    """Unit tests for ``_extract_kcp_scalar_inputs``."""

    def test_ozone_default(self, ozone_input: KoopmansInput) -> None:
        """Ozone input should yield (65.0, 260.0, 10, 2)."""
        assert _extract_kcp_scalar_inputs(ozone_input) == (65.0, 260.0, 10, 2)

    def test_ecutrho_defaults_to_four_times_ecutwfc(self, ozone_input: KoopmansInput) -> None:
        """With ecutrho unset, it should default to 4 * ecutwfc (4 * 65 = 260)."""
        inp = _copy_with_calc_overrides(ozone_input, **{"kcp.system.ecutrho": 0.0})
        assert _extract_kcp_scalar_inputs(inp) == (65.0, 260.0, 10, 2)

    def test_ecutrho_default_with_custom_ecutwfc(self, ozone_input: KoopmansInput) -> None:
        """With ecutwfc=30 and no ecutrho, ecutrho should fall back to 120.0."""
        inp = _copy_with_calc_overrides(
            ozone_input,
            ecutwfc=30.0,
            **{"kcp.system.ecutrho": 0.0},
        )
        assert _extract_kcp_scalar_inputs(inp) == (30.0, 120.0, 10, 2)

    def test_missing_ecutwfc_raises_valueerror(self, ozone_input: KoopmansInput) -> None:
        """Missing ecutwfc (both top-level and kcp.system) should raise ValueError."""
        inp = _copy_with_calc_overrides(
            ozone_input,
            ecutwfc=None,
            **{"kcp.system.ecutwfc": 0.0},
        )
        with pytest.raises(ValueError, match="ecutwfc is required"):
            _extract_kcp_scalar_inputs(inp)

    def test_missing_nbnd_raises_valueerror(self, ozone_input: KoopmansInput) -> None:
        """Missing nbnd (both top-level and kcp.system) should raise ValueError."""
        inp = _copy_with_calc_overrides(
            ozone_input,
            nbnd=None,
            **{"kcp.system.nbnd": None},
        )
        with pytest.raises(ValueError, match="nbnd is required"):
            _extract_kcp_scalar_inputs(inp)


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


class TestProfileRequired:
    """Tests that require a loaded AiiDA profile; skipped by default."""

    @pytest.mark.skip(reason="requires loaded AiiDA profile with pw.x code")
    def test_dfpt_screening_method_raises_notimplemented(self, ozone_input: KoopmansInput) -> None:
        """``screening_method=dfpt`` should raise ``NotImplementedError``.

        ``load_codes_for_task`` tries to load pw.x FIRST, so without a profile
        the test would surface a ``ValueError`` about the missing code instead.
        Skip until we have a profile fixture.
        """
        d = ozone_input.model_dump()
        d["workflow"]["screening_method"] = "dfpt"
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match=r"screening_method=.*dfpt"):
            load_codes_for_task(inp.workflow)

    @pytest.mark.skip(reason="requires loaded AiiDA profile with pw.x code")
    def test_build_workgraph_kipz_raises_notimplemented(self, ozone_input: KoopmansInput) -> None:
        """``build_workgraph`` with KIPZ correction should raise ``NotImplementedError``.

        The dispatcher calls ``load_codes_for_task`` first, which requires a
        profile (pw.x lookup), so this test is skipped until we have one.
        """
        d = ozone_input.model_dump()
        d["workflow"]["correction"] = "kipz"
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match="correction="):
            build_workgraph(inp)
