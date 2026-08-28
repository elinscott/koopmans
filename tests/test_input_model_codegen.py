"""Tests for the generated input-file models and the generator behind them."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from aiida_koopmans.owned_keywords import OWNED, SEEDED_VALUES
from pydantic import ValidationError

import koopmans
from koopmans.input_file import CalculatorParametersInput
from koopmans.input_file._codegen import MODULES, REASONS, UNREACHABLE, generate, render

_GENERATED = Path(koopmans.__file__ or "").parent / "input_file" / "_generated"


class TestGenerationIsReproducible:
    """The checked-in models must be exactly what the generator emits."""

    def test_the_checked_in_files_match_the_generator(self, tmp_path: Path) -> None:
        """A stale generated file would let the schema drift from the ownership data."""
        for path in generate(tmp_path):
            assert path.read_text() == (_GENERATED / path.name).read_text(), path.name

    def test_rendering_twice_gives_the_same_text(self) -> None:
        """Set iteration or dict ordering leaking into the output would churn the diff."""
        for module in MODULES:
            assert render(module) == render(module)


class TestGeneratedFieldSets:
    """Each generated model drops the claimed keywords and keeps every other one."""

    @pytest.mark.parametrize("module", MODULES, ids=lambda m: m.filename)
    def test_exactly_the_claimed_keywords_are_dropped(self, module: Any) -> None:
        """A keyword dropped without a claim would vanish with no explanation."""
        for model in module.models:
            generic = getattr(import_module(model.source), model.name)
            restricted = getattr(
                import_module(f"koopmans.input_file._generated.{module.filename[:-3]}"),
                model.emitted,
            )
            dropped = generic.model_fields.keys() - restricted.model_fields.keys()
            claimed = set(OWNED[model.block]) | set(UNREACHABLE.get(model.block, {}))
            assert dropped == claimed, model.name


class TestSeededDefaultsMatchTheRoster:
    """A keyword a route seeds is the input file's default, and the schema says so."""

    @pytest.mark.parametrize("module", MODULES, ids=lambda m: m.filename)
    def test_seeded_fields_carry_the_roster_value_as_their_default(self, module: Any) -> None:
        """The generated field's default is the value SEEDED_VALUES pins.

        The description gains a seed note only when it says something the
        default doesn't already: when the roster value matches the generic
        model's own literal default, a note stating "koopmans seeds X;
        code's own default is X" would be a redundant restatement, so the
        description is left as the generic model's.
        """
        for model in module.models:
            roster = SEEDED_VALUES.get(model.block, {})
            if not roster:
                continue
            generic = getattr(import_module(model.source), model.name)
            restricted = getattr(
                import_module(f"koopmans.input_file._generated.{module.filename[:-3]}"),
                model.emitted,
            )
            for keyword, value in roster.items():
                field = restricted.model_fields[keyword]
                assert field.default == value, (model.block, keyword)

                generic_field = generic.model_fields[keyword]
                extra = generic_field.json_schema_extra or {}
                derived = bool(
                    extra.get("default_ref")
                    or extra.get("default_expr")
                    or extra.get("computed_default")
                )
                if derived or generic_field.default != value:
                    assert f"koopmans seeds {value!r}" in (field.description or ""), (
                        model.block,
                        keyword,
                    )
                else:
                    assert f"koopmans seeds {value!r}" not in (field.description or ""), (
                        model.block,
                        keyword,
                    )
                    assert field.description == generic_field.description, (model.block, keyword)

    def test_a_ref_encoded_default_states_that_the_code_derives_its_own(self) -> None:
        """kcw.x's own default for niter is a QE-internal ref, not the literal None.

        pydantic-espresso cannot fold ``<default kind="ref">`` into a Python
        literal, so the generic model states ``None`` as a placeholder and
        carries the real story in ``json_schema_extra["default_ref"]``.
        Reporting that placeholder verbatim would tell the reader kcw.x's
        own default for a required iteration count is "None".
        """
        from koopmans.input_file._generated.kcw import ScreenNamelist

        description = ScreenNamelist.model_fields["niter"].description or ""
        assert "kcw.x derives its own default" in description
        assert "kcw.x's own default is None" not in description


class TestUnreachableKeywordsAreRefused:
    """A keyword koopmans cannot pass through says why, rather than vanishing."""

    def test_the_message_explains_why_the_keyword_is_gone(self) -> None:
        """The reader learns the keyword is unreachable, not that they mistyped it."""
        with pytest.raises(ValidationError) as excinfo:
            CalculatorParametersInput.model_validate({"kcw": {"wannier": {"alpha_mix": [0.5]}}})

        message = str(excinfo.value)
        assert "`calculator_parameters.kcw.wannier.alpha_mix` cannot be set" in message
        assert "`SCREEN` namelist" in message


class TestOwnedKeywordsAreRefused:
    """A keyword the workflow determines is not an input-file keyword."""

    @pytest.mark.parametrize(
        ("block", "keyword", "reason_snippet"),
        [
            ("pw.system", "ecutwfc", "calculator_parameters.ecutwfc"),
            ("pw.system", "nspin", "workflow.spin"),
            ("pw.system", "tot_magnetization", "calculator_parameters.tot_magnetization"),
            ("pw.control", "verbosity", "high verbosity"),
            ("ph", "trans", "dft_eps"),
            ("pw2wannier90", "spin_component", "workflow.spin"),
            ("wannier90", "num_wann", "projections"),
            ("wannier90", "write_u_matrices", "gauge products"),
        ],
    )
    def test_the_message_names_what_to_set_instead(
        self, block: str, keyword: str, reason_snippet: str
    ) -> None:
        """The reader is told the input-file field that determines the value."""
        payload: dict[str, Any] = {}
        target = payload
        for part in block.split("."):
            target = target.setdefault(part, {})
        target[keyword] = 1

        with pytest.raises(ValidationError) as excinfo:
            CalculatorParametersInput.model_validate(payload)

        message = str(excinfo.value)
        assert f"`calculator_parameters.{block}.{keyword}` is not a koopmans keyword" in message
        assert reason_snippet in message

    def test_an_unknown_keyword_still_gets_the_generic_complaint(self) -> None:
        """Only owned keywords get an explanation; a typo is not one of them."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CalculatorParametersInput.model_validate({"pw": {"system": {"ecutwfx": 20.0}}})


class TestNonOwnedKeywordsSurvive:
    """Generation must not disturb the keywords a user is meant to set."""

    def test_a_kept_keyword_carries_its_type_and_description(self) -> None:
        """Copying by line rather than by value keeps types, defaults and help text."""
        parameters = CalculatorParametersInput.model_validate(
            {"pw": {"system": {"nbnd": 20, "nosym": True}}, "ph": {"tr2_ph": 1e-14}}
        )
        assert parameters.pw.system.nbnd == 20
        assert parameters.pw.system.nosym is True
        assert parameters.ph.tr2_ph == pytest.approx(1e-14)
        assert type(parameters.ph).model_fields["tr2_ph"].description

    def test_a_copied_field_validator_still_runs(self) -> None:
        """``SYSTEM.smearing`` maps its aliases upstream; the copy must keep doing so."""
        parameters = CalculatorParametersInput.model_validate(
            {"pw": {"system": {"smearing": "gauss"}}}
        )
        assert parameters.pw.system.smearing == "gaussian"

    def test_dump_and_revalidate_roundtrips(self) -> None:
        """The suite re-validates modified inputs through ``model_dump()``."""
        parameters = CalculatorParametersInput.model_validate(
            {"ecutwfc": 20.0, "pw": {"system": {"nbnd": 20}}, "ph": {"tr2_ph": 1e-14}}
        )
        again = CalculatorParametersInput.model_validate(json.loads(parameters.model_dump_json()))
        assert again.pw.system.nbnd == 20

    def test_projections_keep_the_koopmans_block_shape(self) -> None:
        """The public model's own additions survive the generated base."""
        block = [{"fractional_site": [0.25, 0.25, 0.25], "ang_mtm": "sp3"}]
        parameters = CalculatorParametersInput.model_validate(
            {"wannier90": {"projections": [block, block]}}
        )
        assert len(parameters.wannier90.projections) == 2


class TestDriftAlarms:
    """Ownership data and the reason map must be kept in step by hand."""

    def test_an_unexplained_owned_keyword_refuses_to_generate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Claiming a keyword without saying what to set instead is refused."""
        monkeypatch.setitem(OWNED, "ph.INPUTPH", OWNED["ph.INPUTPH"] | {"nmix_ph"})
        with pytest.raises(ValueError, match=r"nmix_ph would vanish.*REASONS"):
            generate(tmp_path)

    def test_an_explanation_for_a_settable_keyword_refuses_to_generate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explanation left behind after a keyword became settable again is stale."""
        monkeypatch.setitem(REASONS, "ph.INPUTPH", {**REASONS["ph.INPUTPH"], "nmix_ph": "no"})
        with pytest.raises(ValueError, match=r"nmix_ph is explained in REASONS but not owned"):
            generate(tmp_path)

    @pytest.mark.parametrize("roster", ["owned", "unreachable"])
    def test_a_claimed_block_with_no_generated_model_refuses_to_generate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, roster: str
    ) -> None:
        """A claimed block no model covers would leave its keywords in the input file."""
        if roster == "owned":
            monkeypatch.setitem(OWNED, "pp.INPUTPP", frozenset({"prefix"}))
        else:
            monkeypatch.setitem(UNREACHABLE, "pp.INPUTPP", {"prefix": "unreachable"})
        with pytest.raises(ValueError, match=r"pp.INPUTPP is claimed"):
            generate(tmp_path)

    def test_a_keyword_claimed_twice_refuses_to_generate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Owned and unreachable are different stories; one keyword cannot have both."""
        monkeypatch.setitem(UNREACHABLE, "ph.INPUTPH", {"trans": "unreachable"})
        with pytest.raises(ValueError, match=r"trans is both owned and unreachable"):
            generate(tmp_path)

    def test_a_keyword_that_is_not_a_field_refuses_to_generate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A misspelt keyword would otherwise be dropped from nothing, silently."""
        monkeypatch.setitem(OWNED, "ph.INPUTPH", OWNED["ph.INPUTPH"] | {"epsilon"})
        monkeypatch.setitem(REASONS, "ph.INPUTPH", {**REASONS["ph.INPUTPH"], "epsilon": "typo"})
        with pytest.raises(ValueError, match=r"declares no epsilon.*stale"):
            generate(tmp_path)

    def test_a_seeded_keyword_that_is_not_a_field_refuses_to_generate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A misspelt roster keyword would otherwise silently keep no default at all."""
        monkeypatch.setitem(
            SEEDED_VALUES, "kcw.SCREEN", {**SEEDED_VALUES["kcw.SCREEN"], "trr2": 1e-18}
        )
        with pytest.raises(ValueError, match=r"declares no trr2.*SEEDED_VALUES is stale"):
            generate(tmp_path)
