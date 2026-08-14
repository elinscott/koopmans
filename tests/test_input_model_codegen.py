"""Tests for the generated input-file models and the generator behind them."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from aiida_koopmans.owned_keywords import OWNED, ROUTE_CONDITIONAL
from pydantic import ValidationError

import koopmans
from koopmans.input_file import CalculatorParametersInput
from koopmans.input_file._codegen import MODULES, REASONS, generate, render
from koopmans.input_file._route_conditional import ROUTE_REFUSALS, check_route_refusals
from koopmans.input_file.workflow import WorkflowConfig


def _never(workflow: WorkflowConfig, path: str) -> str | None:
    """Accept every route; stands in for a real refusal in the drift alarms."""
    return None


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
    """Each generated model drops the owned keywords and keeps every other one."""

    @pytest.mark.parametrize("module", MODULES, ids=lambda m: m.filename)
    def test_exactly_the_owned_keywords_are_dropped(self, module: Any) -> None:
        """A keyword dropped without being owned would vanish with no explanation."""
        for model in module.models:
            generic = getattr(import_module(model.source), model.name)
            restricted = getattr(
                import_module(f"koopmans.input_file._generated.{module.filename[:-3]}"),
                model.emitted,
            )
            dropped = generic.model_fields.keys() - restricted.model_fields.keys()
            assert dropped == set(OWNED[model.block]), model.name


class TestOwnedKeywordsAreRefused:
    """A keyword the workflow determines is not an input-file keyword."""

    @pytest.mark.parametrize(
        ("block", "keyword", "reason_snippet"),
        [
            ("pw.system", "ecutwfc", "calculator_parameters.ecutwfc"),
            ("pw.system", "nspin", "workflow.spin"),
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

    def test_an_owned_block_with_no_generated_model_refuses_to_generate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A claimed block no model covers would leave its keywords in the input file."""
        monkeypatch.setitem(OWNED, "kcw.CONTROL", frozenset({"prefix"}))
        with pytest.raises(ValueError, match=r"kcw.CONTROL, which no generated model covers"):
            generate(tmp_path)

    def test_a_keyword_that_is_not_a_field_refuses_to_generate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A misspelt keyword would otherwise be dropped from nothing, silently."""
        monkeypatch.setitem(OWNED, "ph.INPUTPH", OWNED["ph.INPUTPH"] | {"epsilon"})
        monkeypatch.setitem(REASONS, "ph.INPUTPH", {**REASONS["ph.INPUTPH"], "epsilon": "typo"})
        with pytest.raises(ValueError, match=r"declares no epsilon.*stale"):
            generate(tmp_path)


class TestRouteConditionalAlarms:
    """A route-conditional keyword and its refusal must be declared together."""

    def test_an_unrefused_conditional_keyword_refuses_to_generate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Declaring a keyword conditional without refusing it restores the silent drop."""
        monkeypatch.setitem(
            ROUTE_CONDITIONAL, "pw.SYSTEM", ROUTE_CONDITIONAL["pw.SYSTEM"] | {"nbnd"}
        )
        with pytest.raises(ValueError, match=r"nbnd would stay settable.*ROUTE_REFUSALS"):
            generate(tmp_path)

    def test_a_refusal_for_an_unclassified_keyword_refuses_to_generate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal left behind after a route stopped forcing the keyword is stale."""
        monkeypatch.setitem(
            ROUTE_REFUSALS,
            "pw.SYSTEM",
            {**ROUTE_REFUSALS["pw.SYSTEM"], "nbnd": {"calculator_parameters.nbnd": _never}},
        )
        with pytest.raises(ValueError, match=r"nbnd is refused in ROUTE_REFUSALS"):
            generate(tmp_path)

    def test_every_declared_keyword_is_refused_today(self) -> None:
        """The alarm is only worth having if the checked-in declarations agree."""
        check_route_refusals()
