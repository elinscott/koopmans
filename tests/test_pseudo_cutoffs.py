"""A pseudo family publishing no recommended cutoffs takes them from the input."""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.input_file import KoopmansInput
from tests.fixtures import silicon_pw_input


def _build_scf_pw(code: Any, structure: Any, overrides: dict[str, Any]) -> Any:
    """Return the scf pw sub-builder ``PwBandsWorkChain`` assembles."""
    from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

    builder = PwBandsWorkChain.get_builder_from_protocol(
        code=code, structure=structure, overrides=overrides
    )
    return builder.scf.pw


class TestFamilyWithCutoffs:
    """A family that recommends cutoffs keeps recommending them."""

    def test_an_input_may_still_omit_the_cutoffs(
        self, aiida_profile_clean: Any, installed_pw_code: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """An input naming no cutoffs at all builds on the family's recommendation.

        Discriminates the added branch from a replacement of the old one: if
        the pseudos were pinned unconditionally, this input would be rejected
        for the cutoffs it does not state.
        """
        from koopmans.aiida.workflows import prepare_common_inputs

        inp = KoopmansInput.model_validate(
            silicon_pw_input(pseudo_library="SG15/1.0/PBE/SR", calculator_parameters={})
        )
        structure, _, overrides = prepare_common_inputs(inp, ["scf", "bands"])
        assert "pseudos" not in overrides["scf"]["pw"]

        recommended = fake_sg15_cutoffs_family.get_recommended_cutoffs(
            structure=structure, unit="Ry"
        )
        system = _build_scf_pw(installed_pw_code, structure, overrides).parameters["SYSTEM"]
        assert (system["ecutwfc"], system["ecutrho"]) == pytest.approx(recommended)


class TestFamilyWithoutCutoffs:
    """Both shapes an uncut family takes are driven from the input file."""

    @pytest.mark.parametrize(
        ("fixture", "label"),
        [
            ("fake_sg15_family_without_cutoffs", "SG15/1.2/PBE/FR"),
            ("fake_user_built_family", "MyPseudos/local"),
        ],
    )
    def test_input_cutoffs_and_pinned_pseudos_build(
        self,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        request: Any,
        fixture: str,
        label: str,
    ) -> None:
        """With both cutoffs given, the build succeeds and carries them.

        Covers the family koopmans installs itself (a cutoffs family with no
        stringency) and the one ``aiida-pseudo install family`` produces (a
        plain family, which cannot carry cutoffs at all). Before the fix
        neither reached a builder at all. ``ecutrho`` at 80 Ry shows the
        top-level ``ecutwfc`` shorthand pinning it at four times the input's
        20 Ry.
        """
        from koopmans.aiida.workflows import prepare_common_inputs

        family = request.getfixturevalue(fixture)
        inp = KoopmansInput.model_validate(silicon_pw_input(pseudo_library=label))
        structure, _, overrides = prepare_common_inputs(inp, ["scf", "bands"])

        pw = _build_scf_pw(installed_pw_code, structure, overrides)
        assert pw.parameters["SYSTEM"]["ecutwfc"] == pytest.approx(20.0)
        assert pw.parameters["SYSTEM"]["ecutrho"] == pytest.approx(80.0)
        assert pw.pseudos["Si"].uuid == family.get_pseudos(structure=structure)["Si"].uuid

    def test_missing_ecutrho_names_the_family_and_both_keywords(
        self, aiida_profile_clean: Any, fake_sg15_family_without_cutoffs: Any
    ) -> None:
        """A half-specified input fails on koopmans' own message.

        ``pw.system.ecutwfc`` alone leaves ``ecutrho`` unset — the top-level
        ``calculator_parameters.ecutwfc`` shorthand is what pins both. The
        message must not repeat upstream's claim that the family, installed
        moments earlier, is not installed.
        """
        from koopmans.aiida.workflows import prepare_common_inputs

        inp = KoopmansInput.model_validate(
            silicon_pw_input(
                pseudo_library="SG15/1.2/PBE/FR",
                calculator_parameters={"pw": {"system": {"ecutwfc": 20.0}}},
            )
        )
        with pytest.raises(ValueError) as excinfo:
            prepare_common_inputs(inp, ["scf", "bands"])

        message = str(excinfo.value)
        assert "SG15/1.2/PBE/FR" in message
        assert "calculator_parameters.pw.system.ecutwfc" in message
        assert "calculator_parameters.pw.system.ecutrho" in message
        assert "is not installed" not in message

    def test_workgraph_builds_with_pinned_pseudos(
        self,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        fake_sg15_family_without_cutoffs: Any,
    ) -> None:
        """The pinned pseudo nodes survive the graph input serialization.

        The override dict now carries ``UpfData`` nodes; this is the check
        that ``build_workgraph`` still assembles a WorkGraph around them.
        """
        from koopmans.aiida.workflows import build_workgraph

        inp = KoopmansInput.model_validate(silicon_pw_input(pseudo_library="SG15/1.2/PBE/FR"))
        wg = build_workgraph(inp)
        overrides = wg.tasks["PwBandsWorkChain"].inputs["scf"]["pw"]["parameters"]
        assert overrides.value.get_dict()["SYSTEM"]["ecutwfc"] == pytest.approx(20.0)
