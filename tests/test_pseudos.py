"""Tests for resolving the ``pseudo_library`` label to a pseudopotential family."""

from __future__ import annotations

from typing import Any

import pytest


class TestEnsurePseudoFamilyInstalled:
    """``ensure_pseudo_family_installed`` downloads only what is missing."""

    def test_installed_family_under_arbitrary_label_is_used_as_is(
        self, fake_user_built_family: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A label matching no download format resolves to the installed family."""
        from koopmans.aiida.setup import pseudos

        def _fail(label: str) -> None:
            raise AssertionError(f"attempted to download '{label}'")

        monkeypatch.setattr(pseudos, "install_pseudo_family", _fail)

        pseudos.ensure_pseudo_family_installed(fake_user_built_family.label)

    def test_uninstallable_label_reports_both_routes(self, aiida_profile_clean: Any) -> None:
        """An unknown label names the install command and the download formats."""
        from koopmans.aiida.setup import pseudos

        with pytest.raises(ValueError) as excinfo:
            pseudos.ensure_pseudo_family_installed("my-gaas-fr")

        message = str(excinfo.value)
        assert "No installed pseudopotential family has the label 'my-gaas-fr'" in message
        assert "aiida-pseudo install family <directory> my-gaas-fr\n" in message
        assert "calculator_parameters.ecutwfc" in message
        assert "PseudoDojo/version/functional/relativistic/protocol/format" in message

    def test_the_install_command_asks_for_no_cutoffs_family(self, aiida_profile_clean: Any) -> None:
        """The command offered is the plain-family one, with no cutoffs to set.

        A family built with ``-F pseudo.family.cutoffs`` and no
        ``aiida-pseudo family cutoffs set`` can recommend no cutoffs, which is
        the half-configured shape the message must not send anyone to; the
        cutoffs come from the input file instead.
        """
        from koopmans.aiida.setup import pseudos

        with pytest.raises(ValueError) as excinfo:
            pseudos.ensure_pseudo_family_installed("my-gaas-fr")

        message = str(excinfo.value)
        assert "-F" not in message
        assert "pseudo.family.cutoffs" not in message
        assert "cutoffs set" not in message
