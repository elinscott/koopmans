"""``koopmans pseudos`` lists the values ``workflow.pseudo_library`` accepts."""

from __future__ import annotations

from typing import Any

from click.testing import CliRunner

from koopmans.aiida.setup import pseudos as pseudos_mod
from koopmans.cli import cli


def _run(monkeypatch: Any, installed: set[str] | None = None) -> str:
    """Return the command's output with a fixed installed set, touching no profile."""
    monkeypatch.setattr(pseudos_mod, "installed_pseudo_family_labels", lambda: installed or set())
    result = CliRunner().invoke(cli, ["pseudos"])
    assert result.exit_code == 0, result.output
    return result.output


class TestTheListingIsDerived:
    """The labels come from ``aiida-pseudo``, not from a list in this repo."""

    def test_a_relabelled_library_is_what_gets_printed(self, monkeypatch: Any) -> None:
        """Renaming what PseudoDojo reports renames what the command prints.

        A pasted list would keep printing the real labels and fail here, and
        would keep matching ``get_valid_labels`` only until PseudoDojo
        publishes its next version.
        """
        from aiida_pseudo.groups.family import PseudoDojoFamily

        monkeypatch.setattr(
            PseudoDojoFamily,
            "get_valid_labels",
            classmethod(lambda cls: ("PseudoDojo/9.9/XC/SR/standard/upf",)),
        )
        output = _run(monkeypatch)
        assert "PseudoDojo/9.9/XC/SR/standard/upf" in output
        assert "PseudoDojo/0.4/PBE/SR/standard/upf" not in output

    def test_every_label_the_libraries_publish_is_printed(self, monkeypatch: Any) -> None:
        """The listing is exactly the two libraries' labels plus the four SG15 ones."""
        from aiida_pseudo.groups.family import PseudoDojoFamily, SsspFamily

        output = _run(monkeypatch)
        printed = {
            line.split()[0] for line in output.splitlines() if line.startswith("  Pseudo")
        } | {line.split()[0] for line in output.splitlines() if line.startswith("  SSSP")}
        assert printed == set(PseudoDojoFamily.get_valid_labels()) | set(
            SsspFamily.get_valid_labels()
        )

    def test_sg15_comes_from_the_installers_own_constants(self, monkeypatch: Any) -> None:
        """SG15 has no ``aiida-pseudo`` library, so its labels track the installer."""
        monkeypatch.setattr(pseudos_mod, "_SG15_SUPPORTED_VERSIONS", {"7.7"})
        output = _run(monkeypatch)
        assert "SG15/7.7/PBE/SR" in output
        assert "SG15/1.2/PBE/SR" not in output


class TestTheListingPromisesNothingFalse:
    """A combination that does not exist must not appear."""

    def test_no_lda_full_relativistic_family_is_offered(self, monkeypatch: Any) -> None:
        """PseudoDojo publishes full-relativistic pseudos for PBE and PBEsol only.

        The combination a spin-orbit user goes looking for, since ``kcw.x`` is
        LDA-only for noncollinear runs; a grammar of the label's parts would
        promise it.
        """
        assert "LDA/FR" not in _run(monkeypatch)


class TestInstalledMarkers:
    """Installed families are marked; the rest of the listing does not change."""

    def test_only_the_installed_labels_are_marked(self, monkeypatch: Any) -> None:
        """One installed family, one marker."""
        output = _run(monkeypatch, installed={"SSSP/1.3/PBE/efficiency"})
        marked = [line for line in output.splitlines() if "[installed]" in line]
        assert len(marked) == 1
        assert marked[0].split()[0] == "SSSP/1.3/PBE/efficiency"

    def test_the_listing_works_without_a_profile(self, monkeypatch: Any) -> None:
        """Before ``koopmans install`` there is no profile, and nothing is marked."""
        from koopmans.aiida.setup import profile as profile_mod

        monkeypatch.setattr(profile_mod, "profile_exists", lambda: False)
        result = CliRunner().invoke(cli, ["pseudos"])
        assert result.exit_code == 0, result.output
        assert "[installed]" not in result.output
        assert "SSSP/1.3/PBE/efficiency" in result.output


class TestTheFieldPointsAtTheCommand:
    """The schema's advice names a command that exists."""

    def test_the_description_names_a_real_command(self) -> None:
        """``koopmans pseudos list`` was the previous package's command."""
        from koopmans.input_file.workflow import WorkflowConfig

        description = WorkflowConfig.model_fields["pseudo_library"].description or ""
        assert "koopmans pseudos" in description
        assert "koopmans pseudos list" not in description
        assert "pseudos" in cli.commands
