"""``koopmans pseudos`` lists the values ``workflow.pseudo_library`` accepts."""

from __future__ import annotations

from typing import Any

import pytest
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

    def test_every_upf_label_pseudo_dojo_publishes_is_printed(self, monkeypatch: Any) -> None:
        """The PseudoDojo section is exactly the UPF labels the library publishes.

        Filtering must drop the other three formats and nothing else, so this
        pins both directions: a filter that also lost a UPF version, and one
        that kept a psp8 label, each fail.
        """
        from aiida_pseudo.groups.family import PseudoDojoFamily

        published = set(PseudoDojoFamily.get_valid_labels())
        printed = {
            line.split()[0]
            for line in _run(monkeypatch).splitlines()
            if line.startswith("  Pseudo")
        }
        assert printed == {label for label in published if label.endswith("/upf")}
        assert printed < published

    def test_sg15_comes_from_the_installers_own_constants(self, monkeypatch: Any) -> None:
        """SG15 has no ``aiida-pseudo`` library, so its labels track the installer."""
        monkeypatch.setattr(pseudos_mod, "_SG15_VARIANTS", {"7.7": ("SR",)})
        output = _run(monkeypatch)
        assert "SG15/7.7/PBE/SR" in output
        assert "SG15/1.2/PBE/SR" not in output


class TestOnlyRunnableFamiliesAreOffered:
    """Koopmans functionals need norm-conserving pseudos, and pw.x reads UPF alone."""

    def test_sssp_is_not_offered(self, monkeypatch: Any) -> None:
        """SSSP is not uniformly norm-conserving, so koopmans cannot use it.

        The positives are pinned alongside, so a listing that printed nothing
        at all would not pass.
        """
        output = _run(monkeypatch)
        assert "SSSP" not in output
        assert "PseudoDojo/0.4/LDA/SR/standard/upf" in output
        assert "SG15/1.2/PBE/SR" in output

    def test_no_format_pw_cannot_read_is_offered(self, monkeypatch: Any) -> None:
        """``PwCalculation`` takes ``UpfData``; psp8, psml and jthxml install but cannot run.

        PseudoDojo publishes the same family in all four, so the format suffix
        is the only thing separating the offered label from the unusable one.
        """
        output = _run(monkeypatch)
        for pseudo_format in ("psp8", "psml", "jthxml"):
            assert f"/{pseudo_format}" not in output
        assert "PseudoDojo/0.4/PBE/SR/standard/upf" in output

    def test_sg15_is_offered_only_where_the_archive_publishes_it(self, monkeypatch: Any) -> None:
        """The 2020-02-06 tarball holds no ``*_ONCV_PBE_FR-1.2.upf``.

        A label built by crossing every version with every relativistic
        variant offers ``SG15/1.2/PBE/FR``, which downloads the archive and
        matches nothing in it. The two fully relativistic labels that do
        exist are pinned alongside, so dropping FR wholesale also fails.
        """
        output = _run(monkeypatch)
        assert "SG15/1.2/PBE/FR" not in output
        assert "SG15/1.0/PBE/FR" in output
        assert "SG15/1.1/PBE/FR" in output

    def test_no_lda_full_relativistic_family_is_offered(self, monkeypatch: Any) -> None:
        """PseudoDojo publishes full-relativistic pseudos for PBE and PBEsol only.

        The combination a spin-orbit user goes looking for, since ``kcw.x`` is
        LDA-only for noncollinear runs; a grammar of the label's parts would
        promise it.
        """
        assert "LDA/FR" not in _run(monkeypatch)


class TestTheSG15NotesDescribeTheArchive:
    """The element counts printed are the ones in the pinned 2020-02-06 tarball.

    Enumerating that tarball by the installer's own filename pattern gives 69
    scalar-relativistic elements at 1.0 and at 1.2, 17 at 1.1, and fully
    relativistic files at 1.0 (52 elements) and 1.1 (12) alone.
    """

    def test_no_version_is_described_as_carrying_a_handful_of_elements(
        self, monkeypatch: Any
    ) -> None:
        """1.0 is a full 69-element release, not a two-element one.

        A hand-built family holding H and O alone can sit in a profile under
        an SG15 label and look like the library; the archive is what decides.
        """
        output = _run(monkeypatch)
        assert "only H and O" not in output
        assert "1.0 covers the same 69" in output

    def test_the_fully_relativistic_coverage_is_stated(self, monkeypatch: Any) -> None:
        """Neither FR label is a whole periodic table, and users pick by element.

        Silicon has no fully relativistic SG15 pseudopotential at 1.0, so a
        note that named only 1.0 would send a spin-orbit run to a label that
        cannot supply it.
        """
        output = _run(monkeypatch)
        assert "SG15/1.0/PBE/FR carries 52 elements" in output
        assert "SG15/1.1/PBE/FR the other 12 (Si," in output

    def test_only_ecutwfc_is_asked_for(self, monkeypatch: Any) -> None:
        """``ecutrho`` follows at four times ``ecutwfc`` for norm-conserving pseudos.

        Stating both was right before koopmans derived the dual; asking for
        ``ecutrho`` now invites an input the conversion would reject.
        """
        from koopmans.aiida.conversion import NORM_CONSERVING_DUAL

        assert NORM_CONSERVING_DUAL == 4.0
        output = _run(monkeypatch)
        assert "set `ecutwfc` in your input file" in output
        assert "ecutrho" not in output


class TestInstalledMarkers:
    """Installed families are marked; the rest of the listing does not change."""

    def test_only_the_installed_labels_are_marked(self, monkeypatch: Any) -> None:
        """One installed family, one marker."""
        output = _run(monkeypatch, installed={"SG15/1.2/PBE/SR"})
        marked = [line for line in output.splitlines() if "[installed]" in line]
        assert len(marked) == 1
        assert marked[0].split()[0] == "SG15/1.2/PBE/SR"

    def test_the_listing_works_without_a_profile(self, monkeypatch: Any) -> None:
        """Before ``koopmans install`` there is no profile, and nothing is marked."""
        from koopmans.aiida.setup import profile as profile_mod

        monkeypatch.setattr(profile_mod, "profile_exists", lambda: False)
        result = CliRunner().invoke(cli, ["pseudos"])
        assert result.exit_code == 0, result.output
        assert "[installed]" not in result.output
        assert "PseudoDojo/0.4/LDA/SR/standard/upf" in result.output


class TestInstallRefusesWhatItCannotRun:
    """A label the listing does not offer is refused, not downloaded."""

    def test_an_sssp_label_is_refused_by_name(self) -> None:
        """Naming SSSP explains the constraint rather than reporting an unknown format."""
        with pytest.raises(ValueError, match="norm-conserving"):
            pseudos_mod.install_pseudo_family("SSSP/1.3/PBE/efficiency")

    def test_an_sg15_variant_the_archive_lacks_is_refused_before_downloading(self) -> None:
        """``SG15/1.2/PBE/FR`` names a version and a variant that each exist, apart.

        The guard runs before the 6 MB download, so a refusal here is also
        what keeps the test offline; a version-only check would let this
        through to unpack the archive and find nothing.
        """
        with pytest.raises(ValueError, match=r"no FR pseudopotentials at version 1\.2"):
            pseudos_mod.install_pseudo_family("SG15/1.2/PBE/FR")

    def test_a_non_upf_format_is_refused_and_upf_still_installs(self, monkeypatch: Any) -> None:
        """psp8 stops at the guard; the same family in UPF reaches the installer.

        Stubbing the downloader is what discriminates: a guard that refused
        every PseudoDojo label, and one that let psp8 through, each fail one
        half of this.
        """
        from aiida_pseudo.cli import install as install_mod
        from aiida_pseudo.data.pseudo import UpfData

        received: dict[str, Any] = {}

        class _Family:
            def set_default_stringency(self, stringency: str) -> None:
                """Record the stringency the installer asks for."""
                received["stringency"] = stringency

        def _install(**kwargs: Any) -> _Family:
            received.update(kwargs)
            return _Family()

        monkeypatch.setattr(install_mod, "download_pseudo_dojo", lambda **kwargs: None)
        monkeypatch.setattr(install_mod, "install_pseudo_dojo", _install)

        with pytest.raises(ValueError, match="UPF"):
            pseudos_mod.install_pseudo_family("PseudoDojo/0.4/PBE/SR/standard/psp8")
        assert received == {}

        pseudos_mod.install_pseudo_family("PseudoDojo/0.4/PBE/SR/standard/upf")
        assert received["pseudo_type"] is UpfData


class TestTheFieldPointsAtTheCommand:
    """The schema's advice names a command that exists."""

    def test_the_description_names_a_real_command(self) -> None:
        """``koopmans pseudos list`` was the previous package's command."""
        from koopmans.input_file.workflow import WorkflowConfig

        description = WorkflowConfig.model_fields["pseudo_library"].description or ""
        assert "koopmans pseudos" in description
        assert "koopmans pseudos list" not in description
        assert "pseudos" in cli.commands
