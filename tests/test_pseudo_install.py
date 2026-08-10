"""Tests for the SG15 pseudopotential family installer.

The SG15 archive is never downloaded: the ``offline_sg15_archive`` fixture
serves a synthetic tarball through a patched ``urlopen``, with the pinned
checksum swapped for that tarball's own.
"""

from __future__ import annotations

from typing import Any

SG15_LABEL = "SG15/1.2/PBE/SR"


def _installed(label: str) -> Any:
    """Install the family and return it."""
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    from koopmans.aiida.setup.pseudos import install_pseudo_family

    install_pseudo_family(label)
    return PseudoPotentialFamily.collection.get(label=label)


class TestInstallSg15Family:
    """The class the installer builds, and the pseudos it selects."""

    def test_installs_a_family_that_recommends_no_cutoffs(
        self, aiida_profile_clean: Any, offline_sg15_archive: dict[str, str]
    ) -> None:
        """SG15 publishes no cutoffs, so the family must not claim to have any."""
        from aiida_pseudo.groups.family import PseudoPotentialFamily
        from aiida_pseudo.groups.mixins import RecommendedCutoffMixin

        family = _installed(SG15_LABEL)

        assert type(family) is PseudoPotentialFamily
        assert not isinstance(family, RecommendedCutoffMixin)

    def test_installs_only_the_labelled_version_and_variant(
        self, aiida_profile_clean: Any, offline_sg15_archive: dict[str, str]
    ) -> None:
        """The other revisions and the fully relativistic members stay out.

        1.2 is a complete release, so nothing is laid over it: the silicon it
        installs is 1.2's own, not the 1.1 file sitting beside it.
        """
        family = _installed(SG15_LABEL)

        assert {pseudo.element for pseudo in family.nodes} == {"Si", "O"}
        assert family.count() == 2
        assert family.get_pseudo("Si").get_content() == offline_sg15_archive["Si_ONCV_PBE-1.2.upf"]


class TestVersion11IsComposedOver10:
    """SG15 published 1.1 as a delta of 1.0, so the label installs both."""

    def test_the_fully_relativistic_set_gains_the_elements_1_0_lacks(
        self, aiida_profile_clean: Any, offline_sg15_archive: dict[str, str]
    ) -> None:
        """Silicon is fully relativistic only at 1.1, oxygen only at 1.0.

        Composing is what puts the two under one label; installing 1.1 alone
        would drop the oxygen a run also needs. The 1.0 half is the control:
        it must stay the release SG15 published, without the silicon 1.1 adds.
        """
        composed = _installed("SG15/1.1/PBE/FR")
        pure = _installed("SG15/1.0/PBE/FR")

        assert set(composed.elements) == {"Si", "O"}
        assert set(pure.elements) == {"O"}

    def test_a_revised_element_comes_from_the_1_1_file(
        self, aiida_profile_clean: Any, offline_sg15_archive: dict[str, str]
    ) -> None:
        """Both revisions of silicon match the label, and the newer one wins.

        Element coverage cannot see this: 1.1 revises silicon rather than
        adding it, so an overlay that kept the older file installs the same
        69 elements carrying the pseudopotential the release replaced. Oxygen,
        which 1.1 leaves alone, must still arrive from 1.0.
        """
        family = _installed("SG15/1.1/PBE/SR")

        assert family.get_pseudo("Si").get_content() == offline_sg15_archive["Si_ONCV_PBE-1.1.upf"]
        assert family.get_pseudo("O").get_content() == offline_sg15_archive["O_ONCV_PBE-1.0.upf"]

    def test_the_installed_files_keep_the_names_that_date_them(
        self, aiida_profile_clean: Any, offline_sg15_archive: dict[str, str]
    ) -> None:
        """A composed family holds files from two revisions, and says so.

        The filename is the only place the revision survives, and `koopmans
        pseudos` promises a 1.1 family holds ``-1.0.upf`` files as well.
        """
        family = _installed("SG15/1.1/PBE/SR")

        assert {pseudo.filename for pseudo in family.nodes} == {
            "Si_ONCV_PBE-1.1.upf",
            "O_ONCV_PBE-1.0.upf",
        }

    def test_1_0_installs_its_own_revision_alone(
        self, aiida_profile_clean: Any, offline_sg15_archive: dict[str, str]
    ) -> None:
        """1.0 is complete on its own terms, so nothing is laid over it.

        Composition running one version too far would hand a user who asked
        for the original revision the 1.1 silicon.
        """
        family = _installed("SG15/1.0/PBE/SR")

        assert family.get_pseudo("Si").get_content() == offline_sg15_archive["Si_ONCV_PBE-1.0.upf"]
        assert set(family.elements) == {"Si", "O"}


class TestTheInstallerRefusesWhatTheArchiveLacks:
    """Each guard names what to change, before any pseudo lands in the profile."""

    def test_a_non_pbe_functional_is_refused(self, aiida_profile_clean: Any) -> None:
        """The archive is PBE-only, so the label's functional part must be PBE."""
        import pytest

        from koopmans.aiida.setup.pseudos import install_pseudo_family

        with pytest.raises(ValueError, match="SG15 only provides PBE"):
            install_pseudo_family("SG15/1.2/LDA/SR")

    def test_an_unpackaged_version_is_refused(self, aiida_profile_clean: Any) -> None:
        """A version the archive does not carry is named alongside the ones it does."""
        import pytest

        from koopmans.aiida.setup.pseudos import install_pseudo_family

        with pytest.raises(ValueError, match=r"version '9.9' is not packaged"):
            install_pseudo_family("SG15/9.9/PBE/SR")

    def test_a_missing_variant_is_refused_before_the_download(
        self, aiida_profile_clean: Any, monkeypatch: Any
    ) -> None:
        """1.2 publishes no FR members, and no bytes move before the refusal."""
        import urllib.request

        import pytest

        from koopmans.aiida.setup.pseudos import install_pseudo_family

        def _no_download(url: str) -> None:
            raise AssertionError("the guard must fire before urlopen")

        monkeypatch.setattr(urllib.request, "urlopen", _no_download)
        with pytest.raises(ValueError, match=r"publishes no FR pseudopotentials at version 1\.2"):
            install_pseudo_family("SG15/1.2/PBE/FR")

    def test_a_tampered_archive_is_refused(
        self, aiida_profile_clean: Any, offline_sg15_archive: dict[str, str], monkeypatch: Any
    ) -> None:
        """A checksum other than the pinned one stops the install cold."""
        import pytest

        from koopmans.aiida.setup import pseudos as pseudos_mod
        from koopmans.aiida.setup.pseudos import install_pseudo_family

        monkeypatch.setattr(pseudos_mod, "_SG15_ARCHIVE_SHA256", "0" * 64)
        with pytest.raises(ValueError, match="checksum mismatch"):
            install_pseudo_family(SG15_LABEL)


def test_ensure_installs_an_absent_family(
    aiida_profile_clean: Any, offline_sg15_archive: dict[str, str]
) -> None:
    """``ensure_pseudo_family_installed`` falls through to the installer when nothing matches."""
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    from koopmans.aiida.setup.pseudos import ensure_pseudo_family_installed

    ensure_pseudo_family_installed(SG15_LABEL)

    assert PseudoPotentialFamily.collection.get(label=SG15_LABEL).count() == 2


def test_an_archive_with_no_matching_members_is_named(
    aiida_profile_clean: Any, offline_sg15_archive: dict[str, str], monkeypatch: Any
) -> None:
    """A version the table offers but the archive lacks fails naming the label.

    The pre-download guard reads the table, so a table entry with no archive
    members behind it is only caught here — the layout-changed error is what
    a user would see if upstream re-released the tarball differently.
    """
    import pytest

    from koopmans.aiida.setup import pseudos as pseudos_mod
    from koopmans.aiida.setup.pseudos import install_pseudo_family

    monkeypatch.setitem(pseudos_mod._SG15_VARIANTS, "1.3", {"SR": ("1.3",)})
    with pytest.raises(ValueError, match=r"No UPF files matched .SG15/1\.3/PBE/SR."):
        install_pseudo_family("SG15/1.3/PBE/SR")


def test_installed_labels_come_from_the_profile(
    aiida_profile_clean: Any, offline_sg15_archive: dict[str, str], monkeypatch: Any
) -> None:
    """The installed-label set reflects what the profile holds, not a cache."""
    from koopmans.aiida.setup import profile as profile_mod
    from koopmans.aiida.setup import pseudos as pseudos_mod

    monkeypatch.setattr(profile_mod, "profile_exists", lambda: True)
    monkeypatch.setattr(profile_mod, "load_koopmans_profile", lambda: None)

    assert SG15_LABEL not in pseudos_mod.installed_pseudo_family_labels()
    _installed(SG15_LABEL)
    assert SG15_LABEL in pseudos_mod.installed_pseudo_family_labels()


def test_an_unrecognized_label_names_the_two_formats(aiida_profile_clean: Any) -> None:
    """A label matching neither library's format is refused naming both."""
    import pytest

    from koopmans.aiida.setup.pseudos import install_pseudo_family

    with pytest.raises(ValueError, match="Unrecognized pseudo family format"):
        install_pseudo_family("my-own-pseudos")
