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
