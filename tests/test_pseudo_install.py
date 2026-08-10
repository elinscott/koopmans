"""Tests for the SG15 pseudopotential family installer.

The SG15 archive is never downloaded: a synthetic tarball built from the
``fake_upf_content`` streams is served through a patched ``urlopen``, with the
pinned checksum swapped for that tarball's own.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import urllib.request
from typing import Any

import pytest

from tests.fixtures import fake_upf_content

SG15_LABEL = "SG15/1.2/PBE/SR"

# One member per (element, version, relativistic variant) the flat archive
# bundles, so the installer has to select the label's subset rather than take
# whatever it finds.
_ARCHIVE_MEMBERS = {
    "sg15_oncv_upf_2020-02-06/Si_ONCV_PBE-1.2.upf": ("Si", 4.0),
    "sg15_oncv_upf_2020-02-06/O_ONCV_PBE-1.2.upf": ("O", 6.0),
    "sg15_oncv_upf_2020-02-06/Si_ONCV_PBE-1.0.upf": ("Si", 4.0),
    "sg15_oncv_upf_2020-02-06/Si_ONCV_PBE_FR-1.2.upf": ("Si", 4.0),
}


def _synthetic_archive() -> bytes:
    """Return a gzipped tarball shaped like the published SG15 archive."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, (element, z_valence) in _ARCHIVE_MEMBERS.items():
            payload = fake_upf_content(element, z_valence).encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


@pytest.fixture
def offline_sg15_archive(monkeypatch: pytest.MonkeyPatch) -> bytes:
    """Serve the synthetic archive from ``urlopen`` and pin its checksum."""
    from koopmans.aiida.setup import pseudos

    archive = _synthetic_archive()
    monkeypatch.setattr(pseudos, "_SG15_ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(urllib.request, "urlopen", lambda url: io.BytesIO(archive))
    return archive


class TestInstallSg15Family:
    """The class the installer builds, and the pseudos it selects."""

    def test_installs_a_family_that_recommends_no_cutoffs(
        self, aiida_profile_clean: Any, offline_sg15_archive: bytes
    ) -> None:
        """SG15 publishes no cutoffs, so the family must not claim to have any."""
        from aiida_pseudo.groups.family import PseudoPotentialFamily
        from aiida_pseudo.groups.mixins import RecommendedCutoffMixin

        from koopmans.aiida.setup.pseudos import install_pseudo_family

        install_pseudo_family(SG15_LABEL)

        family = PseudoPotentialFamily.collection.get(label=SG15_LABEL)
        assert type(family) is PseudoPotentialFamily
        assert not isinstance(family, RecommendedCutoffMixin)

    def test_installs_only_the_labelled_version_and_variant(
        self, aiida_profile_clean: Any, offline_sg15_archive: bytes
    ) -> None:
        """The 1.0 and fully relativistic members of the archive stay out."""
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        from koopmans.aiida.setup.pseudos import install_pseudo_family

        install_pseudo_family(SG15_LABEL)

        family = PseudoPotentialFamily.collection.get(label=SG15_LABEL)
        assert {pseudo.element for pseudo in family.nodes} == {"Si", "O"}
        assert family.count() == 2
