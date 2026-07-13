"""Pseudopotential family installers (PseudoDojo / SSSP / SG15)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def ensure_pseudo_family_installed(pseudo_family: str) -> None:
    """Ensure a pseudopotential family is installed, installing it if necessary.

    Supports PseudoDojo families with labels like:
        'PseudoDojo/0.4/LDA/SR/standard/upf'

    SSSP families with labels like:
        'SSSP/1.3/PBEsol/efficiency'

    And SG15 ONCV families with labels like:
        'SG15/1.2/PBE/SR'

    Raises:
        ValueError: If the family format is not recognized or installation fails.
    """
    from aiida.common.exceptions import NotExistent
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    try:
        PseudoPotentialFamily.collection.get(label=pseudo_family)
        logger.debug("Pseudo family '%s' already installed", pseudo_family)
        return
    except NotExistent:
        pass

    logger.info("Installing pseudo family '%s'...", pseudo_family)
    install_pseudo_family(pseudo_family)
    logger.info("Successfully installed pseudo family '%s'", pseudo_family)


def install_pseudo_family(pseudo_family: str) -> None:
    """Install a pseudopotential family. Parse the label and dispatch."""
    parts = pseudo_family.split("/")

    if parts[0] == "PseudoDojo" and len(parts) == 6:
        _install_pseudo_dojo_family(pseudo_family, parts)
    elif parts[0] == "SSSP" and len(parts) == 4:
        _install_sssp_family(pseudo_family, parts)
    elif parts[0] == "SG15" and len(parts) == 4:
        _install_sg15_family(pseudo_family, parts)
    else:
        raise ValueError(
            f"Unrecognized pseudo family format: '{pseudo_family}'. "
            "Expected 'PseudoDojo/version/functional/relativistic/protocol/format', "
            "'SSSP/version/functional/protocol', "
            "or 'SG15/version/functional/relativistic'."
        )


def _install_pseudo_dojo_family(label: str, parts: list[str]) -> None:
    """Install a PseudoDojo pseudopotential family."""
    import contextlib
    import io
    import warnings

    from aiida_pseudo.cli.install import download_pseudo_dojo, install_pseudo_dojo
    from aiida_pseudo.data.pseudo import JthXmlData, PsmlData, Psp8Data, UpfData
    from aiida_pseudo.groups.family import PseudoDojoConfiguration

    _, version, functional, relativistic, protocol, pseudo_format = parts

    format_to_type = {
        "upf": UpfData,
        "psp8": Psp8Data,
        "psml": PsmlData,
        "jthxml": JthXmlData,
    }

    pseudo_type = format_to_type.get(pseudo_format.lower())
    if pseudo_type is None:
        raise ValueError(
            f"Unknown pseudo format '{pseudo_format}'. "
            f"Supported formats: {list(format_to_type.keys())}"
        )

    configuration = PseudoDojoConfiguration(
        version=version,
        functional=functional,
        relativistic=relativistic,
        protocol=protocol,
        pseudo_format=pseudo_format,
    )

    click.echo(f"  Downloading '{label}' pseudopotentials")

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath_archive = Path(tmpdir) / "archive.tgz"
        filepath_metadata = Path(tmpdir) / "metadata.tgz"

        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter("ignore")

            download_pseudo_dojo(
                configuration=configuration,
                filepath_archive=filepath_archive,
                filepath_metadata=filepath_metadata,
                traceback=False,
            )

            family = install_pseudo_dojo(
                configuration=configuration,
                filepath_archive=filepath_archive,
                filepath_metadata=filepath_metadata,
                pseudo_type=pseudo_type,
                label=label,
                traceback=False,
            )

        family.set_default_stringency("normal")


def _install_sssp_family(label: str, parts: list[str]) -> None:
    """Install an SSSP pseudopotential family."""
    import contextlib
    import io
    import warnings

    from aiida_pseudo.cli.install import download_sssp, install_sssp
    from aiida_pseudo.groups.family import SsspConfiguration

    _, version, functional, protocol = parts

    configuration = SsspConfiguration(
        version=version,
        functional=functional,
        protocol=protocol,
    )

    click.echo(f"  Downloading pseudopotentials for '{label}'...")

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath_archive = Path(tmpdir) / "archive.tar.gz"
        filepath_metadata = Path(tmpdir) / "metadata.json"

        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter("ignore")

            download_sssp(
                configuration=configuration,
                filepath_archive=filepath_archive,
                filepath_metadata=filepath_metadata,
                traceback=False,
            )

            install_sssp(
                filepath_archive=filepath_archive,
                filepath_metadata=filepath_metadata,
                label=label,
                traceback=False,
            )

    from aiida_pseudo.groups.family import SsspFamily

    family = SsspFamily.collection.get(label=label)
    click.echo(f"  Successfully installed '{label}' ({family.count()} pseudopotentials)")


# SG15 ONCV is published as a single frozen tarball on quantum-simulation.org. It
# bundles every version x relativistic variant in one flat archive; the label's
# version/relativistic parts select which subset of UPFs to install. There is no
# upstream ``aiida-pseudo`` installer for SG15, so we install as a plain
# ``CutoffsPseudoPotentialFamily`` so recommended cutoffs can be attached later
# via ``family.set_cutoffs`` without a reinstall.
_SG15_ARCHIVE_URL = (
    "http://www.quantum-simulation.org/potentials/sg15_oncv/sg15_oncv_upf_2020-02-06.tar.gz"
)
_SG15_ARCHIVE_SHA256 = "3f3bd74aa5d6e0b038218a6051bb99ed9469dc03d0f05b3ec8a523f0f7a7dff0"
_SG15_SUPPORTED_VERSIONS = {"1.0", "1.2"}
_SG15_SUPPORTED_RELATIVISTIC = {"SR", "FR"}


def _install_sg15_family(label: str, parts: list[str]) -> None:
    """Install an SG15 ONCV pseudopotential family."""
    import hashlib
    import io
    import re
    import tarfile
    import urllib.request

    from aiida_pseudo.data.pseudo import UpfData
    from aiida_pseudo.groups.family import CutoffsPseudoPotentialFamily

    _, version, functional, relativistic = parts

    if functional != "PBE":
        raise ValueError(f"SG15 only provides PBE pseudopotentials; got functional='{functional}'.")
    if version not in _SG15_SUPPORTED_VERSIONS:
        raise ValueError(
            f"SG15 version '{version}' is not packaged in the 2020-02-06 archive. "
            f"Supported versions: {sorted(_SG15_SUPPORTED_VERSIONS)}."
        )
    if relativistic not in _SG15_SUPPORTED_RELATIVISTIC:
        raise ValueError(
            f"SG15 relativistic variant '{relativistic}' is not supported. "
            f"Expected one of: {sorted(_SG15_SUPPORTED_RELATIVISTIC)}."
        )

    fr_suffix = "_FR" if relativistic == "FR" else ""
    filename_re = re.compile(
        rf"^(?P<element>[A-Z][a-z]?)_ONCV_PBE{fr_suffix}-{re.escape(version)}\.upf$"
    )

    click.echo(f"  Downloading '{label}' pseudopotentials")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        with urllib.request.urlopen(_SG15_ARCHIVE_URL) as response:  # noqa: S310
            archive_bytes = response.read()

        digest = hashlib.sha256(archive_bytes).hexdigest()
        if digest != _SG15_ARCHIVE_SHA256:
            raise ValueError(
                f"SG15 archive checksum mismatch: got {digest}, "
                f"expected {_SG15_ARCHIVE_SHA256}. Upstream may have re-released "
                f"{_SG15_ARCHIVE_URL}; pin a new hash after verifying the contents."
            )

        flat = tmp / "flat"
        flat.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                match = filename_re.match(Path(member.name).name)
                if match is None:
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                (flat / f"{match.group('element')}.upf").write_bytes(extracted.read())

        if not any(flat.iterdir()):
            raise ValueError(
                f"No UPF files matched '{label}' in {_SG15_ARCHIVE_URL}. "
                "The archive layout may have changed."
            )

        family = CutoffsPseudoPotentialFamily.create_from_folder(flat, label, pseudo_type=UpfData)

    click.echo(f"  Successfully installed '{label}' ({family.count()} pseudopotentials)")
