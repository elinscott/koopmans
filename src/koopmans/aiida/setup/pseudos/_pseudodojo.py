"""Install PseudoDojo families through ``aiida-pseudo``'s own downloader."""

from __future__ import annotations

import tempfile
from pathlib import Path

import click

# PseudoDojo publishes each family in four formats. ``PwCalculation`` declares
# its ``pseudos`` input with ``valid_type=(LegacyUpfData, UpfData)``, so the
# psp8, psml and jthxml families install but cannot be handed to pw.x. The
# filter also settles what the UPF families contain: PseudoDojo's PAW sets are
# published as jthxml alone, so every label left is norm-conserving ONCV.
PSEUDO_DOJO_FORMAT = "upf"


def available_labels() -> list[str]:
    """Return every PseudoDojo label koopmans accepts, sorted.

    Asked of ``aiida-pseudo`` rather than written down, so a newly published
    version needs no edit here.
    """
    from aiida_pseudo.groups.family import PseudoDojoFamily

    return sorted(
        label
        for label in PseudoDojoFamily.get_valid_labels()
        if label.rsplit("/", 1)[-1].lower() == PSEUDO_DOJO_FORMAT
    )


def install(label: str, parts: list[str]) -> None:
    """Install a PseudoDojo pseudopotential family.

    Raises:
        ValueError: If the label asks for a format other than ``upf``.
    """
    import contextlib
    import io
    import warnings

    from aiida_pseudo.cli.install import download_pseudo_dojo, install_pseudo_dojo
    from aiida_pseudo.data.pseudo import UpfData
    from aiida_pseudo.groups.family import PseudoDojoConfiguration

    _, version, functional, relativistic, protocol, pseudo_format = parts

    if pseudo_format.lower() != PSEUDO_DOJO_FORMAT:
        raise ValueError(
            f"PseudoDojo publishes '{label}' in the {pseudo_format} format, which "
            "pw.x cannot read; it takes UPF only. End the label with "
            f"'/{PSEUDO_DOJO_FORMAT}'."
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
                pseudo_type=UpfData,
                label=label,
                traceback=False,
            )

        family.set_default_stringency("normal")
