"""Pseudopotential family installers (PseudoDojo / SG15)."""

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


def pseudo_family_has_cutoffs(pseudo_family: str) -> bool:
    """Report whether an installed family publishes recommended cutoffs.

    True only if the family defines at least one cutoff stringency; without one
    ``get_recommended_cutoffs`` has nothing to return.

    Raises:
        NotExistent: If the family is not installed.
    """
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    family = PseudoPotentialFamily.collection.get(label=pseudo_family)
    stringencies = getattr(family, "get_cutoff_stringencies", None)
    return stringencies is not None and bool(stringencies())


_LIBRARY_NOTES = {
    "SG15": (
        "SG15/1.2 is the newest scalar-relativistic set and covers all 69 elements; "
        "1.0 covers the same 69 at the original revision, and 1.1 only the 17 "
        "revised in between.",
        "There is no fully relativistic set at 1.2: SG15/1.0/PBE/FR carries 52 "
        "elements and SG15/1.1/PBE/FR the other 12 (Si, P, S and nine more p-block "
        "elements).",
        "SG15 recommends no cutoffs, so set `ecutwfc` in your input file.",
    ),
}


def available_pseudo_families() -> dict[str, list[str]]:
    """Return every valid ``pseudo_library`` label, sorted, keyed by library.

    Every label is norm-conserving and in UPF format: Koopmans functionals are
    defined for norm-conserving pseudopotentials, and ``pw.x`` reads UPF alone.
    PseudoDojo's labels are asked of ``aiida-pseudo`` rather than written down,
    so a newly published version appears here without an edit; its non-UPF
    formats are dropped. SG15 is not an ``aiida-pseudo`` library, so its labels
    are enumerated here.

    No profile is needed.
    """
    from aiida_pseudo.groups.family import PseudoDojoFamily

    return {
        "PseudoDojo": sorted(
            label
            for label in PseudoDojoFamily.get_valid_labels()
            if label.rsplit("/", 1)[-1].lower() == _PSEUDO_DOJO_FORMAT
        ),
        "SG15": sorted(
            f"SG15/{version}/PBE/{relativistic}"
            for version, relativistic_variants in _SG15_VARIANTS.items()
            for relativistic in relativistic_variants
        ),
    }


def installed_pseudo_family_labels() -> set[str]:
    """Return the labels of the families installed in the koopmans profile.

    Empty when no profile exists yet.
    """
    from aiida import orm

    from .profile import load_koopmans_profile, profile_exists

    if not profile_exists():
        return set()

    from aiida_pseudo.groups.family import PseudoPotentialFamily

    load_koopmans_profile()
    query = orm.QueryBuilder().append(PseudoPotentialFamily, project=["label"])
    return {label for (label,) in query.all()}


def list_pseudo_families() -> None:
    """Print every value ``workflow.pseudo_library`` accepts, marking the installed ones."""
    available = available_pseudo_families()
    installed = installed_pseudo_family_labels()

    width = max(len(label) for labels in available.values() for label in labels)
    for library, labels in available.items():
        click.echo(f"\n{library}")
        for label in labels:
            mark = "  [installed]" if label in installed else ""
            click.echo(f"  {label.ljust(width)}{mark}".rstrip())
        notes = _LIBRARY_NOTES.get(library)
        if notes:
            click.echo("")
            for note in notes:
                click.echo(f"  {note}")

    click.echo("\nEvery family listed is norm-conserving and in UPF format, which is what")
    click.echo("Koopmans functionals and `pw.x` require.")
    click.echo("\nName one of these as `pseudo_library` in the input file's `workflow` block, and")
    click.echo("koopmans installs it the first time it is used. To use pseudopotentials of your")
    click.echo("own, run `aiida-pseudo install family <directory> <label>` and name that label.")


def install_pseudo_family(pseudo_family: str) -> None:
    """Install a pseudopotential family. Parse the label and dispatch.

    Raises:
        ValueError: If the label names a family koopmans cannot run with.
    """
    parts = pseudo_family.split("/")

    if parts[0] == "PseudoDojo" and len(parts) == 6:
        _install_pseudo_dojo_family(pseudo_family, parts)
    elif parts[0] == "SG15" and len(parts) == 4:
        _install_sg15_family(pseudo_family, parts)
    elif parts[0] == "SSSP":
        raise ValueError(
            f"'{pseudo_family}' is an SSSP family. SSSP mixes ultrasoft, PAW and "
            "norm-conserving pseudopotentials, and Koopmans functionals are defined "
            "for norm-conserving ones. Name a PseudoDojo or SG15 family instead; run "
            "`koopmans pseudos` for the full list."
        )
    else:
        raise ValueError(
            f"Unrecognized pseudo family format: '{pseudo_family}'. "
            "Expected 'PseudoDojo/version/functional/relativistic/protocol/upf' "
            "or 'SG15/version/functional/relativistic'. "
            "Run `koopmans pseudos` for every label koopmans accepts."
        )


# PseudoDojo publishes each family in four formats. ``PwCalculation`` declares
# its ``pseudos`` input with ``valid_type=(LegacyUpfData, UpfData)``, so the
# psp8, psml and jthxml families install but cannot be handed to pw.x.
_PSEUDO_DOJO_FORMAT = "upf"


def _install_pseudo_dojo_family(label: str, parts: list[str]) -> None:
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

    if pseudo_format.lower() != _PSEUDO_DOJO_FORMAT:
        raise ValueError(
            f"PseudoDojo publishes '{label}' in the {pseudo_format} format, which "
            "pw.x cannot read; it takes UPF only. End the label with "
            f"'/{_PSEUDO_DOJO_FORMAT}'."
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


# SG15 ONCV is published as a single frozen tarball on quantum-simulation.org,
# one flat directory of ``<element>_ONCV_PBE[_FR]-<version>.upf`` files; the
# label's version/relativistic parts select which of them to install. There is
# no upstream ``aiida-pseudo`` installer for SG15, so we build the family
# ourselves. SG15 publishes no recommended cutoffs, so it is a plain
# ``PseudoPotentialFamily`` and ``ecutwfc`` comes from the input file.
_SG15_ARCHIVE_URL = (
    "http://www.quantum-simulation.org/potentials/sg15_oncv/sg15_oncv_upf_2020-02-06.tar.gz"
)
_SG15_ARCHIVE_SHA256 = "3f3bd74aa5d6e0b038218a6051bb99ed9469dc03d0f05b3ec8a523f0f7a7dff0"

# SG15 revises element by element, so a version does not carry both relativistic
# variants just because it carries one: the 2020-02-06 archive holds 69 SR
# elements at 1.0 and at 1.2, 17 at 1.1, and fully relativistic files at 1.0 (52
# elements) and 1.1 (12) only. A label naming a variant the archive lacks would
# install nothing, so the offered labels are read from here.
_SG15_VARIANTS: dict[str, tuple[str, ...]] = {
    "1.0": ("SR", "FR"),
    "1.1": ("SR", "FR"),
    "1.2": ("SR",),
}


def _install_sg15_family(label: str, parts: list[str]) -> None:
    """Install an SG15 ONCV pseudopotential family.

    Raises:
        ValueError: If the label names a functional, version or relativistic
            variant the 2020-02-06 archive does not publish.
    """
    import hashlib
    import io
    import re
    import tarfile
    import urllib.request

    from aiida_pseudo.data.pseudo import UpfData
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    _, version, functional, relativistic = parts

    if functional != "PBE":
        raise ValueError(f"SG15 only provides PBE pseudopotentials; got functional='{functional}'.")
    relativistic_variants = _SG15_VARIANTS.get(version)
    if relativistic_variants is None:
        raise ValueError(
            f"SG15 version '{version}' is not packaged in the 2020-02-06 archive. "
            f"Supported versions: {sorted(_SG15_VARIANTS)}."
        )
    if relativistic not in relativistic_variants:
        raise ValueError(
            f"SG15 publishes no {relativistic} pseudopotentials at version {version}; "
            f"at {version} the archive carries {', '.join(relativistic_variants)}. "
            "Run `koopmans pseudos` for every label koopmans accepts."
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

        family = PseudoPotentialFamily.create_from_folder(flat, label, pseudo_type=UpfData)

    click.echo(f"  Successfully installed '{label}' ({family.count()} pseudopotentials)")
