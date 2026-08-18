"""Install SG15 ONCV families from the published tarball.

SG15 ONCV is published as a single frozen tarball on quantum-simulation.org,
one flat directory of ``<element>_ONCV_PBE[_FR]-<version>.upf`` files; the
label's version/relativistic parts select which of them to install. There is
no upstream ``aiida-pseudo`` installer for SG15, so we build the family
ourselves. SG15 publishes no recommended cutoffs, so it is a plain
``PseudoPotentialFamily`` and ``ecutwfc`` comes from the input file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import click

ARCHIVE_URL = (
    "http://www.quantum-simulation.org/potentials/sg15_oncv/sg15_oncv_upf_2020-02-06.tar.gz"
)
ARCHIVE_SHA256 = "3f3bd74aa5d6e0b038218a6051bb99ed9469dc03d0f05b3ec8a523f0f7a7dff0"

# Which archive files each label installs: the version's own, and for a delta
# release the versions it is layered on, oldest first. SG15 published 1.1 as a
# delta of 1.0 -- 17 revised scalar-relativistic files, 12 fully relativistic --
# so ``SG15/1.1`` is 1.0 with those files laid over it, while 1.0 and 1.2 are
# each one complete revision. Composed, 1.1 holds 69 SR and 64 FR elements
# against 1.0's 69 and 52.
#
# SG15 also revises element by element, so a version does not carry both
# relativistic variants just because it carries one: the 2020-02-06 archive
# publishes nothing fully relativistic at 1.2. A label naming a variant the
# archive lacks would install nothing, so the offered labels are read from here
# too.
VARIANTS: dict[str, dict[str, tuple[str, ...]]] = {
    "1.0": {"SR": ("1.0",), "FR": ("1.0",)},
    "1.1": {"SR": ("1.0", "1.1"), "FR": ("1.0", "1.1")},
    "1.2": {"SR": ("1.2",)},
}

NOTES = (
    "SG15/1.2 is the newest scalar-relativistic set and covers all 69 elements; "
    "1.0 covers the same 69 at the original revision.",
    "SG15 published 1.1 as a delta of 1.0, revising 17 elements, so koopmans "
    "composes it: the 1.1 label installs the 1.0 files with the 1.1 ones over "
    "them. Files keep their archive names, so a 1.1 family holds -1.0.upf files "
    "as well.",
    "Name SG15/1.1/PBE/FR for fully relativistic runs: composed it covers 64 "
    "elements against 1.0's 52, and 1.2 publishes none. Ba, Be, Bi, Li and Ne "
    "have no fully relativistic SG15 pseudopotential at any version.",
    "SG15 recommends no cutoffs, so set `ecutwfc` in your input file.",
)


def available_labels() -> list[str]:
    """Return every SG15 label the 2020-02-06 archive can supply, sorted."""
    return sorted(
        f"SG15/{version}/PBE/{relativistic}"
        for version, relativistic_variants in VARIANTS.items()
        for relativistic in relativistic_variants
    )


def _select_files(
    archive_bytes: bytes, source_versions: tuple[str, ...], fr_suffix: str
) -> dict[str, tuple[str, bytes]]:
    """Return one archive file per element, keyed by element, as ``(name, content)``.

    ``source_versions`` runs oldest first; where an element appears in more
    than one, the later file wins.
    """
    import io
    import re
    import tarfile

    versions = "|".join(re.escape(source) for source in source_versions)
    filename_re = re.compile(
        rf"^(?P<element>[A-Z][a-z]?)_ONCV_PBE{fr_suffix}-(?P<version>{versions})\.upf$"
    )
    revision = {source: index for index, source in enumerate(source_versions)}

    selected: dict[str, tuple[int, str, bytes]] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            filename = Path(member.name).name
            match = filename_re.match(filename)
            if match is None:
                continue
            element = match.group("element")
            index = revision[match.group("version")]
            if element in selected and selected[element][0] > index:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            selected[element] = (index, filename, extracted.read())

    return {element: (name, content) for element, (_, name, content) in selected.items()}


def install(label: str, parts: list[str]) -> None:
    """Install an SG15 ONCV pseudopotential family.

    Each pseudopotential keeps its archive filename, which names the revision
    it came from.

    Raises:
        ValueError: If the label names a functional, version or relativistic
            variant the 2020-02-06 archive does not publish.
    """
    import hashlib
    import urllib.request

    from aiida_pseudo.data.pseudo import UpfData
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    _, version, functional, relativistic = parts

    if functional != "PBE":
        raise ValueError(f"SG15 only provides PBE pseudopotentials; got functional='{functional}'.")
    relativistic_variants = VARIANTS.get(version)
    if relativistic_variants is None:
        raise ValueError(
            f"SG15 version '{version}' is not packaged in the 2020-02-06 archive. "
            f"Supported versions: {sorted(VARIANTS)}."
        )
    if relativistic not in relativistic_variants:
        raise ValueError(
            f"SG15 publishes no {relativistic} pseudopotentials at version {version}; "
            f"at {version} the archive carries {', '.join(relativistic_variants)}. "
            "Run `koopmans pseudos` for every label koopmans accepts."
        )
    source_versions = relativistic_variants[relativistic]
    fr_suffix = "_FR" if relativistic == "FR" else ""

    click.echo(f"  Downloading '{label}' pseudopotentials")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        with urllib.request.urlopen(ARCHIVE_URL) as response:  # noqa: S310
            archive_bytes = response.read()

        digest = hashlib.sha256(archive_bytes).hexdigest()
        if digest != ARCHIVE_SHA256:
            raise ValueError(
                f"SG15 archive checksum mismatch: got {digest}, "
                f"expected {ARCHIVE_SHA256}. Upstream may have re-released "
                f"{ARCHIVE_URL}; pin a new hash after verifying the contents."
            )

        selected = _select_files(archive_bytes, source_versions, fr_suffix)

        flat = tmp / "flat"
        flat.mkdir()
        for filename, content in selected.values():
            (flat / filename).write_bytes(content)

        if not any(flat.iterdir()):
            raise ValueError(
                f"No UPF files matched '{label}' in {ARCHIVE_URL}. "
                "The archive layout may have changed."
            )

        family = PseudoPotentialFamily.create_from_folder(flat, label, pseudo_type=UpfData)

    click.echo(f"  Successfully installed '{label}' ({family.count()} pseudopotentials)")
