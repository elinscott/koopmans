"""Resolve a ``pseudo_library`` label to an installed pseudopotential family.

Each library koopmans can download has a module of its own
(:mod:`._pseudodojo`, :mod:`._sg15`) exposing ``available_labels``,
``install`` and, where it has something to say, ``NOTES``. This module holds
what is common to all of them: the label registry, the profile query, and the
listing ``koopmans pseudos`` prints.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import click

from . import _pseudodojo, _sg15

if TYPE_CHECKING:
    from aiida import orm

logger = logging.getLogger(__name__)

# Libraries koopmans can download, keyed by the label's first segment.
_LIBRARIES = {
    "PseudoDojo": _pseudodojo,
    "SG15": _sg15,
}

# How many segments each library's label has, so a label of the wrong shape
# falls through to the "cannot download this" message rather than raising on
# the unpacking inside an installer.
_LABEL_SEGMENTS = {
    "PseudoDojo": 6,
    "SG15": 4,
}


def ensure_pseudo_family_installed(pseudo_family: str) -> None:
    """Ensure a pseudopotential family is installed, installing it if necessary.

    Any already-installed family is used as it stands, whatever its label. A
    label that names no installed family is downloaded, which koopmans can do
    for the two norm-conserving libraries:
        'PseudoDojo/0.4/LDA/SR/standard/upf'
        'SG15/1.2/PBE/SR'

    Raises:
        ValueError: If no family carries the label and koopmans cannot
            download it, or if the download fails.
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


def require_norm_conserving_family(pseudo_family: str, structure: orm.StructureData) -> None:
    """Reject a family whose pseudopotentials are not norm-conserving.

    Reads the UPF header of the pseudopotential each of ``structure``'s kinds
    would use. A header that states nothing about its type passes: the check
    refuses on positive evidence of an ultrasoft or PAW pseudopotential, never
    on a header it cannot read.

    Raises:
        ValueError: If any pseudopotential the run would use is ultrasoft or PAW.
    """
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    from ._norm_conserving import non_norm_conserving_kinds

    family = PseudoPotentialFamily.collection.get(label=pseudo_family)
    offenders = non_norm_conserving_kinds(family.get_pseudos(structure=structure))
    if not offenders:
        return

    named = ", ".join(f"{kind} ({pseudo_type})" for kind, pseudo_type in offenders.items())
    raise ValueError(
        f"The pseudopotential family `{pseudo_family}` is not norm-conserving: {named}. "
        "Koopmans functionals are defined for norm-conserving pseudopotentials, and "
        "kcp.x and kcw.x accept no other kind. Set `workflow.pseudo_library` to a "
        "norm-conserving family; run `koopmans pseudos` for the ones koopmans can "
        "download."
    )


def available_pseudo_families() -> dict[str, list[str]]:
    """Return every valid ``pseudo_library`` label, sorted, keyed by library.

    Every label is norm-conserving and in UPF format: Koopmans functionals are
    defined for norm-conserving pseudopotentials, and ``pw.x`` reads UPF alone.

    No profile is needed.
    """
    return {library: module.available_labels() for library, module in _LIBRARIES.items()}


def installed_pseudo_family_labels() -> set[str]:
    """Return the labels of the families installed in the koopmans profile.

    Empty when no profile exists yet.
    """
    from aiida import orm

    from ..profile import load_koopmans_profile, profile_exists

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
        notes = getattr(_LIBRARIES[library], "NOTES", ())
        if notes:
            click.echo("")
            for note in notes:
                click.echo(f"  {note}")

    click.echo("\nEvery family listed is norm-conserving and in UPF format, which is what")
    click.echo("Koopmans functionals and `pw.x` require.")
    click.echo("\nName one of these as `pseudo_library` in the input file's `workflow` block, and")
    click.echo("koopmans installs it the first time it is used. To use pseudopotentials of your")
    click.echo("own, run `aiida-pseudo install family <directory> <label>` and name that label;")
    click.echo("it will recommend no cutoffs, so set `ecutwfc` in your input file.")


def install_pseudo_family(pseudo_family: str) -> None:
    """Download and install a pseudopotential family. Parse the label and dispatch.

    No family may already carry the label.

    Raises:
        ValueError: If the label names a family koopmans cannot run with, or
            one it cannot download.
    """
    parts = pseudo_family.split("/")
    library = _LIBRARIES.get(parts[0])

    if library is not None and len(parts) == _LABEL_SEGMENTS[parts[0]]:
        library.install(pseudo_family, parts)
    elif parts[0] == "SSSP":
        raise ValueError(
            f"'{pseudo_family}' is an SSSP family. SSSP mixes ultrasoft, PAW and "
            "norm-conserving pseudopotentials, and Koopmans functionals are defined "
            "for norm-conserving ones. Name a PseudoDojo or SG15 family instead; run "
            "`koopmans pseudos` for the full list."
        )
    else:
        raise ValueError(
            f"No installed pseudopotential family has the label '{pseudo_family}', and "
            "koopmans cannot download one under that label.\n"
            "Install the pseudopotentials yourself, from a directory holding one file "
            "per element:\n"
            f"    aiida-pseudo install family <directory> {pseudo_family}\n"
            "A family installed this way publishes no recommended cutoffs, so set "
            "`calculator_parameters.ecutwfc` in your input file; `ecutrho` follows at "
            "four times it.\n"
            "Alternatively, name a family koopmans can download for you: run "
            "`koopmans pseudos` for every label it accepts."
        )
