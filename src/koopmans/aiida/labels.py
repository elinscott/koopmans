"""Display rules for the steps the progress table lists.

A step is named by the process that runs it: ``aiida-koopmans`` sets
``metadata.label`` on every workchain, calculation and sub-graph it
creates, each route names its own run, and the table shows what the node
says. Nothing here invents a name.

What is left here are the rules that do not belong to any one step:
which containers get a row at all, which rows are counted against their
siblings, and which binary a row ran. A process that carries no label —
one from a run made before the plugin named its steps, or one an
upstream workchain submits with metadata of its own — is shown by the
identifier provenance recorded for it, so what a reader sees is what the
database holds rather than a guess at what it meant.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, NamedTuple

__all__ = ["LabelDisplay", "describe_label", "executable_of", "prettify_label"]


class LabelDisplay(NamedTuple):
    """How one process is shown: its name, its executable, and its role.

    ``code`` is filled from the process itself (:func:`executable_of`),
    not from its label.
    """

    text: str
    code: str | None = None
    transparent: bool = False
    numbered: bool = False


# Containers that add no idea the row above does not already state. They
# get no row; their children rise to the parent's depth. Membership is by
# the identifier provenance records — a call link label, or the class of
# the process — because whether a container is worth a row is a fact
# about the shape of the run, not about what it is called. A container
# that groups two genuinely different steps (``scf_nscf``, "Ground
# state") keeps its row, because that grouping is information.
_TRANSPARENT: frozenset[str | tuple[str, str]] = frozenset(
    {
        "PwBandsWorkChain",
        "Wannier90WorkChain",
        "refine_screening_parameters",
        ("wannier90", "Wannier90WorkChain"),
    }
)

# Steps counted by their position among their siblings rather than by
# anything they are called: the screening recursion runs every iteration
# the same way, and only the order distinguishes them.
_NUMBERED = frozenset({"ScreeningIteration", "screening_iteration"})


def describe_label(raw: str, process_label: str = "") -> LabelDisplay:
    """Return the role one process plays, given the identifier behind it.

    ``raw`` is the call link label :func:`~koopmans.aiida.utils.get_node_label`
    builds, or a ``process_label`` for the root row and the failure
    summary. ``process_label`` disambiguates an identifier that stands
    for different things in different places.

    The text is the identifier itself, which is what a process carrying
    no label of its own is shown as.

    Examples:
    >>> describe_label("scf").text
    'scf'
    >>> describe_label("wannier90", "Wannier90WorkChain").transparent
    True
    >>> describe_label("ScreeningIteration").numbered
    True
    """
    if not raw:
        return LabelDisplay(raw)
    # ``aiida-workgraph`` wraps a graph's process_label as
    # ``WorkGraph<name>``; the envelope repeats what the context already
    # says, and the name inside it is the identifier.
    match = re.fullmatch(r"WorkGraph<(.+)>", raw)
    if match:
        raw = match.group(1)
    transparent = (raw, process_label) in _TRANSPARENT or raw in _TRANSPARENT
    return LabelDisplay(raw, None, transparent, raw in _NUMBERED)


def executable_of(process_node: Any) -> str | None:
    """Return the name of the binary one process ran, or ``None``.

    A calculation runs one program and is shown next to it; a workflow
    runs whatever its steps run and is shown next to nothing. The two are
    told apart by the ``code`` input, which only a calculation takes.

    The name is the basename of that code's own ``filepath_executable``,
    so it is spelled the way the code was installed — a code registered
    under a label of its own (``decompose``, a second pw2wannier90.x)
    still answers with the binary behind it.
    """
    try:
        code = process_node.inputs.code
    except (AttributeError, KeyError, TypeError):
        return None
    path = getattr(code, "filepath_executable", None)
    return None if path is None else PurePosixPath(str(path)).name


def prettify_label(raw: str, process_label: str = "") -> str:
    """Return the display text for one process's identifier.

    Examples:
    >>> prettify_label("ki_trial")
    'ki_trial'
    >>> prettify_label("WorkGraph<KoopmansDSCFWorkflow>")
    'KoopmansDSCFWorkflow'
    """
    return describe_label(raw, process_label).text
