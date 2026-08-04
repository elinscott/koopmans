"""Progress display for AiiDA workgraph execution using rich."""

from __future__ import annotations

import re
from time import sleep
from typing import TYPE_CHECKING, NamedTuple, cast

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from koopmans.aiida.utils import get_node_label, suppress_stdout

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from aiida.orm import ProcessNode
    from aiida_workgraph import WorkGraph


# Acronyms that should stay uppercase after pretty-printing — these come
# from the physics jargon used in task names (functionals, code names,
# etc.). Add new ones here as the workflow grows.
_ACRONYMS = frozenset({"ki", "dft", "dscf", "kipz", "pkipz", "ks", "pz", "scf", "kc", "kcw"})

# Token regex for ``_prettify``: matches a leading run of caps not
# followed by a ``Cap+lowercase`` boundary, an initial cap with
# lowercase tail (``Iteration``), an all-lowercase run, an all-caps run
# (acronyms standalone), or a digit run. Together this handles
# CamelCase, snake_case, and trailing-digit suffixes uniformly.
_PRETTIFY_TOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def prettify_label(raw: str) -> str:
    """Convert an internal task / call-link label into a display string.

    Rules:

    * Strip a ``<plugin>-`` prefix when the left half is lowercase
      (``kcp-ki_trial`` → ``ki_trial``). Plugin names are implementation
      detail; the call_link_label is the action.
    * Split on underscores and CamelCase boundaries
      (``ScreeningIteration1`` → ``["Screening", "Iteration", "1"]``).
    * Keep known acronyms uppercase (``ki`` → ``KI``,
      ``dscf`` → ``DSCF``, …); other tokens get a leading capital.
    * Re-join with single spaces.

    Examples:
    >>> prettify_label("ki_trial")
    'KI Trial'
    >>> prettify_label("kcp-dft_init")
    'DFT Init'
    >>> prettify_label("ScreeningIteration1")  # workgraph numbers from 0; users from 1
    'Iteration 2'
    >>> prettify_label("KoopmansDSCFWorkflow")
    'Koopmans DSCF Workflow'
    >>> prettify_label("convert_spin1_to_spin2")
    'Convert Spin 1 To Spin 2'
    """
    if not raw:
        return raw
    if "-" in raw and raw.split("-", 1)[0].islower():
        raw = raw.split("-", 1)[1]
    # ``aiida-workgraph`` wraps the top-level process_label as
    # ``WorkGraph<KoopmansDSCFWorkflow>``. The user already knows it's a
    # WorkGraph from the context (it's the root of the display), so peel
    # the envelope before tokenising.
    m = re.match(r"^WorkGraph<(.+)>$", raw)
    if m:
        raw = m.group(1)
    out: list[str] = []
    for chunk in raw.split("_"):
        for token in _PRETTIFY_TOKEN_RE.findall(chunk):
            if token.isdigit():
                out.append(token)
            elif token.lower() in _ACRONYMS:
                out.append(token.upper())
            else:
                out.append(token[0].upper() + token[1:].lower())
    s = " ".join(out) if out else raw
    # Physics-paper conventions that read better than the tokenised form.
    # Order matters: the longer (nspin=N; dummy) rule must run before the
    # bare nspin one so it consumes the trailing "Dummy".
    s = re.sub(r"\bNspin (\d+) Dummy\b", r"(nspin=\1; dummy)", s)
    s = re.sub(r"\bNspin (\d+)\b", r"(nspin=\1)", s)
    s = re.sub(r"\bN Minus (\d+)\b", r"N-\1", s)
    s = re.sub(r"\bN Plus (\d+)\b", r"N+\1", s)
    # Compute-screening-parameters context already says "Screening" —
    # the inner iterations are just "Iteration <N>". aiida-workgraph
    # auto-numbers repeated tasks from 0 (first instance bare, second
    # is "1", …) but users count from 1, so shift the index up by one.
    s = re.sub(
        r"\bScreening Iteration (\d+)\b",
        lambda m: f"Iteration {int(m.group(1)) + 1}",
        s,
    )
    s = re.sub(r"\bScreening Iteration\b", "Iteration 1", s)
    # Orbital sub-graphs: parent gives the "screening" context, and
    # ``Orb N`` is just the Map-zone key for ``Orbital N`` — collapse.
    s = re.sub(r"\bOrb (\d+) Filled Orbital Screening\b", r"Orbital \1 (filled)", s)
    s = re.sub(r"\bOrb (\d+) Empty Orbital Screening\b", r"Orbital \1 (empty)", s)
    return s


# Status display styling
STATUS_STYLES = {
    "created": "dim",
    "waiting": "yellow",
    "running": "blue italic",
    "finished": "green",
    "failed": "red",
    "excepted": "red bold italic",
    "killed": "red",
    "paused": "magenta bold",
}


def get_process_state(process_node: ProcessNode, node_type: str = "") -> str:
    """Get the state of a process node.

    Args:
        process_node: An AiiDA ProcessNode.
        node_type: The type of node (calcjob, calcfunc, workchain, etc.)

    Returns:
        String representation of the process state.
    """
    try:
        # ``paused`` overrides everything: AiiDA marks a process paused
        # when a transport task (typically the upload) has failed its
        # retry budget and the daemon has stopped retrying. The process
        # is alive but no longer making progress — almost always means
        # stale scratch state from an earlier run. Surface it loudly so
        # the user knows the live table isn't just slow.
        if getattr(process_node, "paused", False):
            return "paused"
        state = process_node.process_state
        if state is not None:
            state_str = state.value.lower()
            # CalcJobs/CalcFunctions in "waiting" state have been submitted
            # and are effectively "running" from the user's perspective
            if state_str == "waiting" and node_type in ("calcjob", "calcfunc"):
                return "running"
            return state_str
        return "unknown"
    except Exception:
        return "unknown"


def get_node_type(node: ProcessNode) -> str:
    """Get a short type name for a process node."""
    from aiida.orm import CalcFunctionNode, CalcJobNode, WorkChainNode

    if isinstance(node, CalcJobNode):
        return "calcjob"
    elif isinstance(node, CalcFunctionNode):
        return "calcfunc"
    elif isinstance(node, WorkChainNode):
        return "workchain"
    else:
        return "process"


def _is_process_function_node(node: ProcessNode) -> bool:
    """Return True for ``@calcfunction``/``@workfunction``/``@task`` PyFunctions.

    These are internal plumbing — pseudo lookup, electron counts, alpha
    generation, Map source builders, gather steps — and add visual noise
    to the koopmans progress table. The koopmans flow's *user-meaningful*
    rows are CalcJobs (kcp.x / pw.x) and the WorkGraph/sub-WorkGraph
    branches; this predicate is the filter for everything else.
    """
    from aiida.orm import CalcFunctionNode, WorkFunctionNode

    return isinstance(node, (CalcFunctionNode, WorkFunctionNode))


class ProcessRow(NamedTuple):
    """One line of the progress table: what to show, how far in, and its state."""

    label: str
    depth: int
    state: str


# States a process passes through before it terminates, ordered by how
# far along they are. ``paused`` ranks top because it is the one
# non-terminal state the user must act on. Terminal states are absent:
# they never travel from a hidden row to a visible one (see
# :func:`_promoted_state`), and an unrecognised state ranks below all of
# these.
_STATE_RANK = {"created": 0, "waiting": 1, "running": 2, "paused": 3}
_TERMINAL_STATES = frozenset({"finished", "failed", "excepted", "killed"})

# Splits a label into its alternating text and digit runs. The capture
# group keeps the digits, so ``"Orb 10"`` → ``["Orb ", "10", ""]``.
_DIGIT_RUN_RE = re.compile(r"(\d+)")


def _natural_key(label: str) -> tuple[tuple[str, int], ...]:
    """Return a sort key that compares digit runs in a label numerically.

    ``re.split`` on a capturing digit group always alternates text,
    digits, text, …, so pairing them up yields keys whose components
    line up across labels: ``"Orb 2"`` sorts before ``"Orb 10"``.
    """
    parts = _DIGIT_RUN_RE.split(label)
    return tuple(
        (parts[i], int(parts[i + 1]) if i + 1 < len(parts) else -1) for i in range(0, len(parts), 2)
    )


def _family_key(label: str) -> str:
    """Return the label with every digit run masked (``"Orb 10"`` → ``"Orb #"``)."""
    return _DIGIT_RUN_RE.sub("#", label)


def _most_advanced(states: Sequence[str]) -> str:
    """Return the state furthest along ``created`` → ``waiting`` → ``running`` → ``paused``."""
    return max(states, key=lambda state: _STATE_RANK.get(state, -1))


def _promoted_state(state: str) -> str:
    """Return the state a hidden row lends its visible ancestor.

    A terminal state is clamped to ``running``: the ancestor is still
    live, and only its own outcome may be reported as terminal.
    """
    return "running" if state in _TERMINAL_STATES else state


def _reload(pk: int) -> ProcessNode:
    """Re-fetch a node by pk.

    AiiDA keeps Node instances in a session-level cache, and the daemon's
    writes don't always invalidate that cache fast enough to appear in
    the live table — without an explicit reload, a graph task's children
    can sit invisible until the whole run finishes.
    """
    from aiida.orm import load_node

    return cast("ProcessNode", load_node(pk))


def _ordered_children(process_node: ProcessNode) -> list[tuple[str, ProcessNode]]:
    """Return the displayable children as ``(prettified label, node)``, in display order.

    Children are grouped into *families* — the label with its digit runs
    masked, so every ``Compute Alpha Orb N`` shares one family. Families
    follow the creation order of their first member, keeping a
    workflow's sequential steps in the order they ran; within a family
    the order is natural (``Orb 2`` before ``Orb 10``), then ctime, then
    pk.

    ``@calcfunction`` / ``@workfunction`` / ``@task`` PyFunctions are
    dropped here, along with their descendants (see
    :func:`_is_process_function_node`).
    """
    try:
        called_pks = [n.pk for n in process_node.called]
    except Exception:
        return []

    entries: list[tuple[str, ProcessNode]] = []
    for pk in called_pks:
        if pk is None:
            continue
        try:
            child = _reload(pk)
        except Exception:  # noqa: S112 - skip unreadable children
            continue
        if _is_process_function_node(child):
            continue
        entries.append((prettify_label(get_node_label(child, include_code=True)), child))

    first_seen: dict[str, tuple[datetime, int]] = {}
    for label, child in entries:
        stamp = (child.ctime, child.pk or 0)
        family = _family_key(label)
        earliest = first_seen.get(family)
        if earliest is None or stamp < earliest:
            first_seen[family] = stamp

    entries.sort(
        key=lambda entry: (
            first_seen[_family_key(entry[0])],
            _natural_key(entry[0]),
            entry[1].ctime,
            entry[1].pk or 0,
        )
    )
    return entries


def build_progress_rows(process_node: ProcessNode) -> list[ProcessRow]:
    """Assemble one row per displayed process, depth-first from the root.

    The root row shows the top-level ``process_label``
    (``KoopmansDSCFWorkflow`` etc.) rather than a hard-coded
    ``"WorkGraph"``, so the user sees the workflow they invoked.

    Args:
        process_node: The root process node.

    Returns:
        The table's rows, parents before their children.
    """
    root_label = prettify_label(getattr(process_node, "process_label", None) or "WorkGraph")
    rows, _ = _collect_rows(process_node, root_label, depth=0, parent_label=None)
    return rows


def _collect_rows(
    process_node: ProcessNode,
    label: str,
    depth: int,
    parent_label: str | None,
) -> tuple[list[ProcessRow], str | None]:
    """Build the rows for one process and its descendants.

    A process whose label is a prefix of (or identical to) its parent's
    gets no row of its own: it is a redundant single-CalcJob wrapper —
    the ``DFTInitialization`` ``@task.graph`` wraps one ``kcp.x`` call
    whose label (``"DFT Init"``) is already part of the wrapper's
    (``"DFT Init (nspin=1)"``). Its descendants are rendered against the
    nearest visible ancestor, at that ancestor's depth.

    A hidden process lends its state to its visible ancestor: while that
    ancestor is non-terminal it displays whichever of the two states is
    further along, so a wrapper's row reads ``running`` while the
    ``@task.graph`` around it still reports ``waiting``. No process is
    hidden conditionally on its state, so a row that has appeared is
    never withdrawn.

    Args:
        process_node: The process to render.
        label: Its prettified label.
        depth: Indentation depth of the row it would occupy.
        parent_label: The prettified label of the nearest visible
            ancestor, or ``None`` for the root.

    Returns:
        The rows for this subtree, and the state this process lends its
        visible ancestor (``None`` when it has a row of its own).
    """
    node_type = get_node_type(process_node) if depth > 0 else "workgraph"
    state = get_process_state(process_node, node_type)
    if state == "finished" and not process_node.is_finished_ok:
        state = "failed"

    suppress_self = parent_label is not None and (
        label == parent_label or parent_label.startswith(label + " ")
    )
    child_parent_label = parent_label if suppress_self else label
    child_depth = depth if suppress_self else depth + 1

    child_rows: list[ProcessRow] = []
    borrowed: list[str] = []
    for child_label, child in _ordered_children(process_node):
        rows, promoted = _collect_rows(child, child_label, child_depth, child_parent_label)
        child_rows.extend(rows)
        if promoted is not None:
            borrowed.append(promoted)

    if suppress_self:
        return child_rows, _most_advanced([_promoted_state(state), *borrowed])
    if state not in _TERMINAL_STATES:
        state = _most_advanced([state, *borrowed])
    return [ProcessRow(label, depth, state), *child_rows], None


def add_process_rows(table: Table, process_node: ProcessNode) -> None:
    """Add a styled row to the table for each process in the tree.

    Args:
        table: The Table to add rows to.
        process_node: The root process node.
    """
    for row in build_progress_rows(process_node):
        style = STATUS_STYLES.get(row.state, "")
        status_text = f"[{style}]{row.state}[/{style}]" if style else row.state
        table.add_row(f"{'  ' * row.depth}{row.label}", status_text)


def _walk_paused_descendants(node: ProcessNode) -> list[tuple[int | None, str]]:
    """Collect every paused descendant.

    A *paused* sub-process is one whose transport-task retries have been
    exhausted and the daemon has stopped retrying.

    Returns a list of ``(pk, process_label)`` tuples in creation order —
    empty when nothing is paused. Used by :func:`make_progress_table` to
    surface a hint when the live display would otherwise look like a
    normal slow run.

    This walk keeps the PyFunction nodes that the table hides: a paused
    upload belongs in the hint whether or not it has a row.
    """
    out: list[tuple[int | None, str]] = []

    def _visit(n: ProcessNode) -> None:
        if getattr(n, "paused", False):
            out.append((n.pk, n.process_label or n.__class__.__name__))
        try:
            children = sorted(n.called, key=lambda child: (child.ctime, child.pk or 0))
        except Exception:
            return
        for child in children:
            _visit(child)

    _visit(node)
    return out


def make_progress_table(process_node: ProcessNode) -> Table | Group:
    """Build the live progress display: the per-task table plus optional hints.

    Returns a ``rich.console.Group`` containing the task table and, when
    one or more descendants are in the paused state (transport-task
    retry budget exhausted), a short footer line pointing the user at
    the right diagnostic command. The paused-detection logic guards
    against the most common live-table confusion: a process that looks
    like it's just slow but is actually wedged on stale AiiDA scratch
    state from a previous failed run.

    Args:
        process_node: The WorkGraphNode process to display progress for.
    """
    table = Table(box=None)
    table.add_column("Step", no_wrap=True, min_width=70)
    table.add_column("Status", justify="right")

    add_process_rows(table, process_node)

    paused = _walk_paused_descendants(process_node)
    if not paused:
        return table

    hint_lines = [
        Text(""),
        Text(
            f"⚠ {len(paused)} process(es) paused after exhausted transport retries — "
            "the daemon has stopped trying to recover them.",
            style="magenta bold",
        ),
        Text(
            "  This typically means stale AiiDA scratch directories from an "
            "earlier failed run. Diagnose with:",
            style="magenta",
        ),
    ]
    for pk, label in paused[:5]:
        hint_lines.append(Text(f"    verdi process show {pk}  # {label}", style="magenta"))
    if len(paused) > 5:
        hint_lines.append(Text(f"    … and {len(paused) - 5} more", style="magenta"))
    return Group(table, *hint_lines)


def run_with_progress(wg: WorkGraph, refresh_interval: float = 2.0) -> None:
    """Submit and run a workgraph with a live progress display.

    This function submits the workgraph and displays a live-updating
    table showing the status of each task until completion.

    Args:
        wg: The WorkGraph instance to run.
        refresh_interval: How often to refresh the display (in seconds).
    """
    from aiida.orm import load_node

    from koopmans.api import launch

    console = Console()
    console.print()

    # Launch through the api funnel — the one call site for the workgraph
    # verbs — suppressing aiida-workgraph's print statements.
    with suppress_stdout():
        launch(wg, blocking=False)

    # Wait for process to be created
    while wg.process is None:
        sleep(0.1)

    # Display live progress by querying actual process nodes
    pk = wg.process.pk
    process_node = cast("ProcessNode", load_node(pk))
    with Live(make_progress_table(process_node), console=console, refresh_per_second=1) as live:
        while not process_node.is_terminated:
            sleep(refresh_interval)
            # Reload the process node to get fresh state
            process_node = cast("ProcessNode", load_node(pk))
            live.update(make_progress_table(process_node))

        # Final update to show completed status
        live.update(make_progress_table(process_node))

    # Print final status
    if process_node.is_finished_ok:
        console.print("\n[bold green]Workflow completed successfully![/bold green]")
    elif process_node.is_excepted:
        console.print("\n[bold red]Workflow excepted![/bold red]")
    elif process_node.is_killed:
        console.print("\n[bold red]Workflow was killed![/bold red]")
    else:
        console.print(
            f"\n[bold red]Workflow finished with status: {process_node.exit_status}[/bold red]"
        )

    # Update the original wg object so callers can access the results
    wg.process = process_node
