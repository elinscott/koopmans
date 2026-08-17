"""Progress display for AiiDA workgraph execution using rich."""

from __future__ import annotations

import re
from collections import defaultdict
from time import sleep
from typing import TYPE_CHECKING, NamedTuple, cast

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from koopmans.aiida.labels import LabelDisplay, describe_label, prettify_label
from koopmans.aiida.utils import get_node_label, suppress_stdout

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aiida.orm import ProcessNode
    from aiida_workgraph import WorkGraph

# Re-exported so the display names stay reachable as ``progress.<name>``,
# the way every caller already spells them.
__all__ = ["LabelDisplay", "describe_label", "prettify_label"]


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
    code: str | None = None


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


def _ordered_children(process_node: ProcessNode) -> list[tuple[LabelDisplay, ProcessNode]]:
    """Return the displayable children as ``(display, node)``, in display order.

    Children read in creation order, by ctime and then pk. Against that
    order, a *family* — the label with its digit runs masked, so every
    ``Compute Alpha Orb N`` shares one — is sorted naturally (``Orb 2``
    before ``Orb 10``), but only where its members were created
    consecutively. A family split by another step never was a fan-out:
    the spin initialization runs ``nspin=1``, ``nspin=2 (dummy)``,
    ``nspin=2``, and pulling the two ``nspin=2`` rows together would put
    the row that reads the restart files above the one that writes them.

    ``@calcfunction`` / ``@workfunction`` / ``@task`` PyFunctions are
    dropped here, along with their descendants (see
    :func:`_is_process_function_node`).
    """
    try:
        called_pks = [n.pk for n in process_node.called]
    except Exception:
        return []

    entries: list[tuple[LabelDisplay, ProcessNode]] = []
    for pk in called_pks:
        if pk is None:
            continue
        try:
            child = _reload(pk)
        except Exception:  # noqa: S112 - skip unreadable children
            continue
        if _is_process_function_node(child):
            continue
        display = describe_label(
            get_node_label(child, include_code=True),
            getattr(child, "process_label", None) or "",
        )
        entries.append((display, child))

    entries.sort(key=lambda entry: (entry[1].ctime, entry[1].pk or 0))

    positions: dict[str, list[int]] = defaultdict(list)
    for index, (display, _) in enumerate(entries):
        positions[_family_key(display.text)].append(index)

    for indices in positions.values():
        if len(indices) > 1 and indices == list(range(indices[0], indices[-1] + 1)):
            run = sorted(
                (entries[index] for index in indices), key=lambda e: _natural_key(e[0].text)
            )
            entries[indices[0] : indices[-1] + 1] = run
    return entries


class _Row:
    """A row under construction, holding the rows nested beneath it."""

    def __init__(self, display: LabelDisplay, state: str) -> None:
        self.text = display.text
        self.code = display.code
        self.numbered = display.numbered
        self.state = state
        self.children: list[_Row] = []


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
    root = describe_label(getattr(process_node, "process_label", None) or "WorkGraph")
    rows, _ = _collect_rows(process_node, root, is_root=True)
    return _flatten(rows, depth=0)


def _flatten(rows: Sequence[_Row], depth: int) -> list[ProcessRow]:
    """Return the nested rows as a flat list, parents before their children."""
    out: list[ProcessRow] = []
    for row in rows:
        out.append(ProcessRow(row.text, depth, row.state, row.code))
        out.extend(_flatten(row.children, depth + 1))
    return out


def _number_siblings(rows: Sequence[_Row]) -> None:
    """Count the rows that number themselves, 1, 2, 3 in the order they ran.

    The screening recursion names every iteration the same, so only
    position tells them apart.
    """
    index = 0
    for row in rows:
        if row.numbered:
            index += 1
            row.text = f"{row.text} {index}"
            row.numbered = False


def _collect_rows(
    process_node: ProcessNode,
    display: LabelDisplay,
    is_root: bool = False,
) -> tuple[list[_Row], str | None]:
    """Build the rows for one process and its descendants.

    Two kinds of process get no row of their own:

    * one the display table marks transparent — a container that adds no
      idea the row above does not already state. Its children rise to the
      parent's depth.
    * one that is the entire content of its parent. A container whose
      whole subtree is a single leaf calculation collapses into one row
      carrying the container's name and the calculation's code, and
      collapsing chains, so ``scf`` → ``PwBaseWorkChain`` →
      ``PwCalculation`` is the one row ``SCF`` · ``pw.x``. A restarting
      ``PwBaseWorkChain`` runs two calculations and so does not collapse:
      its attempts stay visible, which is when a watcher needs them. The
      root never collapses — it names the workflow, not a step of it.

    A process without a row lends its state to the row that stands for
    it: while that row is non-terminal it displays whichever state is
    further along, so it reads ``running`` while the ``@task.graph``
    around it still reports ``waiting``. No process is hidden
    conditionally on its state, so a row that has appeared is never
    withdrawn.

    Args:
        process_node: The process to render.
        display: How that process is displayed.
        is_root: Whether this is the root of the whole display.

    Returns:
        The rows for this subtree, and the state this process lends its
        nearest visible ancestor (``None`` when it has a row of its own).
    """
    node_type = "workgraph" if is_root else get_node_type(process_node)
    state = get_process_state(process_node, node_type)
    if state == "finished" and not process_node.is_finished_ok:
        state = "failed"

    child_rows: list[_Row] = []
    borrowed: list[str] = []
    for child_display, child in _ordered_children(process_node):
        rows, promoted = _collect_rows(child, child_display)
        child_rows.extend(rows)
        if promoted is not None:
            borrowed.append(promoted)

    if display.transparent:
        return child_rows, _most_advanced([_promoted_state(state), *borrowed])

    # Numbering happens here, once the transparent containers between an
    # iteration and this row have handed their children up as siblings.
    _number_siblings(child_rows)

    row = _Row(display, state)
    # The root names the workflow, never one of its steps, so it keeps
    # both its row and whatever single step it has so far produced.
    if not is_root and len(child_rows) == 1 and not child_rows[0].children:
        collapsed = child_rows.pop()
        row.code = collapsed.code or row.code
        borrowed.append(_promoted_state(collapsed.state))
    row.children = child_rows

    if state not in _TERMINAL_STATES:
        row.state = _most_advanced([state, *borrowed])
    return [row], None


def add_process_rows(table: Table, process_node: ProcessNode) -> None:
    """Add a styled row to the table for each process in the tree.

    Args:
        table: The Table to add rows to.
        process_node: The root process node.
    """
    for row in build_progress_rows(process_node):
        style = STATUS_STYLES.get(row.state, "")
        status_text = f"[{style}]{row.state}[/{style}]" if style else row.state
        table.add_row(f"{'  ' * row.depth}{row.label}", row.code or "", status_text)


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
    table.add_column("Step", no_wrap=True, min_width=56)
    table.add_column("Code", no_wrap=True, min_width=14)
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


def _walk_failed_descendants(
    node: ProcessNode,
) -> list[tuple[int | None, str, int | None, str | None, str]]:
    """Collect every terminated-not-ok process in the tree, including ``node`` itself.

    Returns ``(pk, process_label, exit_status, message, state)`` tuples in
    creation order. ``state`` is ``"excepted"``, ``"killed"``, or
    ``"failed"`` (a normal finished-not-ok exit); ``message`` is the
    exception text for an excepted process, the exit message for a
    failed one, and usually ``None`` for a killed one. A cascading
    failure can appear more than once (a wrapper and the child that
    actually failed), which is expected: the failed leaf carries the
    real cause. As with :func:`_walk_paused_descendants`, PyFunction
    nodes the progress table hides still appear here — a failure is
    diagnostic information whether or not it has a row.
    """
    out: list[tuple[int | None, str, int | None, str | None, str]] = []

    def _visit(n: ProcessNode) -> None:
        if n.is_terminated and not n.is_finished_ok:
            if n.is_excepted:
                state, message = "excepted", n.exception
            elif n.is_killed:
                state, message = "killed", n.exit_message
            else:
                state, message = "failed", n.exit_message
            out.append(
                (n.pk, n.process_label or n.__class__.__name__, n.exit_status, message, state)
            )
        try:
            children = sorted(n.called, key=lambda child: (child.ctime, child.pk or 0))
        except Exception:
            return
        for child in children:
            _visit(child)

    _visit(node)
    return out


def _print_outcome_banner(console: Console, process_node: ProcessNode) -> None:
    """Print the closing line for a process that has terminated."""
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


def render_process_once(process_node: ProcessNode, console: Console | None = None) -> None:
    """Print a process's current state once: the step tree, and any failure detail.

    Unlike :func:`watch_process` this does not wait — it renders whatever
    state the process is in right now, which may still be running. Used
    by ``koopmans status``, and by ``koopmans attach`` when the process
    it was pointed at has already terminated.
    """
    console = Console() if console is None else console
    console.print()
    console.print(make_progress_table(process_node))

    for pk, label, exit_status, message, state in _walk_failed_descendants(process_node):
        detail = f"exit status {exit_status}" if state == "failed" else state
        if message:
            detail += f": {message}"
        console.print(f"  [red]{prettify_label(label)}[/red] (pk {pk}) — {detail}")

    if process_node.is_terminated:
        _print_outcome_banner(console, process_node)


def watch_process(
    process_node: ProcessNode, refresh_interval: float = 2.0, console: Console | None = None
) -> ProcessNode:
    """Drive the live progress display against an already-running process.

    Reloads ``process_node`` on ``refresh_interval`` until it terminates,
    the same schedule :func:`run_with_progress` uses for a process it has
    just submitted itself. Used by ``koopmans attach`` to pick up a
    calculation ``koopmans submit`` started earlier.

    Returns the final, reloaded process node.
    """
    from aiida.orm import load_node

    console = Console() if console is None else console
    pk = process_node.pk
    with Live(make_progress_table(process_node), console=console, refresh_per_second=1) as live:
        while not process_node.is_terminated:
            sleep(refresh_interval)
            # Reload the process node to get fresh state
            process_node = cast("ProcessNode", load_node(pk))
            live.update(make_progress_table(process_node))

        # Final update to show completed status
        live.update(make_progress_table(process_node))

    _print_outcome_banner(console, process_node)
    return process_node


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
    process_node = watch_process(process_node, refresh_interval=refresh_interval, console=console)

    # Update the original wg object so callers can access the results
    wg.process = process_node
