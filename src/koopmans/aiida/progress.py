"""Progress display for AiiDA workgraph execution using rich."""

from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from koopmans.aiida.utils import get_node_label, suppress_stdout

if TYPE_CHECKING:
    from aiida.orm import ProcessNode
    from aiida_workgraph import WorkGraph


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


def get_node_type(node) -> str:
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


def _is_process_function_node(node) -> bool:
    """Return True for ``@calcfunction``/``@workfunction``/``@task`` PyFunctions.

    These are internal plumbing — pseudo lookup, electron counts, alpha
    generation, Map source builders, gather steps — and add visual noise
    to the koopmans progress table. The koopmans flow's *user-meaningful*
    rows are CalcJobs (kcp.x / pw.x) and the WorkGraph/sub-WorkGraph
    branches; this predicate is the filter for everything else.
    """
    from aiida.orm import CalcFunctionNode, WorkFunctionNode

    return isinstance(node, (CalcFunctionNode, WorkFunctionNode))


def add_process_rows(table: Table, process_node, depth: int = 0) -> None:
    """Recursively add rows for a process and its children.

    Skips ``@calcfunction`` / ``@workfunction`` / ``@task`` PyFunctions
    (see :func:`_is_process_function_node`); those are internal helpers.
    The root node is always rendered.

    Children are re-loaded via ``load_node(pk)`` rather than reusing the
    Node objects yielded by ``process_node.called``. AiiDA keeps Node
    instances in a session-level cache, and the daemon's writes don't
    always invalidate that cache fast enough to appear in the live
    table — without an explicit reload, a graph task's children can
    sit invisible until the whole run finishes.

    Args:
        table: The Table to add rows to.
        process_node: The process node to display.
        depth: Current indentation depth.
    """
    if depth > 0 and _is_process_function_node(process_node):
        return

    indent = "  " * depth

    # Get label
    if depth > 0:
        label = get_node_label(process_node, include_code=True)
    else:
        label = "WorkGraph"

    # Get type and state
    node_type = get_node_type(process_node) if depth > 0 else "workgraph"
    state = get_process_state(process_node, node_type)
    if state == "finished" and not process_node.is_finished_ok:
        state = "failed"
    style = STATUS_STYLES.get(state, "")
    if style:
        status_text = f"[{style}]{state}[/{style}]"
    else:
        status_text = state

    table.add_row(f"{indent}{label}", status_text)

    # Recursively add children — reload each one freshly so the live
    # table picks up newly-spawned tasks without waiting for the run
    # to terminate (see docstring).
    from aiida.orm import load_node

    try:
        called_pks = [(n.pk, n.ctime) for n in process_node.called]
    except Exception:
        return
    called_pks.sort(key=lambda pair: pair[1])
    for pk, _ in called_pks:
        try:
            child = load_node(pk)
        except Exception:  # noqa: S112 - skip unreadable children
            continue
        add_process_rows(table, child, depth + 1)


def _walk_paused_descendants(node) -> list:
    """Collect every paused descendant.

    A *paused* sub-process is one whose transport-task retries have been
    exhausted and the daemon has stopped retrying.

    Returns a list of ``(pk, process_label)`` tuples — empty when nothing
    is paused. Used by :func:`make_progress_table` to surface a hint when
    the live display would otherwise look like a normal slow run.
    """
    out: list[tuple[int, str]] = []

    def _visit(n) -> None:
        if getattr(n, "paused", False):
            out.append((n.pk, n.process_label or n.__class__.__name__))
        try:
            for child in n.called:
                _visit(child)
        except Exception:
            return

    _visit(node)
    return out


def make_progress_table(process_node: ProcessNode):
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

    from koopmans.aiida.setup import ensure_daemon_running

    console = Console()
    console.print()

    # Ensure daemon is running before submitting
    ensure_daemon_running()

    # Submit the workgraph (suppress aiida-workgraph's print statements)
    with suppress_stdout():
        wg.submit()

    # Wait for process to be created
    while wg.process is None:
        sleep(0.1)

    # Display live progress by querying actual process nodes
    pk = wg.process.pk
    process_node = load_node(pk)
    with Live(make_progress_table(process_node), console=console, refresh_per_second=1) as live:
        while not process_node.is_terminated:
            sleep(refresh_interval)
            # Reload the process node to get fresh state
            process_node = load_node(pk)
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
