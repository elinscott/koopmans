"""Progress display for AiiDA workgraph execution using rich."""

from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.table import Table

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
    "excepted": "red bold",
    "killed": "red",
}


def get_process_state(process_node: "ProcessNode", node_type: str = "") -> str:
    """Get the state of a process node.

    Args:
        process_node: An AiiDA ProcessNode.
        node_type: The type of node (calcjob, calcfunc, workchain, etc.)

    Returns:
        String representation of the process state.
    """
    try:
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


def add_process_rows(
    table: Table, process_node, depth: int = 0, max_depth: int = 5
) -> None:
    """Recursively add rows for a process and its children.

    Args:
        table: The Table to add rows to.
        process_node: The process node to display.
        depth: Current indentation depth.
        max_depth: Maximum recursion depth.
    """
    indent = "  " * depth

    # Get label
    if depth > 0:
        label = get_node_label(process_node, include_code=True)
    else:
        label = "WorkGraph"

    # Get type and state
    node_type = get_node_type(process_node) if depth > 0 else "workgraph"
    state = get_process_state(process_node, node_type)
    style = STATUS_STYLES.get(state, "")
    if style:
        status_text = f"[{style}]{state}[/{style}]"
    else:
        status_text = state

    table.add_row(f"{indent}{label}", status_text)

    # Recursively add children
    if depth < max_depth:
        try:
            called = list(process_node.called)
            called.sort(key=lambda n: n.ctime)
            for child in called:
                add_process_rows(table, child, depth + 1, max_depth)
        except Exception:
            pass


def make_progress_table(process_node: "ProcessNode") -> Table:
    """Create a rich table showing workgraph progress.

    Queries the called children of the workgraph process to show their states.

    Args:
        process_node: The WorkGraphNode process to display progress for.

    Returns:
        A rich Table object with task status information.
    """
    table = Table(box=None)
    table.add_column("Step", no_wrap=True, min_width=70)
    table.add_column("Status", justify="right")

    add_process_rows(table, process_node)

    return table


def run_with_progress(wg: "WorkGraph", refresh_interval: float = 2.0) -> None:
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
        while not process_node.is_finished:
            sleep(refresh_interval)
            # Reload the process node to get fresh state
            process_node = load_node(pk)
            live.update(make_progress_table(process_node))

        # Final update to show completed status
        live.update(make_progress_table(process_node))

    # Print final status
    if process_node.is_finished_ok:
        console.print("\n[bold green]Workflow completed successfully![/bold green]")
    else:
        console.print(
            f"\n[bold red]Workflow finished with status: {process_node.exit_status}[/bold red]"
        )

    # Update the original wg object so callers can access the results
    wg.process = process_node
