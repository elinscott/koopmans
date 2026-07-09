"""Progress display for AiiDA workgraph execution using rich."""

from __future__ import annotations

import re
from time import sleep
from typing import TYPE_CHECKING, cast

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from koopmans.aiida.utils import get_node_label, suppress_stdout

if TYPE_CHECKING:
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
    >>> prettify_label("ScreeningIteration1")
    'Screening Iteration 1'
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


def add_process_rows(
    table: Table,
    process_node: ProcessNode,
    depth: int = 0,
    parent_label: str | None = None,
) -> None:
    """Recursively add rows for a process and its children.

    Skips ``@calcfunction`` / ``@workfunction`` / ``@task`` PyFunctions
    (see :func:`_is_process_function_node`); those are internal helpers.
    The root node is always rendered.

    Also suppresses a row whose prettified label is a *prefix* of (or
    identical to) its parent's prettified label. This collapses the
    redundant single-CalcJob wrappers — e.g. the ``DFTInitialization``
    ``@task.graph`` wraps one ``kcp.x`` call whose label
    (``"DFT Init"``) is already part of the wrapper's
    (``"DFT Init (nspin=1)"``).

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
        parent_label: The prettified label of the parent row, or ``None``
            for the root. Used to suppress redundant child rows.
    """
    if depth > 0 and _is_process_function_node(process_node):
        return

    indent = "  " * depth

    # Get label. The root row shows the top-level process_label
    # (``KoopmansDSCFWorkflow`` etc.) rather than a hard-coded
    # ``"WorkGraph"``, so the user sees the actual workflow they invoked.
    # Both branches go through ``prettify_label`` for consistent
    # CamelCase / snake_case / acronym handling.
    if depth > 0:
        raw_label = get_node_label(process_node, include_code=True)
    else:
        raw_label = getattr(process_node, "process_label", None) or "WorkGraph"
    label = prettify_label(raw_label)

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

    # Suppress redundant wrapper-rows: a child row whose label is a
    # prefix of (or identical to) its parent's. *Only* once the child
    # is terminal — while it's running we still want to surface the
    # row so the user sees that step is making progress (the parent
    # @task.graph's status can lag behind its child CalcJob's). After
    # termination the duplicate row collapses away.
    _terminal_states = {"finished", "failed", "excepted", "killed"}
    suppress_self = (
        parent_label is not None
        and state in _terminal_states
        and (label == parent_label or parent_label.startswith(label + " "))
    )

    if not suppress_self:
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
    # Suppressed rows pass their parent's label / depth straight through
    # so the grandchild is rendered against the *visible* ancestor.
    child_parent_label = parent_label if suppress_self else label
    child_depth = depth if suppress_self else depth + 1
    for pk, _ in called_pks:
        try:
            child = cast("ProcessNode", load_node(pk))
        except Exception:  # noqa: S112 - skip unreadable children
            continue
        add_process_rows(table, child, child_depth, parent_label=child_parent_label)


def _walk_paused_descendants(node: ProcessNode) -> list[tuple[int | None, str]]:
    """Collect every paused descendant.

    A *paused* sub-process is one whose transport-task retries have been
    exhausted and the daemon has stopped retrying.

    Returns a list of ``(pk, process_label)`` tuples — empty when nothing
    is paused. Used by :func:`make_progress_table` to surface a hint when
    the live display would otherwise look like a normal slow run.
    """
    out: list[tuple[int | None, str]] = []

    def _visit(n: ProcessNode) -> None:
        if getattr(n, "paused", False):
            out.append((n.pk, n.process_label or n.__class__.__name__))
        try:
            for child in n.called:
                _visit(child)
        except Exception:
            return

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
