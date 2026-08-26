"""Command line interface for :mod:`koopmans`.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to
import things from ``__main__`` later, but that will cause problems--the code will get
executed twice:

- When you run ``python3 -m koopmans`` python will execute``__main__.py`` as a script.
  That means there won't be any ``koopmans.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because there's no
  ``koopmans.__main__`` in ``sys.modules``.

.. seealso::

    https://click.palletsprojects.com/en/8.1.x/setuptools/#setuptools-integration
"""

# A command's help text keeps a worked example on the lines it was written on
# by carrying click's own "\b" marker, which a raw docstring would turn into
# two literal characters.
# ruff: noqa: D301

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import click.parser

from koopmans.aiida.dumping import MODEL_FILENAME, dump_workgraph, trained_model_output
from koopmans.aiida.progress import run_with_progress
from koopmans.aiida.setup.codes import list_codes
from koopmans.aiida.setup.daemon import is_daemon_running, start_daemon, stop_daemon
from koopmans.aiida.setup.hq import (
    ensure_hq_running,
    hq_binary,
    install_hq_binary,
    is_hq_worker_running,
    restart_hq_worker,
    running_hq_workers,
    stop_hq_worker,
)
from koopmans.aiida.setup.orchestrate import (
    print_hq_status,
    print_status,
    setup_computers,
    uninstall_backend,
)
from koopmans.aiida.setup.profile import load_koopmans_profile, setup_profile
from koopmans.aiida.utils import suppress_aiida_logging
from koopmans.input_file import read_input_file
from koopmans.plotting.series import EnergyZero

if TYPE_CHECKING:
    from aiida import orm

__all__ = [
    "cli",
    "main",
]


@click.group()
@click.version_option()
@click.option(
    "--pdb",
    is_flag=True,
    default=False,
    help="Drop into ipdb debugger on unhandled exceptions.",
)
@click.option(
    "-l",
    "--logging",
    "enable_logging",
    is_flag=True,
    default=False,
    help="Enable logging to koopmans.log file.",
)
def cli(pdb: bool, enable_logging: bool) -> None:
    """Automated Koopmans functional calculations and workflows."""
    if pdb:
        from koopmans.debugging import enable_pdb

        enable_pdb()

    if enable_logging:
        logging.basicConfig(
            filename="koopmans.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def _anchor_run_submission(input_path: Path, node: orm.ProcessNode) -> None:
    """Record ``node`` in ``input_path``'s anchor file, warning rather than aborting on failure.

    Called once ``koopmans run``'s submission is durably in the daemon, so a
    write failure here must not kill a calculation that is already under
    way — it only costs the shortcut `koopmans status` would otherwise find
    on its own.
    """
    from koopmans.aiida.anchor import anchor_path_for_input, record_submission

    anchor_path = anchor_path_for_input(input_path)
    try:
        record_submission(anchor_path, input_path, node)
    except (ValueError, OSError) as exc:
        click.echo(
            f"Warning: the workflow was submitted, but {anchor_path} could not be "
            f"written ({exc}), so `koopmans status` will not find it on its own. "
            f"Follow it with `koopmans status --uuid {node.uuid}`.",
            err=True,
        )


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
def run(input_file: str) -> None:
    """Run a koopmans calculation from an input file.

    INPUT_FILE is the path to a YAML or JSON input file describing the
    calculation. Records the submission in `<stem>.run.yaml`, next to the
    input file, the same way `koopmans submit` does, so a run interrupted
    (e.g. Ctrl-C) after submission can still be found with `koopmans
    status` or `koopmans attach`.
    """
    from koopmans.aiida.workflows import advice_for, build_workgraph

    input_path = Path(input_file)

    # Print the header
    click.echo(header())

    # Parse input file
    koopmans_input = read_input_file(input_path)

    # Load AiiDA profile
    load_koopmans_profile()

    # Build the appropriate workgraph based on task
    wg = build_workgraph(koopmans_input)

    # Graph validation runs when the engine takes the graph, past the build
    # boundary where `build_workgraph` attaches advice — a missing
    # route-conditional code surfaces here, so translate at this boundary too.
    try:
        with suppress_aiida_logging():
            run_with_progress(
                wg, on_submitted=functools.partial(_anchor_run_submission, input_path)
            )
    except Exception as exc:
        advice = advice_for(exc)
        if advice is not None:
            exc.add_note(advice)
        raise

    if wg.process is not None:
        dump_path = input_path.parent / input_path.stem
        dump_workgraph(wg.process, output_path=dump_path, overwrite=True)
        if trained_model_output(wg.process) is not None:
            # `ml: model_file` reads a relative path against the input file's
            # own directory, so the snippet drops the leading directories the
            # written path carries when the run was started from elsewhere.
            click.echo(
                f"Trained model written to {dump_path / MODEL_FILENAME} — reuse it from "
                f"an input file beside {input_path.name} with "
                f"`ml: {{model_file: {Path(input_path.stem) / MODEL_FILENAME}}}`."
            )


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
def submit(input_file: str) -> None:
    """Submit a koopmans calculation to the daemon without waiting for it.

    INPUT_FILE is the path to a YAML or JSON input file describing the
    calculation. Records the submission in `<stem>.run.yaml`, next to the
    input file, appending rather than overwriting if it already exists;
    `koopmans status` and `koopmans attach` read that file to find the
    calculation again.
    """
    from koopmans.aiida.anchor import anchor_path_for_input, record_submission
    from koopmans.aiida.workflows import advice_for, build_workgraph
    from koopmans.api import launch

    input_path = Path(input_file)
    koopmans_input = read_input_file(input_path)

    load_koopmans_profile()
    wg = build_workgraph(koopmans_input)

    # Graph validation runs when the engine takes the graph, past the build
    # boundary where `build_workgraph` attaches advice — translate here too.
    try:
        with suppress_aiida_logging():
            node = launch(wg, blocking=False, wait=False)
    except Exception as exc:
        advice = advice_for(exc)
        if advice is not None:
            exc.add_note(advice)
        raise

    anchor_path = anchor_path_for_input(input_path)
    try:
        record_submission(anchor_path, input_path, node)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except OSError as exc:
        # The daemon already has the job; losing the run file only loses
        # the *shortcut* back to it, not the submission itself.
        raise click.ClickException(
            f"The workflow was submitted, but {anchor_path} could not be written "
            f"({exc}), so `koopmans status` will not find it on its own. Follow it "
            f"with `koopmans status --uuid {node.uuid}`."
        ) from exc

    click.echo("🚀 Workflow submitted")


def _load_target_process(target: str | None, uuid_: str | None, pk_: int | None) -> orm.ProcessNode:
    """Resolve and load the process a status/attach target refers to.

    Loads the koopmans AiiDA profile as a side effect, since resolution
    only touches the filesystem but loading the node needs the profile.
    """
    from aiida import orm
    from aiida.common.exceptions import NotExistent

    from koopmans.aiida.anchor import resolve_target

    try:
        resolved = resolve_target(target, uuid=uuid_, pk=pk_)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    load_koopmans_profile()
    # Both errors below name the identifier the user gave or the run file
    # recorded, not one read off the node they landed on.
    identifier = resolved.uuid if resolved.uuid is not None else resolved.pk
    try:
        node = (
            orm.load_node(uuid=resolved.uuid)
            if resolved.uuid is not None
            else orm.load_node(pk=resolved.pk)
        )
    except NotExistent as exc:
        raise click.ClickException(
            f"No AiiDA node found for {identifier!r}. It may have been deleted; check "
            "`verdi process list -a` for what is still in the database."
        ) from exc
    if not isinstance(node, orm.ProcessNode):
        raise click.ClickException(
            f"{identifier!r} is not a calculation; it holds a {type(node).__name__}."
        )
    return node


# Shared options for `status`/`attach`
target_argument = click.argument("target", required=False)
uuid_option = click.option(
    "--uuid",
    "uuid_",
    default=None,
    metavar="UUID",
    help="Load the process by its AiiDA node UUID directly, bypassing any run file.",
)
pk_option = click.option(
    "--pk",
    "pk_",
    type=int,
    default=None,
    metavar="PK",
    help="Load the process by its AiiDA node pk directly, bypassing any run file.",
)


@cli.command(name="status")
@target_argument
@uuid_option
@pk_option
def show_status(target: str | None, uuid_: str | None, pk_: int | None) -> None:
    """Show the current state of a submitted calculation, once.

    TARGET is a `<stem>.run.yaml` file, the input file next to one, or
    omitted to use the single such file in the current directory. Prints
    the workflow tree with each step's state, and the exit status and
    message of any step that failed. Exits nonzero if the calculation's
    root process failed.
    """
    from koopmans.aiida.progress import render_process_once

    node = _load_target_process(target, uuid_, pk_)

    with suppress_aiida_logging():
        render_process_once(node)

    if node.is_terminated and not node.is_finished_ok:
        raise SystemExit(1)


@cli.command()
@target_argument
@uuid_option
@pk_option
def attach(target: str | None, uuid_: str | None, pk_: int | None) -> None:
    """Attach the live progress display to an already-submitted calculation.

    TARGET is resolved exactly as for `koopmans status`. Displays the
    same live-updating table `koopmans run` shows, until the calculation
    terminates; a calculation that has already terminated is shown once,
    as `koopmans status` would. Exits nonzero if the calculation's root
    process failed.
    """
    from koopmans.aiida.progress import render_process_once, watch_process

    node = _load_target_process(target, uuid_, pk_)

    with suppress_aiida_logging():
        if node.is_terminated:
            render_process_once(node)
        else:
            node = watch_process(node)

    if node.is_terminated and not node.is_finished_ok:
        raise SystemExit(1)


# Shared option for caching
cache_option = click.option(
    "--cache/--no-cache",
    default=True,
    help="Enable AiiDA caching to reuse results from previous identical calculations.",
)

max_procs_option = click.option(
    "--max-procs",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Total MPI ranks allowed concurrently across all running calcs. "
        "Default: physical core count."
    ),
)


def _validate_code_labels(labels: tuple[str, ...], param_hint: str) -> set[str]:
    """Return the given code labels, rejecting any koopmans does not register.

    For options that only change how an already-registered code runs, an
    unknown label can only be a typo.
    """
    from koopmans.aiida.setup.codes import code_specs

    known = code_specs()
    unknown = sorted({label for label in labels if label not in known})
    if unknown:
        raise click.BadParameter(
            f"Unknown code(s): {', '.join(unknown)}. Known codes: {', '.join(sorted(known))}",
            param_hint=param_hint,
        )
    return set(labels)


def _reject_parallel_on_serial_codes(labels: set[str]) -> None:
    """Reject ``--parallel`` for a code that must never run under mpirun."""
    from koopmans.aiida.setup.codes import SERIAL_CODES, parallel_override_error

    for label in sorted(labels):
        if label in SERIAL_CODES:
            raise click.BadParameter(parallel_override_error(label), param_hint="--parallel")


@cli.command()
@click.option(
    "--use-postgres",
    is_flag=True,
    default=False,
    help="Use PostgreSQL instead of SQLite for storage (recommended for production).",
)
@click.option(
    "--procs-per-calc",
    type=int,
    default=None,
    help="MPI ranks each calc launches (default: auto-detect physical cores).",
)
@click.option(
    "--code",
    "code_overrides",
    multiple=True,
    metavar="NAME=PATH",
    help="Specify an executable path for a code, e.g. --code pw=/opt/qe/bin/pw.x",
)
@click.option(
    "--serial",
    "serial_labels",
    multiple=True,
    metavar="NAME",
    help="Register a code to run without mpirun, overriding what its binary declares.",
)
@click.option(
    "--parallel",
    "parallel_labels",
    multiple=True,
    metavar="NAME",
    help="Register a code to run under mpirun, overriding what its binary declares.",
)
@click.option(
    "--migrate/--no-migrate",
    default=True,
    help=(
        "Replace already-registered codes that run the wrong way. Replacing a code "
        "orphans the results cached against it; --no-migrate leaves them as they are."
    ),
)
@max_procs_option
@cache_option
def install(
    use_postgres: bool,
    procs_per_calc: int | None,
    code_overrides: tuple[str, ...],
    serial_labels: tuple[str, ...],
    parallel_labels: tuple[str, ...],
    migrate: bool,
    max_procs: int | None,
    cache: bool,
) -> None:
    """Auto-install the AiiDA backend.

    This command:
    1. Creates an AiiDA profile with SQLite storage (or PostgreSQL with --use-postgres)
    2. Downloads the bundled HyperQueue binary and starts the HQ server + worker
    3. Configures the localhost computer (HyperQueue scheduler)
    4. Detects and registers the executables koopmans runs on PATH
    5. Starts the AiiDA daemon with caching enabled

    Use --code to specify a custom executable path for a code, e.g.:

        koopmans install --code pw=/opt/qe/bin/pw.x --code wannier90=/usr/local/bin/wannier90.x

    Whether each code is launched under mpirun is decided by inspecting its
    binary. Use --serial/--parallel to overrule that, e.g.:

        koopmans install --parallel wannier90

    Rerunning this command also replaces codes registered earlier that run the
    wrong way, which orphans the results cached against them; --no-migrate
    leaves them alone.
    """
    # Parse code overrides into a dict
    explicit_codes: dict[str, str] = {}
    for override in code_overrides:
        if "=" not in override:
            raise click.BadParameter(
                f"Expected NAME=PATH format, got '{override}'", param_hint="--code"
            )
        name, path = override.split("=", 1)
        path = path.strip()
        if not Path(path).is_file():
            raise click.BadParameter(f"Executable not found: {path}", param_hint="--code")
        explicit_codes[name.strip()] = path

    serial = _validate_code_labels(serial_labels, "--serial")
    parallel = _validate_code_labels(parallel_labels, "--parallel")
    both = sorted(serial & parallel)
    if both:
        raise click.BadParameter(
            f"Cannot be both serial and parallel: {', '.join(both)}", param_hint="--serial"
        )
    _reject_parallel_on_serial_codes(parallel)

    click.echo("Setting up koopmans AiiDA backend...")
    click.echo("=" * 60)
    setup_profile(use_postgres=use_postgres)

    # HyperQueue is required for the localhost backend — there is no
    # ``core.direct`` fallback. ``install_hq_binary`` raises if the box
    # isn't supported (non-Linux / non-x86_64) or the download fails,
    # which surfaces to the user as a clear install failure.
    click.echo("\nInstalling HyperQueue...")
    install_hq_binary()
    if not ensure_hq_running(cpus=max_procs):
        raise click.ClickException(
            "Failed to start HyperQueue. The localhost backend requires HQ; "
            "inspect the log under ${AIIDA_CONFIG}/koopmans/ for details."
        )

    setup_computers(
        nprocs=procs_per_calc,
        explicit_codes=explicit_codes,
        serial_labels=sorted(serial),
        parallel_labels=sorted(parallel),
        migrate=migrate,
    )

    # Clean up any input_tmp.in files created by QE executables during version detection
    for tmp_file in Path.cwd().glob("input_tmp*.in"):
        tmp_file.unlink()

    # Start the daemon
    click.echo("")
    _start_daemon_with_caching(cache)

    click.echo("\nInstallation complete!")


@cli.command()
def pseudos() -> None:
    """List the pseudopotential families `workflow.pseudo_library` accepts.

    Families koopmans has installed are marked. The listing itself needs no
    AiiDA profile, so it works before `koopmans install`.
    """
    from koopmans.aiida.setup.pseudos import list_pseudo_families

    list_pseudo_families()


@cli.group()
def backend() -> None:
    """Manage the AiiDA backend."""


@backend.command()
def status() -> None:
    """Show the status of the AiiDA installation."""
    print_status()


@backend.command()
def codes() -> None:
    """List all registered codes."""
    list_codes()


@backend.group()
def daemon() -> None:
    """Manage the AiiDA daemon."""


def _start_daemon_with_caching(cache: bool) -> None:
    """Start the daemon with caching configuration (internal helper)."""
    load_koopmans_profile()

    if is_daemon_running():
        click.echo("Daemon is already running.")
        if cache:
            click.echo("Note: Caching is enabled. Restart the daemon for changes to take effect.")
        return
    click.echo("Starting daemon...")
    if cache:
        click.echo("Caching: enabled")
    if start_daemon(wait=True, cache=cache):
        click.echo("Daemon started successfully.")
    else:
        raise click.ClickException("Failed to start daemon.")


@daemon.command(name="start")
@cache_option
def daemon_start(cache: bool) -> None:
    """Start the AiiDA daemon."""
    _start_daemon_with_caching(cache)


@daemon.command(name="stop")
def daemon_stop() -> None:
    """Stop the AiiDA daemon."""
    load_koopmans_profile()
    if not is_daemon_running():
        click.echo("Daemon is not running.")
        return
    click.echo("Stopping daemon...")
    if stop_daemon():
        click.echo("Daemon stopped successfully.")
    else:
        raise click.ClickException("Failed to stop daemon.")


@daemon.command(name="restart")
@cache_option
@click.pass_context
def daemon_restart(ctx: click.Context, cache: bool) -> None:
    """Restart the AiiDA daemon."""
    ctx.invoke(daemon_stop)
    ctx.invoke(daemon_start, cache=cache)


@daemon.command(name="status")
def daemon_status() -> None:
    """Check if the AiiDA daemon is running."""
    load_koopmans_profile()
    if is_daemon_running():
        click.echo("Daemon is running.")
    else:
        click.echo("Daemon is not running.")


@backend.group()
def hq() -> None:
    """Manage the HyperQueue worker that runs your calculations."""


def _require_hq_binary() -> None:
    """Abort with an install pointer if no ``hq`` binary is available."""
    if hq_binary() is None:
        raise click.ClickException(
            "No HyperQueue binary found. Run 'koopmans install' to download it, "
            "or point KOOPMANS_HQ_BINARY at one you already have."
        )


def _require_one_worker(action: str) -> None:
    """Abort if more than one worker is running, since ``action`` would guess.

    koopmans manages a single worker. Several means someone arranged them by
    hand, and collapsing them would change the machine's capacity silently.
    """
    workers = running_hq_workers()
    if len(workers) > 1:
        listed = ", ".join(f"{w.id} ({w.cpus} CPUs)" for w in workers)
        raise click.ClickException(
            f"{len(workers)} HyperQueue workers are running: {listed}. koopmans "
            f"manages one, and will not {action} several at once. Use 'hq worker "
            f"stop <id>' to leave one running, then repeat this command."
        )


@hq.command(name="start")
@max_procs_option
def hq_start(max_procs: int | None) -> None:
    """Start the HyperQueue worker.

    Brings the HyperQueue server up first if it is down. Does nothing if a
    worker is already running; use 'restart' to change its CPU pool.
    """
    _require_hq_binary()
    if is_hq_worker_running():
        click.echo("HyperQueue worker is already running.")
        print_hq_status()
        return
    click.echo("Starting HyperQueue worker...")
    if not ensure_hq_running(cpus=max_procs):
        raise click.ClickException(
            "Failed to start the HyperQueue worker; inspect the log under "
            "${AIIDA_CONFIG}/koopmans/hq.worker.log for details."
        )
    click.echo("HyperQueue worker started.")
    print_hq_status()


@hq.command(name="stop")
def hq_stop() -> None:
    """Stop the HyperQueue worker.

    Leaves the HyperQueue server up, so the queue is not discarded.
    'koopmans backend uninstall' removes the server as well.

    Check 'hq job summary' first if work is in flight: a task whose worker
    disappears may be retried on the next one, in the same directory.
    """
    _require_hq_binary()
    if not is_hq_worker_running():
        click.echo("HyperQueue worker is not running.")
        return
    _require_one_worker("stop")
    click.echo("Stopping HyperQueue worker...")
    if not stop_hq_worker():
        raise click.ClickException("Failed to stop the HyperQueue worker.")
    click.echo("HyperQueue worker stopped.")


@hq.command(name="restart")
@max_procs_option
def hq_restart(max_procs: int | None) -> None:
    """Restart the HyperQueue worker, optionally resizing its CPU pool.

    Without --max-procs the replacement keeps the pool it had. Starts the
    HyperQueue server too if it is down. The caution under 'stop' about work
    in flight applies here as well.
    """
    _require_hq_binary()
    _require_one_worker("restart")
    click.echo("Restarting HyperQueue worker...")
    if not restart_hq_worker(cpus=max_procs):
        raise click.ClickException(
            "Failed to restart the HyperQueue worker; inspect the log under "
            "${AIIDA_CONFIG}/koopmans/hq.worker.log for details."
        )
    print_hq_status()


@hq.command(name="status")
def hq_status() -> None:
    """Show the state of the HyperQueue server and worker."""
    _require_hq_binary()
    print_hq_status()


@backend.command()
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def uninstall(yes: bool) -> None:
    """Completely remove the AiiDA backend.

    This will delete the AiiDA profile and all associated data including:
    - The database (all calculation history)
    - The file repository
    - Registered computers and codes

    This action cannot be undone!
    """
    if not yes:
        click.confirm(
            "This will permanently delete all koopmans AiiDA data. Continue?",
            abort=True,
        )

    # ``uninstall_backend`` itself handles stopping the AiiDA daemon and
    # the HyperQueue server + worker before deleting the profile. We do
    # nothing here beyond the confirmation prompt so the call works even
    # when the profile is already broken / partially-deleted.
    uninstall_backend()


@cli.group()
def plot() -> None:
    """Draw a figure from one or more finished runs."""


# Shared by every ``plot`` subcommand.
#
# ``types-click`` still constrains ``path_type`` to str/bytes, which click
# itself has not done for years, so each use of it needs the ignore below.
output_option = click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),  # type: ignore[type-var]
    default=None,
    help="Where to write the figure; the extension sets the format.",
)
show_option = click.option(
    "--show",
    is_flag=True,
    default=False,
    help="Open an interactive window instead of writing a file (-o still writes one).",
)
zero_option = click.option(
    "--zero",
    type=click.Choice([kind.value for kind in EnergyZero]),
    default=EnergyZero.VBM.value,
    show_default=True,
    help="Which energy to put at zero. One shift, from the first series that reports it, "
    "is applied to every series on the axes.",
)
data_option = click.option(
    "--data",
    "data_path",
    type=click.Path(dir_okay=False, path_type=Path),  # type: ignore[type-var]
    default=None,
    help="Also write the series the figure was drawn from, as JSON.",
)


def _check_ylim(
    ctx: click.Context, param: click.Parameter, value: tuple[float, float] | None
) -> tuple[float, float] | None:
    """Reject a range that frames nothing."""
    if value is not None and value[0] >= value[1]:
        raise click.BadParameter(
            f"MIN must be below MAX; got {value[0]} and {value[1]}.", ctx=ctx, param=param
        )
    return value


def _check_styles(
    ctx: click.Context, param: click.Parameter, value: tuple[str, ...]
) -> tuple[str, ...]:
    """Reject a style matplotlib cannot read, before any run is looked up."""
    from koopmans.plotting import StyleError, check_style

    for style in value:
        try:
            check_style(style)
        except StyleError as exc:
            raise click.BadParameter(
                f"{exc}. A format string combines a color, a marker and a line style: "
                "'k-' is a black line, 'rx' red crosses, 'C1--' a dashed line in the "
                "second automatic color.",
                ctx=ctx,
                param=param,
            ) from exc
    return value


ylim_option = click.option(
    "--ylim",
    nargs=2,
    type=float,
    default=None,
    callback=_check_ylim,
    metavar="MIN MAX",
    help="Show only this range of the energy axis, in the units it is drawn in "
    "and measured from the zero --zero sets. Defaults to every band in full.",
)


class _PositionalAwareOption(click.Option):
    """A ``multiple=True`` option that records how many folders preceded each use.

    click hands a ``multiple=True`` option's callback the fully-parsed tuple
    of values, with no memory of where among a ``nargs=-1`` argument's
    positionals each one was written — exactly what ``_FolderPairingCommand``
    needs to pair a ``--style``/``--label`` with the folder it followed.
    click's own parser already knows this while it parses:
    :class:`click.parser.ParsingState` builds ``largs`` up as a running list
    of the positional tokens seen so far, in the same single left-to-right
    pass that calls an option's low-level :meth:`click.parser.Option.process`
    the moment its flag is read — so ``len(state.largs)`` at that moment is
    the number of folders written before this occurrence. This wraps the
    ``process`` callable that :meth:`click.Option.add_to_parser` registers
    for this option's flags, to record that count for every occurrence in
    encounter order — the same order click builds the ``multiple=True``
    tuple in, so the two line up positionally once parsing finishes.

    ``ParsingState.largs`` and ``click.parser.Option.process`` are
    undocumented parser internals; ``test_positional_recording_canary``
    fails loudly, naming what moved, if a future click stops exposing them.
    """

    def positions_key(self) -> str:
        """Return the ``ctx.meta`` key this option's recorded positions live under."""
        return f"_folder_pairing_positions:{self.name}"

    def add_to_parser(self, parser: click.parser.OptionParser, ctx: click.Context) -> None:
        """Register the option, wrapping its parser hook to record positions."""
        super().add_to_parser(parser, ctx)
        positions: list[int] = []
        ctx.meta[self.positions_key()] = positions

        seen: set[int] = set()
        for internal in (*parser._long_opt.values(), *parser._short_opt.values()):
            if internal.dest != self.name or id(internal) in seen:
                continue
            seen.add(id(internal))
            try:
                original_process = internal.process
            except AttributeError as exc:
                raise RuntimeError(
                    "click.parser.Option no longer exposes 'process'; "
                    "_PositionalAwareOption's position-recording recipe needs "
                    "updating for this click version."
                ) from exc
            internal.process = _recording_process(  # type: ignore[assignment]
                original_process, positions
            )


def _recording_process(
    process: Callable[[Any, click.parser.ParsingState], None], positions: list[int]
) -> Callable[[Any, click.parser.ParsingState], None]:
    """Wrap a parser option's ``process`` to record its folder count first."""

    def wrapped(value: Any, state: click.parser.ParsingState) -> None:
        """Record the folders seen so far, then hand the value to click."""
        try:
            largs = state.largs
        except AttributeError as exc:
            raise RuntimeError(
                "click.parser.ParsingState no longer exposes 'largs'; "
                "_PositionalAwareOption's position-recording recipe needs "
                "updating for this click version."
            ) from exc
        positions.append(len(largs))
        process(value, state)

    return wrapped


class _FolderPairingCommand(click.Command):
    """A command whose ``--style``/``--label`` pair with the folder argument.

    click gathers a ``nargs=-1`` argument and a ``multiple=True`` option into
    two separate lists, so by the time a callback sees them the order they
    were interleaved in is gone — only "the nth folder" and "the nth style"
    remain, not which folder each style was written after.
    :class:`_PositionalAwareOption` recovers that, per paired option, from
    click's own parser; this command turns each recovered position into a
    binding:

    - given once per folder, the values pair with the folders positionally,
      in listing order, wherever among the folders they were written.
    - given for fewer folders than that, each value binds to the folder it
      immediately followed; folders it did not follow one for stay unstyled
      or unlabelled.
    - given for more folders than that, there is no folder left for the
      extra values to mean, and the command refuses.

    Everything else — tokenizing, ``--``, ``--help``, error formatting — is
    left to click.
    """

    #: Parameter names paired with the folder argument, one per folder.
    _paired_params = ("styles", "labels")
    _folder_param = "folders"

    @staticmethod
    def _bind_values(
        flag: str,
        occurrences: list[tuple[int, str]],
        folder_tokens: tuple[Path, ...],
    ) -> tuple[str | None, ...] | None:
        """Return one value per folder for a paired option, or ``None`` if unused.

        As many values as folders pair positionally, in listing order,
        regardless of where among the folders each was written. Fewer values
        than folders binds each one to the folder it immediately followed;
        ``None`` marks a folder that got none of its own, distinct from one
        given an explicit empty string (a no-op matplotlib format string,
        still a value the user typed). More values than folders, or a value
        before any folder, has no folder left to mean and is refused.

        :raises click.UsageError: if there were more values than folders, a
            value came before any folder, or two values bound to one folder
            — the last two only matter when the count fell short, since an
            exact count pairs by position and ignores where each was written.
        """
        nfolders = len(folder_tokens)
        count = len(occurrences)
        if count == 0:
            return None
        if count == nfolders:
            return tuple(value for _, value in occurrences)
        if count > nfolders:
            raise click.UsageError(
                f"{count} {flag} values were given for {nfolders} folder(s). Give "
                f"one {flag} per folder to pair them in listing order, fewer with "
                f"each one just after the folder it names, or none at all."
            )

        bound: dict[int, str] = {}
        for index, value in occurrences:
            if index < 0:
                raise click.UsageError(
                    f"{flag} must follow the folder it applies to — give a folder "
                    f"before the first {flag}, not a bare {flag} up front."
                )
            if index in bound:
                raise click.UsageError(
                    f"{flag} was already given for {folder_tokens[index]!r} "
                    f"({bound[index]!r}); write only one {flag} per folder, or one "
                    f"{flag} per folder overall to pair them by listing order instead."
                )
            bound[index] = value
        return tuple(bound.get(i) for i in range(nfolders))

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Parse normally, then rebind each paired option to its folder(s)."""
        rv = super().parse_args(ctx, args)

        params = {p.name: p for p in self.get_params(ctx) if p.name is not None}
        folder_tokens: tuple[Path, ...] = ctx.params[self._folder_param]

        for name in self._paired_params:
            param = params[name]
            if not isinstance(param, _PositionalAwareOption):
                raise TypeError(f"{name!r} must be declared with cls=_PositionalAwareOption.")
            positions = ctx.meta.pop(param.positions_key(), [])
            values: tuple[str, ...] = ctx.params[name]
            occurrences = list(zip((p - 1 for p in positions), values, strict=True))
            ctx.params[name] = self._bind_values(param.opts[0], occurrences, folder_tokens)

        return rv


@plot.command(cls=_FolderPairingCommand)
@click.argument(
    "folders",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),  # type: ignore[type-var]
)
@output_option
@show_option
@zero_option
@data_option
@ylim_option
@click.option(
    "--label",
    "labels",
    cls=_PositionalAwareOption,
    multiple=True,
    metavar="TEXT",
    help="Name a folder on the legend. One per folder pairs them in listing "
    "order; fewer than that, each names the folder it was written just after, "
    "and a folder with none of its own keeps its derived name. A folder drawn "
    "as several curves keeps what tells them apart, such as the spin channel.",
)
@click.option(
    "--style",
    "styles",
    cls=_PositionalAwareOption,
    multiple=True,
    metavar="FORMAT",
    callback=_check_styles,
    help="Draw a folder in a matplotlib format string, such as 'x' for "
    "crosses, 'k--' for a dashed black line or '-' for a plain one. One per "
    "folder pairs them in listing order; fewer than that, each draws the "
    "folder it was written just after, and a folder with none of its own is "
    "drawn as the figure would draw it on its own. A color the string names "
    "replaces the one this command would have chosen, and every curve the "
    "folder draws is drawn the same way — pass a calculation directory of "
    "its own to draw one result differently from its siblings.",
)
def bandstructure(
    folders: tuple[Path, ...],
    output_path: Path | None,
    show: bool,
    zero: str,
    data_path: Path | None,
    ylim: tuple[float, float] | None,
    labels: tuple[str | None, ...],
    styles: tuple[str | None, ...],
) -> None:
    """Draw the band structures of finished runs on one set of axes.

    FOLDERS are directories `koopmans run` wrote, or single calculation
    directories inside them. Each is taken as given and nothing beneath it is
    searched, so a step that ran wannier90 twice is never chosen between on
    your behalf; a directory that names no run of its own lists the ones under
    it that can be drawn. Naming a run and a step inside it draws that step
    twice — pass either the run or its steps. Every band structure across all
    the folders is drawn, so a DFT run and a Koopmans run given together
    overlay, referenced to a single energy zero. Each is named after the step
    that produced it unless --label names it, and --style says how it is
    drawn; the recommended way to write either is right after its own
    folder, as in the examples below, but that is only where it counts: with
    one --style (or --label) for every folder, they pair with the folders in
    listing order wherever among the folders each was written, and with
    fewer than that, each pairs with the folder it immediately followed. A
    folder left with neither keeps the figure's own choice of name and
    appearance — only the reference bands are styled here, the wannierized
    ones are left as the figure would draw them:

    \b
        koopmans plot bandstructure \\
            si/02-bands --style rx \\
            si/03-wannierize_emp_1/01-wannier90/03-wannier90 \\
            si/04-wannierize_occ_1/01-wannier90/03-wannier90

    --label works the same way, and can be mixed with --style on the same
    folder, or with fewer values than --style has:

    \b
        koopmans plot bandstructure dft --label DFT ki --style k-- --label "KI@LDA"

    Writing every --style (or --label) after every folder reads the same as
    interleaving them, as long as there is one for each:

    \b
        koopmans plot bandstructure pw wannier --style x --style -

    Writing more --style (or --label) values than there are folders is
    refused, and so is writing one before any folder when there are fewer
    values than folders — in either case there is no folder left for it to
    mean.

    Each is drawn in a color of this command's choosing unless --style says
    how, as crosses at the k-points pw.x computed and a line through the
    wannier90 interpolation of them:

        koopmans plot bandstructure pw --style x wannier --style -

    One style covers everything its folder draws, so drawing the results of a
    single run differently from each other means naming their calculation
    directories. A wannierize run has one wannier90 calculation per block, and
    a block folder that carries no run of its own will name it for you:

    \b
        koopmans plot bandstructure \\
            zno/02-wannierize/01-bands --style rx \\
            zno/02-wannierize/02-wannierize_emp/01-wannier90/03-wannier90 --style b-

    To export one band structure in Grace, gnuplot or dat form instead, use
    `verdi data core.bands export`: those exporters take one node at a time,
    and so lose both the overlay and its shared zero.
    """
    from koopmans.plotting import (
        NoEnergyZeroError,
        PathMismatchError,
        PlottingError,
        apply_energy_zero,
        check_paths_agree,
        describe_energy_zero,
        render_band_structures,
        resolve_band_series,
        write_series_json,
    )

    load_koopmans_profile()

    kind = EnergyZero(zero)
    try:
        series, warnings = resolve_band_series(folders, labels, styles)
        check_paths_agree(series)
        value, reference = apply_energy_zero(series, kind)
    except (PlottingError, PathMismatchError, NoEnergyZeroError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    for warning in warnings:
        click.echo(f"Warning: {warning}", err=True)

    caption = describe_energy_zero(kind, value, reference, (reference or series[0]).units)

    if data_path is not None:
        write_series_json(series, data_path)
        click.echo(f"Wrote {data_path} ({len(series)} series)")

    target = output_path if output_path is not None or show else Path("bandstructure.png")
    # Naming a series is asking for it to be named on the figure, so an
    # explicit label brings the legend back for a single curve.
    render_band_structures(
        series,
        output_path=target,
        show=show,
        zero=kind,
        ylim=ylim,
        legend=True if labels else None,
    )
    if target is not None:
        click.echo(f"Wrote {target} ({len(series)} series, {caption})")


def main() -> None:
    """Entry point for the CLI."""
    cli()


def header() -> str:
    """Return the output header."""
    from koopmans.version import VERSION

    lines = [
        "",
        click.style("koopmans", bold=True),
        click.style(  # type: ignore[call-arg]
            "Koopmans spectral functional calculations with Quantum ESPRESSO", italic=True
        ),
        "",
        f"📦 Version: {VERSION}",
        "🧑 Authors: Edward Linscott, Nicola Colonna, Riccardo De Gennaro, Ngoc Linh Nguyen, "
        "Giovanni Borghi, Andrea Ferretti, Ismaila Dabo, and Nicola Marzari",
        "📚 Documentation: https://koopmans-functionals.org",
        "❓ Support: https://groups.google.com/g/koopmans-users",
        "🐛 Report a bug: https://github.com/epfl-theos/koopmans/issues/new",
        "",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    main()
