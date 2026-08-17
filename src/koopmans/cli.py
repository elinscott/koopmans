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

from __future__ import annotations

import logging
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

import click

from koopmans.aiida.dumping import dump_workgraph, trained_model_output
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


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
def run(input_file: str) -> None:
    """Run a koopmans calculation from an input file.

    INPUT_FILE is the path to a YAML or JSON input file describing the calculation.
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
            run_with_progress(wg)
    except Exception as exc:
        advice = advice_for(exc)
        if advice is not None:
            exc.add_note(advice)
        raise

    if wg.process is not None:
        dump_workgraph(wg.process, output_path=input_path.parent / input_path.stem, overwrite=True)
        model_node = trained_model_output(wg.process)
        if model_node is not None:
            click.echo(
                f"Trained model stored as node {model_node.pk} ({model_node.uuid}) — "
                f"reference it via `ml: {{model: {model_node.pk}}}`."
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
    from datetime import datetime

    from koopmans.aiida.anchor import AnchorEntry, anchor_path_for_input, append_anchor_entry
    from koopmans.aiida.setup.profile import PROFILE_NAME
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

    if node.pk is None:
        raise click.ClickException("The submitted process was never stored, so it has no id.")

    anchor_path = anchor_path_for_input(input_path)
    entry = AnchorEntry(
        uuid=node.uuid,
        pk=node.pk,
        input=input_path.name,
        profile=PROFILE_NAME,
        submitted=datetime.now(UTC).isoformat(),
    )
    try:
        append_anchor_entry(anchor_path, entry)
    except OSError as exc:
        # The daemon already has the job; losing the run file only loses
        # the *shortcut* back to it, not the submission itself.
        raise click.ClickException(
            f"Workflow submitted as pk {node.pk} ({node.uuid}), but {anchor_path} could "
            f"not be written ({exc}). Recover with `koopmans status --uuid {node.uuid}`."
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
    try:
        node = (
            orm.load_node(uuid=resolved.uuid)
            if resolved.uuid is not None
            else orm.load_node(pk=resolved.pk)
        )
    except NotExistent as exc:
        identifier = resolved.uuid if resolved.uuid is not None else resolved.pk
        raise click.ClickException(
            f"No AiiDA node found for {identifier!r}. It may have been deleted; check "
            "`verdi process list -a` for what is still in the database."
        ) from exc
    if not isinstance(node, orm.ProcessNode):
        raise click.ClickException(
            f"Node {node.pk} is not a calculation; it holds a {type(node).__name__}."
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


@plot.command()
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
    multiple=True,
    metavar="TEXT",
    help="Name a folder on the legend; repeat once per folder, in the order the "
    "folders are listed. A folder drawn as several curves keeps what tells them "
    "apart, such as the spin channel.",
)
@click.option(
    "--style",
    "styles",
    multiple=True,
    metavar="FORMAT",
    callback=_check_styles,
    help="Draw a curve in a matplotlib format string, such as 'x' for crosses, "
    "'k--' for a dashed black line or '-' for a plain one. Repeat once per "
    "folder, in the order the folders are listed, which draws everything a "
    "folder contributes the same way; or once per curve, in the order they are "
    "drawn, which draws a folder's own curves differently from each other. A "
    "color the string names replaces the one this command would have chosen. "
    "--label stays one per folder either way.",
)
def bandstructure(
    folders: tuple[Path, ...],
    output_path: Path | None,
    show: bool,
    zero: str,
    data_path: Path | None,
    ylim: tuple[float, float] | None,
    labels: tuple[str, ...],
    styles: tuple[str, ...],
) -> None:
    """Draw the band structures of finished runs on one set of axes.

    FOLDERS are directories `koopmans run` wrote. Every band structure across
    all of them is drawn, so a DFT run and a Koopmans run given together
    overlay, referenced to a single energy zero. Each is named after the step
    that produced it unless --label names it:

        koopmans plot bandstructure dft ki --label DFT --label "KI@LDA"

    Each is drawn in a color of this command's choosing unless --style says
    how. One style per folder draws everything that folder contributes alike;
    one per curve draws them differently, in the order they are drawn. A
    wannierize run of silicon draws its pw.x bands and one interpolation per
    block, so crosses at the k-points pw.x computed and a line through each
    interpolation of them read:

        koopmans plot bandstructure si --style rx --style b- --style b-

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
