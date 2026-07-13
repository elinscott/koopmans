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

import logging
from pathlib import Path

import click

from koopmans.aiida.dumping import dump_workgraph
from koopmans.aiida.progress import run_with_progress
from koopmans.aiida.setup.codes import list_codes
from koopmans.aiida.setup.daemon import is_daemon_running, start_daemon, stop_daemon
from koopmans.aiida.setup.hq import ensure_hq_running, install_hq_binary
from koopmans.aiida.setup.orchestrate import (
    print_status,
    setup_computers,
    uninstall_backend,
)
from koopmans.aiida.setup.profile import load_koopmans_profile, setup_profile
from koopmans.aiida.utils import suppress_aiida_logging
from koopmans.input_file import read_input_file

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
    from koopmans.aiida.workflows import build_workgraph

    input_path = Path(input_file)

    # Print the header
    click.echo(header())

    # Parse input file
    koopmans_input = read_input_file(input_path)

    # Load AiiDA profile
    load_koopmans_profile()

    # Build the appropriate workgraph based on task
    wg = build_workgraph(koopmans_input)

    with suppress_aiida_logging():
        run_with_progress(wg)

    if wg.process is not None:
        dump_workgraph(wg.process, output_path=input_path.parent / input_path.stem, overwrite=True)


# Shared option for caching
cache_option = click.option(
    "--cache/--no-cache",
    default=True,
    help="Enable AiiDA caching to reuse results from previous identical calculations.",
)


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
    "--max-procs",
    type=int,
    default=None,
    help=(
        "Total MPI ranks allowed concurrently across all running calcs. "
        "Default: physical core count."
    ),
)
@cache_option
def install(
    use_postgres: bool,
    procs_per_calc: int | None,
    code_overrides: tuple[str, ...],
    max_procs: int | None,
    cache: bool,
) -> None:
    """Auto-install the AiiDA backend.

    This command:
    1. Creates an AiiDA profile with SQLite storage (or PostgreSQL with --use-postgres)
    2. Downloads the bundled HyperQueue binary and starts the HQ server + worker
    3. Configures the localhost computer (HyperQueue scheduler)
    4. Detects and registers Quantum ESPRESSO executables on PATH
    5. Starts the AiiDA daemon with caching enabled

    Use --code to specify a custom executable path for a code, e.g.:

        koopmans install --code pw=/opt/qe/bin/pw.x --code wannier90=/usr/local/bin/wannier90.x
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

    click.echo("Setting up koopmans AiiDA backend...")
    click.echo("=" * 60)
    setup_profile(use_postgres=use_postgres)

    # HyperQueue is required for the localhost backend — there is no
    # ``core.direct`` fallback. ``install_hq_binary`` raises if the box
    # isn't supported (non-Linux / non-x86_64) or the download fails,
    # which surfaces to the user as a clear install failure.
    click.echo("\nInstalling HyperQueue...")
    install_hq_binary()
    if not ensure_hq_running(resources=max_procs):
        raise click.ClickException(
            "Failed to start HyperQueue. The localhost backend requires HQ; "
            "inspect the log under ${AIIDA_CONFIG}/koopmans/ for details."
        )

    setup_computers(nprocs=procs_per_calc, explicit_codes=explicit_codes)

    # Clean up any input_tmp.in files created by QE executables during version detection
    for tmp_file in Path.cwd().glob("input_tmp*.in"):
        tmp_file.unlink()

    # Start the daemon
    click.echo("")
    _start_daemon_with_caching(cache)

    click.echo("\nInstallation complete!")


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
