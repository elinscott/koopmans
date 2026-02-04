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

from koopmans.aiida.setup import (
    is_daemon_running,
    list_codes,
    load_koopmans_profile,
    print_status,
    setup_computers,
    setup_profile,
    start_daemon,
    stop_daemon,
    uninstall_backend,
)
from koopmans.aiida.dumping import dump_workgraph
from koopmans.aiida.progress import run_with_progress
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
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Parse input and build workgraph without submitting.",
)
@click.option(
    "--cache/--no-cache",
    default=False,
    help="Enable AiiDA caching to reuse results from previous identical calculations.",
)
def run(input_file: str, dry_run: bool, cache: bool) -> None:
    """Run a koopmans calculation from an input file.

    INPUT_FILE is the path to a YAML or JSON input file describing the calculation.
    """
    from aiida import orm
    from aiida_koopmans.workgraphs import PwBandsTaskViaBuilder

    from koopmans.aiida import convert_koopmans_input_for_builder
    from koopmans.input_file.workflow import Task

    input_path = Path(input_file)

    # Print the header
    click.echo(header())

    # Parse input file
    koopmans_input = read_input_file(input_path)

    # Load AiiDA profile
    load_koopmans_profile()

    # Get the pw.x code
    try:
        code = orm.load_code("pw@localhost")
    except Exception as exc:
        raise click.ClickException(
            f"Could not load pw.x code: {exc}\n"
            "Please run 'koopmans install' first to set up the AiiDA backend."
        ) from exc

    # Convert input to AiiDA data nodes
    aiida_data = convert_koopmans_input_for_builder(koopmans_input)

    # Build the appropriate workgraph based on task
    task = koopmans_input.workflow.task
    if task in (Task.WANNIERIZE, Task.DFT_BANDS):
        wg = PwBandsTaskViaBuilder.build(
            code=code,
            **aiida_data,
        )
    else:
        raise click.ClickException(
            f"Task '{task.value}' is not yet implemented. Supported tasks: wannierize, dft_bands"
        )

    if dry_run:
        click.echo(f"Dry run: workgraph '{wg.name}' built successfully.")
        click.echo(f"  Tasks: {[t.name for t in wg.tasks]}")
        return

    # Enable caching at the profile level if requested (context managers don't affect daemon)
    config = None
    profile_name = None
    original_caching = None
    if cache:
        from aiida.manage import get_config

        config = get_config()
        profile_name = config.default_profile_name
        original_caching = config.get_option("caching.default_enabled", scope=profile_name)
        config.set_option("caching.default_enabled", True, scope=profile_name)
        config.store()

    try:
        with suppress_aiida_logging():
            run_with_progress(wg)
    finally:
        # Restore original caching setting
        if config is not None and profile_name is not None:
            config.set_option("caching.default_enabled", original_caching, scope=profile_name)
            config.store()

    dump_workgraph(wg.process, output_path=input_path.parent / input_path.stem, overwrite=True)


@cli.command()
@click.option(
    "--use-postgres",
    is_flag=True,
    default=False,
    help="Use PostgreSQL instead of SQLite for storage (recommended for production).",
)
def install(use_postgres: bool) -> None:
    """Auto-install the AiiDA backend.

    This command:
    1. Creates an AiiDA profile with SQLite storage (or PostgreSQL with --use-postgres)
    2. Configures the localhost computer
    3. Detects and registers Quantum ESPRESSO executables on PATH
    """
    click.echo("Setting up koopmans AiiDA backend...")
    click.echo("=" * 60)
    setup_profile(use_postgres=use_postgres)
    setup_computers()
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


@daemon.command(name="start")
def daemon_start() -> None:
    """Start the AiiDA daemon."""
    load_koopmans_profile()
    if is_daemon_running():
        click.echo("Daemon is already running.")
        return
    click.echo("Starting daemon...")
    if start_daemon(wait=True):
        click.echo("Daemon started successfully.")
    else:
        raise click.ClickException("Failed to start daemon.")


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
