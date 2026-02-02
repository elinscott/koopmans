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

import click

from koopmans.aiida.setup import (
    list_codes,
    load_koopmans_profile,
    print_status,
    setup_computers,
    setup_profile,
)
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
def run(input_file: str, dry_run: bool) -> None:
    """Run a koopmans calculation from an input file.

    INPUT_FILE is the path to a YAML or JSON input file describing the calculation.
    """
    from aiida import orm
    from aiida_koopmans.workgraphs import scf_bands_workgraph

    from koopmans.aiida import convert_koopmans_input
    from koopmans.input_file.workflow import Task

    # Print the header
    click.echo(header())

    # Parse input file
    click.echo(f"Loading input from {input_file}...")
    koopmans_input = read_input_file(input_file)

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
    click.echo("Converting input to AiiDA data nodes...")
    aiida_data = convert_koopmans_input(koopmans_input)

    # Build the appropriate workgraph based on task
    task = koopmans_input.workflow.task
    click.echo(f"Building workgraph for task: {task.value}")

    if task in (Task.WANNIERIZE, Task.DFT_BANDS):
        wg = scf_bands_workgraph.build(
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

    wg.to_html("test.html")

    wg.run()


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
