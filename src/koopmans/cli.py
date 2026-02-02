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

import click

from koopmans.aiida_setup import (
    list_codes,
    load_koopmans_profile,
    print_status,
    setup_computers,
    setup_profile,
)

__all__ = [
    "main",
    "cli",
]


@click.group()
@click.version_option()
def cli() -> None:
    """Automated Koopmans functional calculations and workflows."""


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
def run(input_file: str) -> None:
    """Run a koopmans calculation from an input file.

    INPUT_FILE is the path to a YAML input file describing the calculation.
    """
    load_koopmans_profile()
    click.echo(f"Loading input from {input_file}...")
    click.echo("Running koopmans calculation...")
    # TODO: Implement actual calculation logic


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


if __name__ == "__main__":
    main()
