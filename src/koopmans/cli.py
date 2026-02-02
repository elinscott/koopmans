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

__all__ = [
    "run",
]


@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable or disable debug mode')
def cli():
    pass

@cli.command()
def run():
    """Run the main application."""
    click.echo("Running the main application...")