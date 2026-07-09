"""AiiDA profile creation for the koopmans backend.

Mirrors ``verdi presto``: SQLite storage by default, optional PostgreSQL
or RabbitMQ broker if available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)

PROFILE_NAME = "koopmans"


def profile_exists() -> bool:
    """Check if the koopmans profile already exists."""
    from aiida.manage.configuration import get_config

    config = get_config()
    return PROFILE_NAME in config.profile_names


def setup_profile(*, use_postgres: bool = False) -> None:
    """Set up the AiiDA profile for koopmans.

    Args:
        use_postgres: If True, use PostgreSQL instead of SQLite for storage.
    """
    from aiida.manage.configuration import create_profile, get_config, load_profile

    if profile_exists():
        click.echo(f"Profile '{PROFILE_NAME}' already exists.")
        load_profile(PROFILE_NAME)
        return

    click.echo(f"Creating AiiDA profile '{PROFILE_NAME}'...")

    config = get_config()

    if use_postgres:
        click.echo("  Detecting PostgreSQL configuration...")
        try:
            from aiida.cmdline.commands.cmd_presto import (
                detect_postgres_config,
                get_default_presto_profile_name,
            )

            storage_config = detect_postgres_config(
                profile_name=get_default_presto_profile_name(),  # type: ignore[no-untyped-call]
                postgres_hostname="localhost",
                postgres_port=5432,
                postgres_username="postgres",
                postgres_password="",
                non_interactive=False,
            )

            storage_backend = "core.psql_dos"
            click.echo("  PostgreSQL configured successfully.")
        except ConnectionError as exc:
            raise click.ClickException(
                f"PostgreSQL detection failed: {exc}. Ensure PostgreSQL is running and accessible."
            ) from exc
    else:
        storage_backend = "core.sqlite_dos"
        storage_config = {
            "filepath": str(Path(config.dirpath) / PROFILE_NAME / "storage"),
        }
        click.echo("  Using SQLite storage backend.")

    broker_backend = None
    broker_config = None
    try:
        from aiida.brokers.rabbitmq.defaults import detect_rabbitmq_config

        broker_config = detect_rabbitmq_config()
        broker_backend = "core.rabbitmq"
        click.echo("  RabbitMQ broker detected.")
    except Exception:
        click.echo("  RabbitMQ not detected (daemon features will be limited).")

    create_profile(
        config,
        name=PROFILE_NAME,
        email="koopmans@localhost",
        storage_backend=storage_backend,
        storage_config=storage_config,
        broker_backend=broker_backend,
        broker_config=broker_config,
    )

    config.set_default_profile(PROFILE_NAME)  # type: ignore[no-untyped-call]
    config.store()  # type: ignore[no-untyped-call]

    load_profile(PROFILE_NAME)

    click.echo(f"Successfully created profile '{PROFILE_NAME}'.")


def load_koopmans_profile() -> None:
    """Load the koopmans AiiDA profile.

    Raises an error if the profile doesn't exist.
    """
    from aiida.manage.configuration import load_profile

    if not profile_exists():
        raise click.ClickException(
            f"AiiDA profile '{PROFILE_NAME}' not found. "
            "Please run 'koopmans install' first to set up the AiiDA backend."
        )

    load_profile(PROFILE_NAME)


def koopmans_dir() -> Path:
    """Return the koopmans-managed dir under the AiiDA config dir.

    Holds the bundled HQ binary, the HQ server / worker pid + log files,
    and similar koopmans-only state. Mirrors AiiDA's own daemon-files
    convention.
    """
    from aiida.manage.configuration import get_config

    config = get_config()
    path = Path(config.dirpath) / "koopmans"
    path.mkdir(parents=True, exist_ok=True)
    return path
