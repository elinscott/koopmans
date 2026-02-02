"""Automatic AiiDA setup for koopmans.

This module handles automatic configuration of AiiDA profiles, computers, and codes
so that users can run koopmans calculations without needing AiiDA knowledge.

Uses the same approach as `verdi presto` for profile and computer setup.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from aiida.orm import Computer, InstalledCode

# Quantum ESPRESSO and related executables that koopmans needs
# Maps executable name to the AiiDA plugin entry point
QE_EXECUTABLES: dict[str, str] = {
    "pw.x": "quantumespresso.pw",
    "ph.x": "quantumespresso.ph",
    "pp.x": "quantumespresso.pp",
    "projwfc.x": "quantumespresso.projwfc",
    "dos.x": "quantumespresso.dos",
    "wannier90.x": "wannier90.wannier90",
    "pw2wannier90.x": "quantumespresso.pw2wannier90",
    "kcw.x": "quantumespresso.kcw",
    "wann2kc.x": "quantumespresso.wann2kc",
    "kc_screen.x": "quantumespresso.kc_screen",
    "kc_ham.x": "quantumespresso.kc_ham",
}

PROFILE_NAME = "koopmans"
COMPUTER_LABEL = "localhost"


def profile_exists() -> bool:
    """Check if the koopmans profile already exists."""
    from aiida.manage.configuration import get_config

    config = get_config()
    return PROFILE_NAME in config.profile_names


def setup_profile(*, use_postgres: bool = False) -> None:
    """Set up the AiiDA profile for koopmans.

    Uses the same approach as `verdi presto`: creates a profile with SQLite
    storage backend (core.sqlite_dos) by default, or PostgreSQL if requested.
    Optionally configures RabbitMQ if available.

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

    # Storage configuration - reuse presto's detection for postgres
    if use_postgres:
        click.echo("  Detecting PostgreSQL configuration...")
        try:
            from aiida.cmdline.commands.cmd_presto import detect_postgres_config

            storage_config = detect_postgres_config()
            storage_backend = "core.psql_dos"
            click.echo("  PostgreSQL configured successfully.")
        except ConnectionError as exc:
            raise click.ClickException(
                f"PostgreSQL detection failed: {exc}. "
                "Ensure PostgreSQL is running and accessible."
            ) from exc
    else:
        storage_backend = "core.sqlite_dos"
        storage_config = {
            "filepath": str(Path(config.dirpath) / PROFILE_NAME / "storage"),
        }
        click.echo("  Using SQLite storage backend.")

    # Try to detect RabbitMQ broker (optional, like verdi presto)
    broker_backend = None
    broker_config = None
    try:
        from aiida.brokers.rabbitmq.defaults import detect_rabbitmq_config

        broker_config = detect_rabbitmq_config()
        broker_backend = "core.rabbitmq"
        click.echo("  RabbitMQ broker detected.")
    except Exception:
        click.echo("  RabbitMQ not detected (daemon features will be limited).")

    # Create the profile using AiiDA's create_profile (same as verdi presto)
    create_profile(
        config,
        name=PROFILE_NAME,
        email="koopmans@localhost",
        storage_backend=storage_backend,
        storage_config=storage_config,
        broker_backend=broker_backend,
        broker_config=broker_config,
    )

    # Set as default and store config
    config.set_default_profile(PROFILE_NAME)
    config.store()

    # Load the profile
    load_profile(PROFILE_NAME)

    click.echo(f"Successfully created profile '{PROFILE_NAME}'.")


def load_koopmans_profile() -> None:
    """Load the koopmans AiiDA profile.

    Raises an error if the profile doesn't exist (user should run 'koopmans install' first).
    """
    from aiida.manage.configuration import load_profile

    if not profile_exists():
        raise click.ClickException(
            f"AiiDA profile '{PROFILE_NAME}' not found. "
            "Please run 'koopmans install' first to set up the AiiDA backend."
        )

    load_profile(PROFILE_NAME)


def computer_exists() -> bool:
    """Check if the localhost computer already exists."""
    from aiida import orm

    try:
        orm.load_computer(COMPUTER_LABEL)
        return True
    except Exception:
        return False


def get_localhost_computer() -> Computer:
    """Get or create the localhost computer.

    Uses the same configuration as `verdi presto`: local transport, direct scheduler.
    """
    from aiida import orm
    from aiida.manage.configuration import get_config

    if computer_exists():
        return orm.load_computer(COMPUTER_LABEL)

    click.echo(f"Creating computer '{COMPUTER_LABEL}'...")

    config = get_config()

    # Create workdir in the same location as verdi presto
    workdir = Path(config.dirpath) / "scratch" / PROFILE_NAME
    workdir.mkdir(parents=True, exist_ok=True)

    # Create computer with same settings as verdi presto
    computer = orm.Computer(
        label=COMPUTER_LABEL,
        description="Localhost computer configured by koopmans",
        hostname="localhost",
        transport_type="core.local",
        scheduler_type="core.direct",
        workdir=str(workdir),
    )
    computer.store()

    # Configure with presto-style settings
    computer.configure()
    computer.set_minimum_job_poll_interval(0.0)
    computer.set_default_mpiprocs_per_machine(1)

    click.echo(f"Successfully created computer '{COMPUTER_LABEL}'.")
    return computer


def find_executable(name: str) -> str | None:
    """Find an executable on the system PATH.

    Args:
        name: Name of the executable to find.

    Returns:
        Absolute path to the executable, or None if not found.
    """
    path = shutil.which(name)
    if path:
        return str(Path(path).resolve())
    return None


def get_executable_version(path: str) -> str | None:
    """Try to get the version of a Quantum ESPRESSO executable.

    Args:
        path: Path to the executable.

    Returns:
        Version string if detected, None otherwise.
    """
    import re

    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        output = result.stdout + result.stderr
        match = re.search(r"v?(\d+\.\d+(?:\.\d+)?)", output)
        if match:
            return match.group(1)
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        pass
    return None


def code_exists(label: str) -> bool:
    """Check if a code with the given label exists."""
    from aiida import orm

    try:
        orm.load_code(label)
        return True
    except Exception:
        return False


def setup_code(
    executable_name: str,
    executable_path: str,
    plugin: str,
    computer: Computer,
) -> InstalledCode | None:
    """Set up an AiiDA code for an executable.

    Args:
        executable_name: Name of the executable (e.g., 'pw.x').
        executable_path: Absolute path to the executable.
        plugin: AiiDA plugin entry point (e.g., 'quantumespresso.pw').
        computer: The computer to associate the code with.

    Returns:
        The created Code object, or None if it already exists.
    """
    from aiida.orm import InstalledCode

    # Create a label from the executable name (remove .x suffix)
    label = executable_name.replace(".x", "")

    if code_exists(f"{label}@{computer.label}"):
        click.echo(f"  Code '{label}@{computer.label}' already exists, skipping.")
        return None

    code = InstalledCode(
        label=label,
        computer=computer,
        filepath_executable=executable_path,
        default_calc_job_plugin=plugin,
        description=f"{executable_name} on {computer.label}",
    )
    code.store()
    click.echo(f"  Registered code '{label}@{computer.label}' -> {executable_path}")
    return code


def setup_computers() -> None:
    """Detect and set up computers and codes for koopmans.

    This function:
    1. Creates a localhost computer if it doesn't exist (presto-style)
    2. Scans PATH for Quantum ESPRESSO executables (only for codes not yet registered)
    3. Registers found executables as AiiDA codes
    """
    computer = get_localhost_computer()

    # First, check which codes already exist (fast database lookup)
    existing_codes = []
    codes_to_find = {}
    for executable, plugin in QE_EXECUTABLES.items():
        label = executable.replace(".x", "")
        full_label = f"{label}@{computer.label}"
        if code_exists(full_label):
            existing_codes.append(executable)
        else:
            codes_to_find[executable] = plugin

    if existing_codes:
        click.echo(f"\n{len(existing_codes)} code(s) already registered, skipping:")
        for code in existing_codes:
            click.echo(f"  - {code}")

    if not codes_to_find:
        click.echo("\nAll codes already registered. Nothing to do.")
        return

    # Only search PATH for codes that don't exist yet
    click.echo(f"\nScanning for {len(codes_to_find)} missing executable(s)...")

    found_codes = []
    missing_codes = []

    for executable, plugin in codes_to_find.items():
        path = find_executable(executable)
        if path:
            version = get_executable_version(path)
            version_str = f" (v{version})" if version else ""
            click.echo(f"  Found {executable}{version_str}: {path}")
            code = setup_code(executable, path, plugin, computer)
            if code:
                found_codes.append(executable)
        else:
            missing_codes.append(executable)

    # Summary
    click.echo("\n" + "=" * 60)
    click.echo("Setup Summary")
    click.echo("=" * 60)

    if found_codes:
        click.echo(f"\nRegistered {len(found_codes)} new code(s):")
        for code in found_codes:
            click.echo(f"  - {code}")

    if missing_codes:
        click.echo(f"\nNot found on PATH ({len(missing_codes)} executable(s)):")
        for code in missing_codes:
            click.echo(f"  - {code}")

    # Check for essential codes (considering both existing and newly found)
    essential = ["pw.x"]
    all_registered = existing_codes + found_codes
    missing_essential = [e for e in essential if e not in all_registered]
    if missing_essential:
        click.echo(
            "\nWarning: Essential executable(s) not found: "
            + ", ".join(missing_essential)
        )
        click.echo("Please ensure Quantum ESPRESSO is installed and in your PATH.")
    else:
        click.echo("\nAll essential executables found. Ready to run calculations!")


def verify_installation() -> dict[str, bool]:
    """Verify the koopmans AiiDA installation.

    Returns:
        Dictionary with component names as keys and their status as values.
    """
    status = {
        "profile": False,
        "computer": False,
        "pw.x": False,
    }

    status["profile"] = profile_exists()

    if status["profile"]:
        load_koopmans_profile()
        status["computer"] = computer_exists()

        if status["computer"]:
            status["pw.x"] = code_exists(f"pw@{COMPUTER_LABEL}")

    return status


def print_status() -> None:
    """Print the current status of the koopmans AiiDA installation."""
    status = verify_installation()

    click.echo("\nBackend status")
    click.echo("=" * 40)

    for component, ok in status.items():
        icon = "✓" if ok else "✗"
        click.echo(f"  {icon} {component}")

    if all(status.values()):
        click.echo("\nAll components configured correctly!")
    else:
        click.echo("\nSome components are missing. Run 'koopmans install' to set up.")


def list_codes() -> None:
    """List all codes registered for koopmans."""
    from aiida import orm

    if not profile_exists():
        click.echo("Profile not found. Run 'koopmans install' first.")
        return

    load_koopmans_profile()

    click.echo("\nRegistered Codes")
    click.echo("=" * 60)

    query = orm.QueryBuilder()
    query.append(orm.InstalledCode, project=["label", "description"])

    codes = query.all()
    if codes:
        for label, description in codes:
            click.echo(f"  {label}: {description}")
    else:
        click.echo("  No codes registered.")
