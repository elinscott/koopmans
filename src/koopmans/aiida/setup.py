"""Automatic AiiDA setup for koopmans.

This module handles automatic configuration of AiiDA profiles, computers, and codes
so that users can run koopmans calculations without needing AiiDA knowledge.

Uses the same approach as `verdi presto` for profile and computer setup.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import click

logger = logging.getLogger(__name__)

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

# Codes that must always run in serial (no MPI).
SERIAL_CODES: set[str] = set()

def ensure_pseudo_family_installed(pseudo_family: str) -> None:
    """Ensure a pseudopotential family is installed, installing it if necessary.

    Supports PseudoDojo families with labels like:
        'PseudoDojo/0.4/LDA/SR/standard/upf'

    And SSSP families with labels like:
        'SSSP/1.3/PBEsol/efficiency'

    Args:
        pseudo_family: The label of the pseudopotential family.

    Raises:
        ValueError: If the family format is not recognized or installation fails.
    """
    from aiida.common.exceptions import NotExistent
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    # Check if already installed
    try:
        PseudoPotentialFamily.collection.get(label=pseudo_family)
        logger.debug("Pseudo family '%s' already installed", pseudo_family)
        return  # Already installed
    except NotExistent:
        pass

    logger.info("Installing pseudo family '%s'...", pseudo_family)

    # Parse the family label and install
    install_pseudo_family(pseudo_family)

    logger.info("Successfully installed pseudo family '%s'", pseudo_family)


def install_pseudo_family(pseudo_family: str) -> None:
    """Install a pseudopotential family.

    Parse the label and dispatch to the appropriate installer.
    """
    parts = pseudo_family.split("/")

    if parts[0] == "PseudoDojo" and len(parts) == 6:
        _install_pseudo_dojo_family(pseudo_family, parts)
    elif parts[0] == "SSSP" and len(parts) == 4:
        _install_sssp_family(pseudo_family, parts)
    else:
        raise ValueError(
            f"Unrecognized pseudo family format: '{pseudo_family}'. "
            "Expected 'PseudoDojo/version/functional/relativistic/protocol/format' "
            "or 'SSSP/version/functional/protocol'."
        )


def _install_pseudo_dojo_family(label: str, parts: list[str]) -> None:
    """Install a PseudoDojo pseudopotential family.

    Args:
        label: The full label for the family.
        parts: The parsed parts of the label.
    """
    import contextlib
    import io
    import warnings

    from aiida_pseudo.cli.install import download_pseudo_dojo, install_pseudo_dojo
    from aiida_pseudo.data.pseudo import JthXmlData, PsmlData, Psp8Data, UpfData
    from aiida_pseudo.groups.family import PseudoDojoConfiguration

    # Unpack the label parts
    _, version, functional, relativistic, protocol, pseudo_format = parts

    # Map format string to pseudo type class
    format_to_type = {
        "upf": UpfData,
        "psp8": Psp8Data,
        "psml": PsmlData,
        "jthxml": JthXmlData,
    }

    pseudo_type = format_to_type.get(pseudo_format.lower())
    if pseudo_type is None:
        raise ValueError(
            f"Unknown pseudo format '{pseudo_format}'. "
            f"Supported formats: {list(format_to_type.keys())}"
        )

    configuration = PseudoDojoConfiguration(
        version=version,
        functional=functional,
        relativistic=relativistic,
        protocol=protocol,
        pseudo_format=pseudo_format,
    )

    click.echo(f"  Downloading '{label}' pseudopotentials")

    with tempfile.TemporaryDirectory() as tmpdir:
        # PseudoDojo uses .tgz archives for both pseudos and metadata
        filepath_archive = Path(tmpdir) / "archive.tgz"
        filepath_metadata = Path(tmpdir) / "metadata.tgz"

        # Suppress verbose output and warnings from aiida-pseudo
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter("ignore")

            download_pseudo_dojo(
                configuration=configuration,
                filepath_archive=filepath_archive,
                filepath_metadata=filepath_metadata,
                traceback=False,
            )

            family = install_pseudo_dojo(
                configuration=configuration,
                filepath_archive=filepath_archive,
                filepath_metadata=filepath_metadata,
                pseudo_type=pseudo_type,
                label=label,
                traceback=False,
            )

        # Set the default stringency (required for get_recommended_cutoffs)
        family.set_default_stringency("normal")


def _install_sssp_family(label: str, parts: list[str]) -> None:
    """Install an SSSP pseudopotential family.

    Args:
        label: The full label for the family.
        parts: The parsed parts of the label.
    """
    import contextlib
    import io
    import warnings

    from aiida_pseudo.cli.install import download_sssp, install_sssp
    from aiida_pseudo.groups.family import SsspConfiguration

    # Unpack the label parts
    _, version, functional, protocol = parts

    configuration = SsspConfiguration(
        version=version,
        functional=functional,
        protocol=protocol,
    )

    click.echo(f"  Downloading pseudopotentials for '{label}'...")

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath_archive = Path(tmpdir) / "archive.tar.gz"
        filepath_metadata = Path(tmpdir) / "metadata.json"

        # Suppress verbose output and warnings from aiida-pseudo
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter("ignore")

            download_sssp(
                configuration=configuration,
                filepath_archive=filepath_archive,
                filepath_metadata=filepath_metadata,
                traceback=False,
            )

            install_sssp(
                filepath_archive=filepath_archive,
                filepath_metadata=filepath_metadata,
                label=label,
                traceback=False,
            )

    # Get the installed family to report the count
    from aiida_pseudo.groups.family import SsspFamily

    family = SsspFamily.collection.get(label=label)
    click.echo(f"  Successfully installed '{label}' ({family.count()} pseudopotentials)")


def profile_exists() -> bool:
    """Check if the koopmans profile already exists."""
    from aiida.manage.configuration import get_config

    config = get_config()  # type: ignore[no-untyped-call]
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

    config = get_config()  # type: ignore[no-untyped-call]

    # Storage configuration - reuse presto's detection for postgres
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


def _detect_num_cores() -> int:
    """Detect the number of available CPU cores."""
    import os

    return os.cpu_count() or 1


def get_localhost_computer(nprocs: int | None = None) -> Computer:
    """Get or create the localhost computer.

    Uses the same configuration as `verdi presto`: local transport, direct scheduler.

    Args:
        nprocs: Number of MPI processes per machine. If None, auto-detects CPU count.
    """
    from aiida import orm
    from aiida.manage.configuration import get_config

    if nprocs is None:
        nprocs = _detect_num_cores()

    if computer_exists():
        computer = orm.load_computer(COMPUTER_LABEL)
        computer.set_default_mpiprocs_per_machine(nprocs)
        click.echo(f"Computer '{COMPUTER_LABEL}' already exists (set nprocs={nprocs}).")
        return computer

    click.echo(f"Creating computer '{COMPUTER_LABEL}'...")

    config = get_config()  # type: ignore[no-untyped-call]

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
    computer.set_default_mpiprocs_per_machine(nprocs)

    click.echo(f"Created computer '{COMPUTER_LABEL}' (nprocs={nprocs}).")
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
    import os
    import re

    # Security: validate path is an absolute path to an existing executable
    # The path comes from shutil.which() via find_executable(), which only
    # returns paths to actual executables found in PATH
    if not os.path.isabs(path) or not os.path.isfile(path) or not os.access(path, os.X_OK):
        return None

    try:
        result = subprocess.run(  # noqa: S603 - path validated above
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
    force: bool = False,
) -> InstalledCode | None:
    """Set up an AiiDA code for an executable.

    Args:
        executable_name: Name of the executable (e.g., 'pw.x').
        executable_path: Absolute path to the executable.
        plugin: AiiDA plugin entry point (e.g., 'quantumespresso.pw').
        computer: The computer to associate the code with.
        force: If True, replace an existing code with the same label.

    Returns:
        The created Code object, or None if skipped.
    """
    from aiida import orm
    from aiida.orm import InstalledCode

    # Create a label from the executable name (remove .x suffix)
    label = executable_name.replace(".x", "")
    full_label = f"{label}@{computer.label}"

    if code_exists(full_label):
        if not force:
            click.echo(f"  Code '{full_label}' already exists, skipping.")
            return None
        # Delete the old code to replace it
        old_code = orm.load_code(full_label)
        old_code.base.extras.set("replaced", True)
        old_code.label = f"{label}_old"

    code = InstalledCode(
        label=label,
        computer=computer,
        filepath_executable=executable_path,
        default_calc_job_plugin=plugin,
        description=f"{executable_name} on {computer.label}",
        with_mpi=label not in SERIAL_CODES,
    )
    code.store()
    click.echo(f"  Registered code '{full_label}' -> {executable_path}")
    return code


def _get_codes_to_register(computer: Computer) -> tuple[list[str], dict[str, str]]:
    """Check which codes already exist and which need to be registered.

    Args:
        computer: The AiiDA computer to check codes on.

    Returns:
        A tuple of (existing_codes, codes_to_find) where existing_codes is a list
        of executable names already registered and codes_to_find is a dict mapping
        executable names to their plugins.
    """
    existing_codes = []
    codes_to_find = {}
    for executable, plugin in QE_EXECUTABLES.items():
        label = executable.replace(".x", "")
        full_label = f"{label}@{computer.label}"
        if code_exists(full_label):
            existing_codes.append(executable)
        else:
            codes_to_find[executable] = plugin
    return existing_codes, codes_to_find


def _scan_and_register_codes(
    codes_to_find: dict[str, str],
    computer: Computer,
    explicit_codes: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Scan PATH for executables and register them as AiiDA codes.

    Args:
        codes_to_find: Dict mapping executable names to their plugins.
        computer: The AiiDA computer to register codes on.
        explicit_codes: Dict mapping code labels (e.g. "pw") to executable paths,
            provided via --code CLI option. These take priority over PATH scanning.

    Returns:
        A tuple of (found_codes, missing_codes) listing executables found and not found.
    """
    explicit_codes = explicit_codes or {}
    # Build a lookup from label to explicit path
    explicit_by_executable = {
        f"{label}.x": path for label, path in explicit_codes.items()
    }

    found_codes = []
    missing_codes = []

    for executable, plugin in codes_to_find.items():
        path = explicit_by_executable.get(executable) or find_executable(executable)
        if path:
            version = get_executable_version(path)
            version_str = f" (v{version})" if version else ""
            source = "specified" if executable in explicit_by_executable else "found"
            click.echo(f"  {source.capitalize()} {executable}{version_str}: {path}")
            is_explicit = executable in explicit_by_executable
            setup_code(executable, path, plugin, computer, force=is_explicit)
            found_codes.append(executable)
        else:
            missing_codes.append(executable)

    return found_codes, missing_codes


def _print_setup_summary(
    existing_codes: list[str], found_codes: list[str], missing_codes: list[str]
) -> None:
    """Print a summary of the setup process.

    Args:
        existing_codes: Codes that were already registered.
        found_codes: Codes that were newly registered.
        missing_codes: Codes that were not found on PATH.
    """
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
        click.echo("\nWarning: Essential executable(s) not found: " + ", ".join(missing_essential))
        click.echo("Please ensure Quantum ESPRESSO is installed and in your PATH.")
    else:
        click.echo("\nAll essential executables found. Ready to run calculations!")


def setup_computers(
    nprocs: int | None = None,
    explicit_codes: dict[str, str] | None = None,
) -> None:
    """Detect and set up computers and codes for koopmans.

    This function:
    1. Creates a localhost computer if it doesn't exist (presto-style)
    2. Scans PATH for Quantum ESPRESSO executables (only for codes not yet registered)
    3. Registers found executables as AiiDA codes

    Args:
        nprocs: Number of MPI processes per machine. If None, auto-detects CPU count.
        explicit_codes: Dict mapping code labels (e.g. "pw") to executable paths.
    """
    computer = get_localhost_computer(nprocs=nprocs)

    existing_codes, codes_to_find = _get_codes_to_register(computer)

    # If the user explicitly specified a code that already exists, re-register it
    if explicit_codes:
        for label in explicit_codes:
            executable = f"{label}.x"
            if executable in QE_EXECUTABLES and executable not in codes_to_find:
                codes_to_find[executable] = QE_EXECUTABLES[executable]
                if executable in existing_codes:
                    existing_codes.remove(executable)

    if existing_codes:
        click.echo(f"\n{len(existing_codes)} code(s) already registered, skipping:")
        for code in existing_codes:
            click.echo(f"  - {code}")

    if not codes_to_find:
        click.echo("\nAll codes already registered. Nothing to do.")
        return

    click.echo(f"\nScanning for {len(codes_to_find)} missing executable(s)...")

    found_codes, missing_codes = _scan_and_register_codes(codes_to_find, computer, explicit_codes)

    _print_setup_summary(existing_codes, found_codes, missing_codes)


def verify_installation() -> dict[str, bool]:
    """Verify the koopmans AiiDA installation.

    Returns:
        Dictionary with component names as keys and their status as values.
    """
    status = {
        "profile": False,
        "computer": False,
        "pw.x": False,
        "daemon": False,
    }

    status["profile"] = profile_exists()

    if status["profile"]:
        load_koopmans_profile()
        status["computer"] = computer_exists()
        status["daemon"] = is_daemon_running()

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


def is_daemon_running() -> bool:
    """Check if the AiiDA daemon is running.

    Returns:
        True if the daemon is running, False otherwise.
    """
    from aiida.engine.daemon.client import get_daemon_client

    try:
        client = get_daemon_client()
        return client.is_daemon_running
    except Exception:
        return False


def start_daemon(wait: bool = True, cache: bool = True) -> bool:
    """Start the AiiDA daemon if it's not already running.

    Args:
        wait: If True, wait for the daemon to be fully started.
        cache: If True, enable AiiDA caching for calculations.

    Returns:
        True if the daemon was started or was already running, False on failure.
    """
    from aiida.engine.daemon.client import get_daemon_client
    from aiida.manage import get_config

    # Set caching configuration before starting the daemon
    config = get_config()
    config.set_option("caching.default_enabled", cache)
    config.store()

    if is_daemon_running():
        return True

    try:
        client = get_daemon_client()
        response = client.start_daemon()

        if wait:
            # Wait for daemon to be fully operational
            import time

            for _ in range(30):  # Wait up to 30 seconds
                if client.is_daemon_running:
                    return True
                time.sleep(1)
            return False

        return response is not None
    except Exception as e:
        logger.warning("Failed to start daemon: %s", e)
        return False


def stop_daemon() -> bool:
    """Stop the AiiDA daemon.

    Returns:
        True if the daemon was stopped, False on failure.
    """
    from aiida.engine.daemon.client import get_daemon_client

    if not is_daemon_running():
        return True

    try:
        client = get_daemon_client()
        client.stop_daemon(wait=True)
        return True
    except Exception as e:
        logger.warning("Failed to stop daemon: %s", e)
        return False


def ensure_daemon_running() -> None:
    """Ensure the AiiDA daemon is running, starting it if necessary.

    Raises:
        click.ClickException: If the daemon cannot be started.
    """
    if is_daemon_running():
        return

    click.echo("Starting AiiDA daemon...")
    if start_daemon(wait=True):
        click.echo("Daemon started successfully.")
    else:
        raise click.ClickException(
            "Failed to start the AiiDA daemon. "
            "This may be because RabbitMQ is not available. "
            "Please check your installation with 'koopmans backend status'."
        )


def uninstall_backend() -> None:
    """Completely remove the koopmans AiiDA backend.

    This will:
    1. Delete the AiiDA profile
    2. Remove all associated storage (database, repository)
    """
    from aiida.manage.configuration import get_config

    if not profile_exists():
        click.echo(f"Profile '{PROFILE_NAME}' does not exist. Nothing to uninstall.")
        return

    config = get_config()  # type: ignore[no-untyped-call]
    profile = config.get_profile(PROFILE_NAME)

    # Get storage path before deleting
    storage_config = profile.storage_config
    storage_path = storage_config.get("filepath") if storage_config else None

    click.echo(f"Deleting profile '{PROFILE_NAME}'...")

    # Delete the profile (this also handles storage cleanup for sqlite_dos)
    config.delete_profile(PROFILE_NAME, delete_storage=True)
    config.store()

    click.echo(f"  Profile '{PROFILE_NAME}' deleted.")

    # Clean up any remaining storage directory
    if storage_path:
        storage_dir = Path(storage_path)
        if storage_dir.exists():
            shutil.rmtree(storage_dir)
            click.echo(f"  Removed storage directory: {storage_dir}")

    # Also clean up the profile directory if it exists
    profile_dir = Path(config.dirpath) / PROFILE_NAME
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
        click.echo(f"  Removed profile directory: {profile_dir}")

    # Clean up scratch directory
    scratch_dir = Path(config.dirpath) / "scratch" / PROFILE_NAME
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
        click.echo(f"  Removed scratch directory: {scratch_dir}")

    click.echo("\nUninstall complete. The AiiDA backend has been removed.")
