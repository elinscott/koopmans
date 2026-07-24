"""High-level install / uninstall / status orchestration."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import click

from .codes import (
    code_exists,
    get_codes_to_register,
    print_setup_summary,
    scan_and_register_codes,
)
from .computer import (
    COMPUTER_LABEL,
    computer_exists,
    computer_has_thread_pin,
    get_localhost_computer,
)
from .daemon import is_daemon_running, stop_daemon
from .hq import (
    is_hq_server_running,
    is_hq_worker_running,
    stop_hq,
)
from .profile import PROFILE_NAME, load_koopmans_profile, profile_exists

logger = logging.getLogger(__name__)


def setup_computers(
    nprocs: int | None = None,
    explicit_codes: dict[str, str] | None = None,
) -> None:
    """Detect and set up computers and codes for koopmans.

    1. Creates a localhost computer if it doesn't exist.
    2. Scans PATH for Quantum ESPRESSO executables.
    3. Registers found executables as AiiDA codes.
    """
    from .codes import QE_EXECUTABLES

    computer = get_localhost_computer(nprocs=nprocs)

    existing_codes, codes_to_find = get_codes_to_register(computer)

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

    found_codes, missing_codes = scan_and_register_codes(codes_to_find, computer, explicit_codes)

    print_setup_summary(existing_codes, found_codes, missing_codes)


def verify_installation() -> dict[str, bool]:
    """Return a status dict for ``koopmans backend status``."""
    status: dict[str, bool] = {
        "profile": False,
        "computer": False,
        "computer.thread_pin": False,
        "pw.x": False,
        "daemon": False,
        "hq.server": False,
        "hq.worker": False,
    }

    status["profile"] = profile_exists()

    if status["profile"]:
        from aiida import orm

        load_koopmans_profile()
        status["computer"] = computer_exists()
        status["daemon"] = is_daemon_running()
        status["hq.server"] = is_hq_server_running()
        status["hq.worker"] = is_hq_worker_running()

        if status["computer"]:
            status["pw.x"] = code_exists(f"pw@{COMPUTER_LABEL}")
            status["computer.thread_pin"] = computer_has_thread_pin(
                orm.load_computer(COMPUTER_LABEL)
            )

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


def uninstall_backend() -> None:
    """Completely remove the koopmans AiiDA backend.

    1. Stops the HQ worker + server.
    2. Stops the AiiDA daemon.
    3. Deletes the AiiDA profile + storage.
    4. Removes the koopmans-managed dir (HQ binary, pid files).
    """
    from aiida.manage.configuration import get_config

    # Stop HQ first so the AiiDA daemon shutdown doesn't race against
    # any in-flight CalcJobs.
    if is_hq_server_running() or is_hq_worker_running():
        click.echo("Stopping HyperQueue server + worker...")
        stop_hq()

    if profile_exists():
        load_koopmans_profile()
        if is_daemon_running():
            click.echo("Stopping AiiDA daemon...")
            stop_daemon()

    if not profile_exists():
        click.echo(f"Profile '{PROFILE_NAME}' does not exist. Nothing to uninstall.")
        _purge_koopmans_dir()
        return

    config = get_config()
    profile = config.get_profile(PROFILE_NAME)

    storage_config = profile.storage_config
    storage_path = storage_config.get("filepath") if storage_config else None

    click.echo(f"Deleting profile '{PROFILE_NAME}'...")

    config.delete_profile(PROFILE_NAME, delete_storage=True)
    config.store()  # type: ignore[no-untyped-call]

    click.echo(f"  Profile '{PROFILE_NAME}' deleted.")

    if storage_path:
        storage_dir = Path(storage_path)
        if storage_dir.exists():
            shutil.rmtree(storage_dir)
            click.echo(f"  Removed storage directory: {storage_dir}")

    profile_dir = Path(config.dirpath) / PROFILE_NAME
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
        click.echo(f"  Removed profile directory: {profile_dir}")

    scratch_dir = Path(config.dirpath) / "scratch" / PROFILE_NAME
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
        click.echo(f"  Removed scratch directory: {scratch_dir}")

    _purge_koopmans_dir()

    click.echo("\nUninstall complete. The AiiDA backend has been removed.")


def _purge_koopmans_dir() -> None:
    """Remove the koopmans-managed dir (HQ binary, pid + log files)."""
    from aiida.manage.configuration import get_config

    config = get_config()
    koopmans_dir_path = Path(config.dirpath) / "koopmans"
    if koopmans_dir_path.exists():
        shutil.rmtree(koopmans_dir_path)
        click.echo(f"  Removed koopmans backend directory: {koopmans_dir_path}")
