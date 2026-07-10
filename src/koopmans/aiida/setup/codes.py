"""Code (AiiDA executable) registration helpers.

Scans PATH for Quantum ESPRESSO executables, registers each one against
the localhost Computer with the appropriate plugin entry point.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import click

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aiida.orm import Computer, InstalledCode

# Quantum ESPRESSO and related executables that koopmans needs
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
    "kcp.x": "koopmans.kcp",
}

# Codes that must always run in serial (no MPI).
SERIAL_CODES: set[str] = set()


def find_executable(name: str) -> str | None:
    """Find an executable on the system PATH."""
    path = shutil.which(name)
    if path:
        return str(Path(path).resolve())
    return None


def get_executable_version(path: str) -> str | None:
    """Try to get the version of a Quantum ESPRESSO executable."""
    import os
    import re

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
    """Set up an AiiDA code for an executable."""
    from aiida import orm
    from aiida.orm import InstalledCode

    label = executable_name.replace(".x", "")
    full_label = f"{label}@{computer.label}"

    if code_exists(full_label):
        if not force:
            click.echo(f"  Code '{full_label}' already exists, skipping.")
            return None
        old_code = orm.load_code(full_label)
        old_code.base.extras.set("replaced", True)
        # Uniquify the retired label: a second forced reinstall would otherwise
        # collide with the previous <label>_old on the same computer.
        retired_label = f"{label}_old"
        suffix = 1
        while code_exists(f"{retired_label}@{computer.label}"):
            suffix += 1
            retired_label = f"{label}_old{suffix}"
        old_code.label = retired_label

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


def get_codes_to_register(computer: Computer) -> tuple[list[str], dict[str, str]]:
    """Return ``(existing_codes, codes_to_find)``."""
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


def scan_and_register_codes(
    codes_to_find: dict[str, str],
    computer: Computer,
    explicit_codes: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Scan PATH for executables and register them as AiiDA codes."""
    explicit_codes = explicit_codes or {}
    explicit_by_executable = {f"{label}.x": path for label, path in explicit_codes.items()}

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


def list_codes() -> None:
    """List all codes registered for koopmans."""
    from aiida import orm

    from .profile import load_koopmans_profile, profile_exists

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


def print_setup_summary(
    existing_codes: list[str], found_codes: list[str], missing_codes: list[str]
) -> None:
    """Print a summary of the setup process."""
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

    essential = ["pw.x"]
    all_registered = existing_codes + found_codes
    missing_essential = [e for e in essential if e not in all_registered]
    if missing_essential:
        click.echo("\nWarning: Essential executable(s) not found: " + ", ".join(missing_essential))
        click.echo("Please ensure Quantum ESPRESSO is installed and in your PATH.")
    else:
        click.echo("\nAll essential executables found. Ready to run calculations!")
