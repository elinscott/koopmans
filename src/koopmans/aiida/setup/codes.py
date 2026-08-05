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

# Quantum ESPRESSO and related executables that koopmans needs, mapped to the
# code's default CalcJob entry point. kcw.x maps to None: the one binary backs
# three CalcJobs (koopmans.kcw_wann2kc / kcw_screen / kcw_ham, selected via
# control.calculation), so no single default is honest — the workgraph tasks
# name their process class explicitly.

QE_EXECUTABLES: dict[str, str | None] = {
    "pw.x": "quantumespresso.pw",
    "ph.x": "quantumespresso.ph",
    "pp.x": "quantumespresso.pp",
    "projwfc.x": "quantumespresso.projwfc",
    "dos.x": "quantumespresso.dos",
    "wannier90.x": "wannier90.wannier90",
    "pw2wannier90.x": "quantumespresso.pw2wannier90",
    "wann2kcp.x": "koopmans.wann2kcp",
    "merge_evc.x": "koopmans.merge_evc",
    "kcw.x": None,
    "kcp.x": "koopmans.kcp",
}


# Codes whose executable name collides with a stock one because they are a
# different build of it. PATH cannot tell the two apart, so these are
# registered only when ``koopmans install --code <label>=<path>`` names the
# binary. ``wan_mode='decompose'`` — the pw2wannier90.x mode that builds the
# power-spectrum ML descriptors — exists only in a patched build, and the
# stock binary rejects its namelist keys at run time.
VARIANT_CODES: dict[str, tuple[str, str | None]] = {
    "pw2wannier90_decompose": ("pw2wannier90.x", "koopmans.pw2wannier_decompose"),
}


def code_specs() -> dict[str, tuple[str, str | None]]:
    """Return every code koopmans registers, as ``{label: (executable, plugin)}``."""
    specs = {
        executable.replace(".x", ""): (executable, plugin)
        for executable, plugin in QE_EXECUTABLES.items()
    }
    specs.update(VARIANT_CODES)
    return specs


# Codes that must always run in serial (no MPI): wann2kcp.x races on its
# buffer scratch under multiple ranks, and merge_evc.x is a plain
# concatenation tool. The CalcJobs also enforce single-rank resources; this
# additionally stops mpirun from being prepended at all.
SERIAL_CODES: set[str] = {"wann2kcp", "merge_evc"}


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
    plugin: str | None,
    computer: Computer,
    force: bool = False,
    label: str | None = None,
) -> InstalledCode | None:
    """Set up an AiiDA code for an executable.

    ``label`` defaults to the executable stem.
    """
    from aiida import orm
    from aiida.orm import InstalledCode

    label = label or executable_name.replace(".x", "")
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


def get_codes_to_register(
    computer: Computer,
) -> tuple[list[str], dict[str, tuple[str, str | None]]]:
    """Return ``(existing_labels, codes_to_find)``, both keyed by code label."""
    existing_codes = []
    codes_to_find = {}
    for label, spec in code_specs().items():
        if code_exists(f"{label}@{computer.label}"):
            existing_codes.append(label)
        else:
            codes_to_find[label] = spec
    return existing_codes, codes_to_find


def scan_and_register_codes(
    codes_to_find: dict[str, tuple[str, str | None]],
    computer: Computer,
    explicit_codes: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Scan PATH for executables and register them as AiiDA codes.

    ``codes_to_find`` and ``explicit_codes`` are both keyed by code label,
    so a label whose binary differs from the one PATH resolves (a
    decompose-capable pw2wannier90.x, say) can be pointed at explicitly.
    A label in :data:`VARIANT_CODES` is registered only from
    ``explicit_codes``: PATH resolves its executable name to the stock
    build, which is not the binary that label promises.
    """
    explicit_codes = explicit_codes or {}

    found_codes = []
    missing_codes = []

    for label, (executable, plugin) in codes_to_find.items():
        if label in VARIANT_CODES:
            path = explicit_codes.get(label)
        else:
            path = explicit_codes.get(label) or find_executable(executable)
        if path:
            version = get_executable_version(path)
            version_str = f" (v{version})" if version else ""
            is_explicit = label in explicit_codes
            source = "Specified" if is_explicit else "Found"
            click.echo(f"  {source} {executable}{version_str}: {path}")
            setup_code(executable, path, plugin, computer, force=is_explicit, label=label)
            found_codes.append(label)
        else:
            missing_codes.append(label)

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

    unspecified = [code for code in missing_codes if code in VARIANT_CODES]
    not_on_path = [code for code in missing_codes if code not in VARIANT_CODES]

    if not_on_path:
        click.echo(f"\nNot found on PATH ({len(not_on_path)} executable(s)):")
        for code in not_on_path:
            click.echo(f"  - {code}")

    for code in unspecified:
        executable = VARIANT_CODES[code][0]
        click.echo(
            f"\nNot registered: {code}. This is a non-standard build of "
            f"{executable} that PATH cannot single out, so give its path:\n"
            f"  koopmans install --code {code}=/path/to/{executable}"
        )

    essential = ["pw"]
    all_registered = existing_codes + found_codes
    missing_essential = [e for e in essential if e not in all_registered]
    if missing_essential:
        click.echo("\nWarning: Essential executable(s) not found: " + ", ".join(missing_essential))
        click.echo("Please ensure Quantum ESPRESSO is installed and in your PATH.")
    else:
        click.echo("\nAll essential executables found. Ready to run calculations!")
