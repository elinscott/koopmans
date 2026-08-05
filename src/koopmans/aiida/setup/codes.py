"""Code (AiiDA executable) registration helpers.

Scans PATH for Quantum ESPRESSO executables, registers each one against
the localhost Computer with the appropriate plugin entry point.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

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


def code_specs() -> dict[str, tuple[str, str | None]]:
    """Return every code koopmans registers, as ``{label: (executable, plugin)}``."""
    return {
        executable.replace(".x", ""): (executable, plugin)
        for executable, plugin in QE_EXECUTABLES.items()
    }


# Codes that must not use MPI whatever their build supports, mapped to the
# reason quoted in the install summary. The CalcJobs also enforce single-rank
# resources; this additionally stops mpirun from being prepended at all.
SERIAL_CODES: dict[str, str] = {
    "wann2kcp": "always serial: races on its buffer scratch",
    "merge_evc": "always serial: a plain concatenation tool",
}

# Dynamic symbols only an MPI program carries: a serial build never calls
# MPI_Init. Fortran bindings lower-case and suffix the name.
MPI_INIT_SYMBOLS = ("MPI_Init", "mpi_init_", "MPI_Init_thread")

# Soname prefix of the MPI runtimes: OpenMPI's libmpi / libmpi_mpifh, MPICH's
# libmpi / libmpich, Intel MPI's libmpi / libmpifort. Matched against the
# soname alone, never the resolved path — a ScaLAPACK or OpenBLAS build
# installed under an ``openmpi/`` directory spells "mpi" without being one.
MPI_LIBRARY_PREFIXES = ("libmpi",)


class MpiDecision(NamedTuple):
    """Whether one code runs under mpirun, and the evidence behind it."""

    label: str
    with_mpi: bool
    reason: str


class MpiMigration(NamedTuple):
    """A registered code whose stored ``with_mpi`` disagreed with its binary."""

    label: str
    retired_label: str
    decision: MpiDecision


def mpi_evidence(dynamic_symbols: str, linked_libraries: str, raw_strings: str) -> str | None:
    """Return why a binary looks MPI-capable, or ``None`` if nothing says so.

    Evidence, strongest first: an ``MPI_Init`` entry among the dynamic symbols
    (``nm -D``); an MPI runtime among the linked sonames (``ldd``); the
    ``MPI_Init`` string in the binary, which catches a statically linked MPI
    whose symbols were stripped. The bare substring "mpi" is not evidence —
    OpenBLAS, ScaLAPACK and HDF5 paths carry it and none of them make a
    program parallel.
    """
    for symbol in MPI_INIT_SYMBOLS:
        if symbol in dynamic_symbols:
            return f"declares {symbol}"

    for soname in _linked_sonames(linked_libraries):
        if soname.startswith(MPI_LIBRARY_PREFIXES):
            return f"links {soname}"

    if "MPI_Init" in raw_strings:
        return "contains the MPI_Init string"

    return None


def declares_mpi(dynamic_symbols: str, linked_libraries: str, raw_strings: str) -> bool:
    """Report whether the collected evidence shows the binary is MPI-capable."""
    return mpi_evidence(dynamic_symbols, linked_libraries, raw_strings) is not None


def _linked_sonames(linked_libraries: str) -> list[str]:
    """Extract the library names from ``ldd`` output, dropping resolved paths."""
    sonames = []
    for line in linked_libraries.splitlines():
        entry = line.split("=>")[0].split("(")[0].strip()
        if entry:
            sonames.append(Path(entry).name)
    return sonames


def _run_probe(command: list[str], executable_path: str) -> str:
    """Run an inspection command over a binary, returning "" if it cannot run."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed command, path validated by caller
            [*command, executable_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as exc:
        logger.debug("MPI probe %s failed on %s: %s", command[0], executable_path, exc)
        return ""
    return result.stdout


def collect_mpi_evidence(executable_path: str) -> tuple[str, str, str]:
    """Inspect a binary, returning its dynamic symbols, linked libraries and strings.

    Every probe that cannot run — missing tool, unreadable binary, timeout —
    contributes an empty string rather than raising, so an undecidable binary
    ends up serial.
    """
    if not os.path.isfile(executable_path):
        return "", "", ""

    symbols = _run_probe(["nm", "-D"], executable_path)
    libraries = _run_probe(["ldd"], executable_path)
    if declares_mpi(symbols, libraries, ""):
        # The cheap probes already decided; skip reading the whole binary.
        return symbols, libraries, ""
    return symbols, libraries, _run_probe(["strings", "-a"], executable_path)


def decide_with_mpi(
    label: str,
    executable_path: str,
    serial_labels: Iterable[str] = (),
    parallel_labels: Iterable[str] = (),
) -> MpiDecision:
    """Decide whether a code is registered to run under mpirun.

    The default is set by an asymmetry: running an MPI-capable binary in
    serial is correct and merely slower, while running a serial binary under
    mpirun starts N independent copies that race on one working directory and
    corrupt each other's output. So evidence only ever promotes a code to MPI,
    and anything undecidable stays serial.
    """
    if label in serial_labels:
        return MpiDecision(label, False, "requested by --serial")
    if label in parallel_labels:
        return MpiDecision(label, True, "requested by --parallel")
    if label in SERIAL_CODES:
        return MpiDecision(label, False, SERIAL_CODES[label])

    evidence = mpi_evidence(*collect_mpi_evidence(executable_path))
    if evidence is not None:
        return MpiDecision(label, True, evidence)
    return MpiDecision(label, False, "no MPI symbols, libraries or strings in the binary")


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


def retire_code(label: str, computer: Computer, suffix: str) -> str:
    """Relabel the code registered under ``label`` as ``<label><suffix>``.

    A repeat of the same operation would otherwise collide with the code it
    retired last time, so the suffix gains a counter until it is free.
    """
    from aiida import orm

    old_code = orm.load_code(f"{label}@{computer.label}")
    old_code.base.extras.set("replaced", True)

    retired_label = f"{label}{suffix}"
    counter = 1
    while code_exists(f"{retired_label}@{computer.label}"):
        counter += 1
        retired_label = f"{label}{suffix}{counter}"
    old_code.label = retired_label
    return retired_label


def setup_code(
    executable_name: str,
    executable_path: str,
    plugin: str | None,
    computer: Computer,
    force: bool = False,
    label: str | None = None,
    with_mpi: bool | None = None,
) -> InstalledCode | None:
    """Set up an AiiDA code for an executable.

    ``label`` defaults to the executable stem. ``with_mpi`` defaults to what
    inspecting the binary decides (see :func:`decide_with_mpi`).
    """
    from aiida.orm import InstalledCode

    label = label or executable_name.replace(".x", "")
    full_label = f"{label}@{computer.label}"

    if code_exists(full_label):
        if not force:
            click.echo(f"  Code '{full_label}' already exists, skipping.")
            return None
        retire_code(label, computer, "_old")

    if with_mpi is None:
        with_mpi = decide_with_mpi(label, executable_path).with_mpi

    code = InstalledCode(
        label=label,
        computer=computer,
        filepath_executable=executable_path,
        default_calc_job_plugin=plugin,
        description=f"{executable_name} on {computer.label}",
        with_mpi=with_mpi,
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
    serial_labels: Iterable[str] = (),
    parallel_labels: Iterable[str] = (),
) -> tuple[list[str], list[str], list[MpiDecision]]:
    """Scan PATH for executables and register them as AiiDA codes.

    ``codes_to_find`` and ``explicit_codes`` are both keyed by code label,
    so a label whose binary differs from the one PATH resolves (a
    decompose-capable pw2wannier90.x, say) can be pointed at explicitly.
    """
    explicit_codes = explicit_codes or {}

    found_codes = []
    missing_codes = []
    decisions = []

    for label, (executable, plugin) in codes_to_find.items():
        path = explicit_codes.get(label) or find_executable(executable)
        if path:
            version = get_executable_version(path)
            version_str = f" (v{version})" if version else ""
            is_explicit = label in explicit_codes
            source = "Specified" if is_explicit else "Found"
            click.echo(f"  {source} {executable}{version_str}: {path}")
            decision = decide_with_mpi(label, path, serial_labels, parallel_labels)
            setup_code(
                executable,
                path,
                plugin,
                computer,
                force=is_explicit,
                label=label,
                with_mpi=decision.with_mpi,
            )
            found_codes.append(label)
            decisions.append(decision)
        else:
            missing_codes.append(label)

    return found_codes, missing_codes, decisions


def migrate_code_mpi_flags(
    labels: Iterable[str],
    computer: Computer,
    serial_labels: Iterable[str] = (),
    parallel_labels: Iterable[str] = (),
) -> list[MpiMigration]:
    """Re-register any already-installed code whose stored ``with_mpi`` is wrong.

    Code nodes are immutable, so a code registered before its binary was
    inspected keeps whatever flag it was given. Each mismatching code is
    relabelled ``<label>_mpi_pre`` and a replacement stored under the original
    label; a code whose binary has since disappeared is left alone.
    """
    from aiida import orm

    specs = code_specs()
    migrations = []

    for label in labels:
        code = orm.load_code(f"{label}@{computer.label}")
        executable_path = str(code.filepath_executable)
        if not os.path.isfile(executable_path):
            continue

        decision = decide_with_mpi(label, executable_path, serial_labels, parallel_labels)
        if code.with_mpi == decision.with_mpi:
            continue

        executable, plugin = specs[label]
        retired_label = retire_code(label, computer, "_mpi_pre")
        setup_code(
            executable,
            executable_path,
            plugin,
            computer,
            label=label,
            with_mpi=decision.with_mpi,
        )
        migrations.append(MpiMigration(label, retired_label, decision))

    return migrations


def print_mpi_decisions(decisions: Iterable[MpiDecision]) -> None:
    """Print how each registered code was decided to run, and why."""
    decisions = list(decisions)
    if not decisions:
        return

    width = max(len(decision.label) for decision in decisions)
    click.echo("\nMPI:")
    for decision in decisions:
        mode = "parallel" if decision.with_mpi else "serial"
        click.echo(f"  {decision.label:<{width}}  {mode:<8}  ({decision.reason})")


def print_mpi_migrations(migrations: Iterable[MpiMigration]) -> None:
    """Print which codes were re-registered because their stored flag was wrong."""
    migrations = list(migrations)
    if not migrations:
        return

    click.echo(f"\nCorrected the MPI setting of {len(migrations)} registered code(s):")
    for migration in migrations:
        mode = "parallel" if migration.decision.with_mpi else "serial"
        click.echo(
            f"  {migration.label}: now {mode} ({migration.decision.reason}); "
            f"the previous code node is kept as '{migration.retired_label}'"
        )
    click.echo(
        "  A replacement code node hashes differently from the one it replaces, "
        "so results calculated with the previous node will not be reused from the cache."
    )


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

    essential = ["pw"]
    all_registered = existing_codes + found_codes
    missing_essential = [e for e in essential if e not in all_registered]
    if missing_essential:
        click.echo("\nWarning: Essential executable(s) not found: " + ", ".join(missing_essential))
        click.echo("Please ensure Quantum ESPRESSO is installed and in your PATH.")
    else:
        click.echo("\nAll essential executables found. Ready to run calculations!")
