"""Code (AiiDA executable) registration helpers.

Scans PATH for Quantum ESPRESSO executables, registers each one against
the localhost Computer with the appropriate plugin entry point.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Container, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import click

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aiida.orm import AbstractCode, Computer, InstalledCode

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
# reason quoted in the install summary. Each entry names a property of the
# program, not of the build, so ``--parallel`` on one of these labels is
# rejected rather than honoured. The CalcJobs also enforce single-rank
# resources; this additionally stops mpirun from being prepended at all.
SERIAL_CODES: dict[str, str] = {
    "wann2kcp": "always serial: races on its buffer scratch",
    "merge_evc": "always serial: a plain concatenation tool",
}

# The MPI initialization entry points, after lower-casing, dropping the
# Fortran trailing underscores and dropping the ``PMPI_`` profiling prefix.
# ``MPI_Initialized`` is deliberately absent: a program may ask whether MPI
# is running without ever starting it.
MPI_INIT_NAMES = frozenset({"mpi_init", "mpi_init_thread"})

# Soname prefixes of the MPI runtime itself: OpenMPI's libmpi / libmpi_mpifh /
# libopen-pal / libopen-rte, MPICH's libmpi / libmpich, Intel MPI's libmpi /
# libmpifort. A runtime library calls MPI_Init by construction — OpenMPI's
# Fortran bindings carry three undefined PMPI_Init entries — so it is excluded
# from the search rather than counted as evidence. Matched against the soname
# alone, never the resolved path: a ScaLAPACK or OpenBLAS build installed
# under an ``openmpi/`` directory spells "mpi" without being one.
MPI_RUNTIME_PREFIXES = ("libmpi", "libpmpi", "libmpich", "libopen-pal", "libopen-rte")


class BinaryProbe(NamedTuple):
    """What the inspection tools reported about one executable.

    ``library_symbols`` maps the soname of each shared library the executable
    links to that library's own ``nm -D`` output.
    """

    dynamic_symbols: str
    library_symbols: dict[str, str]
    raw_strings: str = ""


class MpiDecision(NamedTuple):
    """Whether one code runs under mpirun, and the evidence behind it."""

    label: str
    with_mpi: bool
    reason: str


class MpiMigration(NamedTuple):
    """A registered code whose effective ``with_mpi`` disagreed with its binary."""

    label: str
    retired_label: str
    decision: MpiDecision


def is_mpi_init(symbol: str) -> bool:
    """Report whether a symbol name is one of MPI's initialization entry points.

    Accepts the Fortran spellings (``mpi_init_``, ``mpi_init__``), the
    ``PMPI_`` profiling aliases and glibc version suffixes. ``MPI_Initialized``
    is not a match, even though ``MPI_Init`` is a substring of it.
    """
    name = symbol.split("@", 1)[0].lower().rstrip("_")
    if name.startswith("pmpi_"):
        name = name[1:]
    return name in MPI_INIT_NAMES


def is_mpi_runtime(soname: str) -> bool:
    """Report whether a soname belongs to the MPI runtime rather than to a caller."""
    return soname.startswith(MPI_RUNTIME_PREFIXES)


def undefined_symbols(dynamic_symbols: str) -> Iterator[str]:
    """Yield the undefined symbol names in ``nm -D`` output.

    An undefined entry carries no address, so its line holds two fields: the
    type letter (``U``, or ``w``/``v`` for a weak reference) and the name.
    """
    for line in dynamic_symbols.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in {"U", "w", "v"}:
            yield fields[1]


def calls_mpi_init(dynamic_symbols: str) -> str | None:
    """Return an undefined MPI initialization symbol in ``nm -D`` output, if any."""
    for symbol in undefined_symbols(dynamic_symbols):
        if is_mpi_init(symbol):
            return symbol
    return None


def mpi_evidence(probe: BinaryProbe) -> str | None:
    """Return why a binary looks MPI-capable, or ``None`` if nothing says so.

    Evidence, strongest first:

    1. an undefined MPI_Init entry in the executable's own dynamic symbols,
       meaning the executable calls MPI itself;
    2. an undefined MPI_Init entry in one of the shared libraries it links,
       excluding the MPI runtime's own libraries — the GNU cmake build of
       Quantum ESPRESSO puts every MPI call in ``libqe_modules``, so the
       executable's own symbols miss it;
    3. a line of ``strings`` output that is exactly an MPI_Init symbol name,
       which catches a statically linked MPI whose symbols were stripped.

    Linking the MPI runtime is not evidence. Anything compiled with ``mpif90``
    or ``mpiicc`` records ``libmpi`` in its ``DT_NEEDED`` list whether or not a
    single MPI call survives compilation, and ``ldd`` reports the whole
    transitive closure, so a serial program linking parallel HDF5 or ScaLAPACK
    inherits the same evidence.

    Requiring an undefined MPI_Init outside the runtime is narrower, not
    sound: a serial program linked against a parallel build of HDF5 inherits
    HDF5's own ``U MPI_Init`` and is still read as parallel. Register such a
    code with ``--serial``.
    """
    symbol = calls_mpi_init(probe.dynamic_symbols)
    if symbol is not None:
        return f"calls {symbol}"

    for soname, symbols in probe.library_symbols.items():
        if is_mpi_runtime(soname):
            continue
        symbol = calls_mpi_init(symbols)
        if symbol is not None:
            return f"links {soname}, which calls {symbol}"

    for line in probe.raw_strings.splitlines():
        if is_mpi_init(line.strip()):
            return "contains an MPI_Init symbol name"

    return None


def declares_mpi(probe: BinaryProbe) -> bool:
    """Report whether the collected evidence shows the binary is MPI-capable."""
    return mpi_evidence(probe) is not None


def linked_libraries(ldd_output: str) -> list[tuple[str, str]]:
    """Return ``(soname, resolved path)`` for each resolvable entry of ``ldd`` output.

    Entries the loader could not resolve, and the virtual ``linux-vdso``, have
    no file to inspect and are dropped.
    """
    resolved = []
    for line in ldd_output.splitlines():
        left, _, right = line.partition("=>")
        if right:
            path = right.split(" (")[0].strip()
            soname = Path(left.strip()).name
        else:
            path = left.split(" (")[0].strip()
            soname = Path(path).name
        if path.startswith("/") and soname:
            resolved.append((soname, path))
    return resolved


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


def _closure_symbols(executable_path: str) -> dict[str, str]:
    """Return ``nm -D`` output for each non-runtime library the binary links.

    The walk stops at the first library that calls MPI_Init: one hit already
    decides, and a Quantum ESPRESSO binary links forty libraries.
    """
    symbols = {}
    for soname, path in linked_libraries(_run_probe(["ldd"], executable_path)):
        if is_mpi_runtime(soname):
            continue
        symbols[soname] = _run_probe(["nm", "-D"], path)
        if calls_mpi_init(symbols[soname]) is not None:
            break
    return symbols


def collect_mpi_evidence(executable_path: str) -> BinaryProbe:
    """Inspect a binary and its linked libraries for MPI initialization calls.

    Every probe that cannot run — missing tool, unreadable binary, timeout —
    contributes an empty string rather than raising, so an undecidable binary
    ends up serial. A path that is not a regular file is not probed at all.
    """
    if not os.path.isfile(executable_path):
        return BinaryProbe("", {})

    symbols = _run_probe(["nm", "-D"], executable_path)
    if calls_mpi_init(symbols) is not None:
        # The strongest evidence is already in; the closure need not be walked.
        return BinaryProbe(symbols, {})

    probe = BinaryProbe(symbols, _closure_symbols(executable_path))
    if declares_mpi(probe):
        # The cheap probes already decided; skip reading the whole binary.
        return probe
    return probe._replace(raw_strings=_run_probe(["strings", "-a"], executable_path))


def parallel_override_error(label: str) -> str:
    """Return the message for ``--parallel`` on a code that is always serial."""
    return (
        f"Cannot register {label} to run under mpirun: {label} is "
        f"{SERIAL_CODES[label]}. That is a property of the program, not of the "
        f"build, so there is nothing for --parallel to overrule; drop {label} "
        "from --parallel."
    )


def decide_with_mpi(
    label: str,
    executable_path: str,
    serial_labels: Container[str] = frozenset(),
    parallel_labels: Container[str] = frozenset(),
) -> MpiDecision:
    """Decide whether a code is registered to run under mpirun.

    Precedence: ``SERIAL_CODES`` first, then ``--serial``, then ``--parallel``,
    then the binary. ``--parallel`` on a ``SERIAL_CODES`` label raises
    ``ValueError`` — the entry describes the program, so no build overrules it.

    The default is set by an asymmetry: running an MPI-capable binary in
    serial is correct and merely slower, while running a serial binary under
    mpirun starts N independent copies that race on one working directory and
    corrupt each other's output. So evidence only ever promotes a code to MPI,
    and anything undecidable stays serial.
    """
    if label in SERIAL_CODES:
        if label in parallel_labels:
            raise ValueError(parallel_override_error(label))
        return MpiDecision(label, False, SERIAL_CODES[label])
    if label in serial_labels:
        return MpiDecision(label, False, "requested by --serial")
    if label in parallel_labels:
        return MpiDecision(label, True, "requested by --parallel")

    evidence = mpi_evidence(collect_mpi_evidence(executable_path))
    if evidence is not None:
        return MpiDecision(label, True, evidence)
    return MpiDecision(label, False, "no MPI_Init call in the binary or the libraries it links")


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


def retire_code(code: AbstractCode, computer: Computer, suffix: str) -> str:
    """Relabel ``code`` as ``<label><suffix>`` and mark it replaced.

    A repeat of the same operation would otherwise collide with the code it
    retired last time, so the suffix gains a counter until it is free.
    """
    code.base.extras.set("replaced", True)

    retired_label = f"{code.label}{suffix}"
    counter = 1
    while code_exists(f"{retired_label}@{computer.label}"):
        counter += 1
        retired_label = f"{code.label}{suffix}{counter}"
    code.label = retired_label
    return retired_label


def plugin_default_with_mpi(entry_point: str) -> bool:
    """Return a CalcJob plugin's ``metadata.options.withmpi`` default.

    Mirrors what aiida-core falls back to when neither the code nor the
    submission sets ``withmpi``: the default declared in the process spec, or
    ``False`` when the plugin declares none.
    """
    from aiida.plugins import CalculationFactory

    process_class: Any = CalculationFactory(entry_point)
    try:
        return bool(process_class.spec().inputs["metadata"]["options"]["withmpi"].default)
    except (KeyError, RuntimeError):
        # ``plumpy.InputPort.default`` raises ``RuntimeError`` when unset.
        return False


def effective_with_mpi(code: AbstractCode) -> bool | None:
    """Return how a registered code actually runs, or ``None`` if that is undecidable.

    A code node's ``with_mpi`` may be unset, in which case aiida-core takes
    the ``metadata.options.withmpi`` default of the CalcJob being submitted —
    so an unset flag is not the same as serial. A code with no default plugin
    (kcw.x backs three CalcJobs) can be submitted through more than one
    process class, so its behaviour cannot be read off the node.
    """
    from aiida.common.exceptions import EntryPointError

    if code.with_mpi is not None:
        return bool(code.with_mpi)
    plugin = code.default_calc_job_plugin
    if plugin is None:
        return None
    try:
        return plugin_default_with_mpi(plugin)
    except EntryPointError:
        return None


def _store_code(
    executable_name: str,
    executable_path: str,
    plugin: str | None,
    computer: Computer,
    label: str,
    with_mpi: bool,
) -> InstalledCode:
    """Build and store one ``InstalledCode``."""
    from aiida.orm import InstalledCode

    code = InstalledCode(
        label=label,
        computer=computer,
        filepath_executable=executable_path,
        default_calc_job_plugin=plugin,
        description=f"{executable_name} on {computer.label}",
        with_mpi=with_mpi,
    )
    code.store()
    click.echo(f"  Registered code '{label}@{computer.label}' -> {executable_path}")
    return code


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
    inspecting the binary decides (see :func:`decide_with_mpi`). When
    ``force`` replaces an existing code, the replacement is stored before the
    old node is relabelled, so a store that fails leaves the label resolving
    to the code that was already there.
    """
    from aiida import orm

    label = label or executable_name.replace(".x", "")
    full_label = f"{label}@{computer.label}"

    superseded = None
    if code_exists(full_label):
        if not force:
            click.echo(f"  Code '{full_label}' already exists, skipping.")
            return None
        superseded = orm.load_code(full_label)

    if with_mpi is None:
        with_mpi = decide_with_mpi(label, executable_path).with_mpi

    code = _store_code(executable_name, executable_path, plugin, computer, label, with_mpi)
    if superseded is not None:
        retire_code(superseded, computer, "_old")
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

    ``serial_labels`` and ``parallel_labels`` are tested once per code, so
    they are materialized before the loop and may be given as generators.
    """
    explicit_codes = explicit_codes or {}
    serial_labels = frozenset(serial_labels)
    parallel_labels = frozenset(parallel_labels)

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
    """Re-register any already-installed code that runs the wrong way.

    Code nodes are immutable, so correcting one means replacing it, which
    changes its hash and orphans every calculation cached against it. The
    comparison is therefore against how the code *behaves*, not against the
    value stored on the node: a code with no stored ``with_mpi`` runs under
    the CalcJob plugin's own default (see :func:`effective_with_mpi`), and one
    whose default already agrees with the binary is left alone.

    Each code that does disagree is relabelled ``<label>_mpi_pre`` after its
    replacement is stored under the original label. A code whose binary has
    disappeared, or whose behaviour cannot be determined, is left alone.

    ``serial_labels`` and ``parallel_labels`` are tested once per code, so
    they are materialized before the loop and may be given as generators.
    """
    from aiida import orm

    serial_labels = frozenset(serial_labels)
    parallel_labels = frozenset(parallel_labels)
    specs = code_specs()
    migrations = []

    for label in labels:
        code = orm.load_code(f"{label}@{computer.label}")
        executable_path = str(code.filepath_executable)
        if not os.path.isfile(executable_path):
            continue

        current = effective_with_mpi(code)
        if current is None:
            continue

        decision = decide_with_mpi(label, executable_path, serial_labels, parallel_labels)
        if current == decision.with_mpi:
            continue

        executable, plugin = specs[label]
        _store_code(executable, executable_path, plugin, computer, label, decision.with_mpi)
        retired_label = retire_code(code, computer, "_mpi_pre")
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
    """Print which codes were replaced because they ran the wrong way."""
    migrations = list(migrations)
    if not migrations:
        return

    replaced = ", ".join(migration.label for migration in migrations)
    click.echo(f"\nReplaced {len(migrations)} registered code(s): {replaced}")
    for migration in migrations:
        mode = "parallel" if migration.decision.with_mpi else "serial"
        click.echo(
            f"  {migration.label}: now {mode} ({migration.decision.reason}); "
            f"the previous code node is kept as '{migration.retired_label}'"
        )
    click.echo(
        f"  A replacement code node hashes differently from the one it replaces, so "
        f"calculations already run with {replaced} stop being reused from the cache "
        f"and will run again. Pass --no-migrate to keep the existing codes instead."
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
