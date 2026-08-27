"""Shared test data, helper classes, and pytest fixtures for koopmans2.

Definitions live here; ``conftest.py`` just re-exports the fixtures so
pytest's collection machinery picks them up for every test module. Mirrors
the pattern used by the sibling ``aiida-koopmans2/tests/fixtures.py``.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

# ----------------------------------------------------------------------
# Plain-data fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def tutorials_dir() -> Path:
    """Return the path to the tutorials directory shipped with the docs."""
    return Path(__file__).parent.parent / "docs" / "source" / "tutorials"


@pytest.fixture
def read_input_dict(tmp_path: Path) -> Callable[[dict[str, Any]], Any]:
    """Return a callable parsing an input dict the way the CLI reads a file.

    Writes the dict to JSON and hands it to
    :func:`koopmans.input_file.read_input_file`, so a rejected input raises
    the ``ValueError`` carrying the input-file error report rather than a
    raw ``ValidationError``.
    """
    from koopmans.input_file import read_input_file

    def _read(input_dict: dict[str, Any]) -> Any:
        path = tmp_path / "input.json"
        path.write_text(json.dumps(input_dict))
        return read_input_file(path)

    return _read


@pytest.fixture
def write_multiframe_xyz() -> Callable[..., Path]:
    """Return a factory that writes a multi-frame water ``xyz`` file.

    The factory signature is ``(directory, n_frames, *, xyz_cell=...) -> Path``.
    Each frame is a slightly displaced 3-atom water molecule in Cartesian
    Angstrom; ``xyz_cell`` (default a 5 Angstrom cube) is written into every
    frame's ``Lattice=`` header so tests can assert the input-file cell wins
    over the one embedded in the xyz.
    """

    def _write(
        directory: Path,
        n_frames: int = 3,
        *,
        xyz_cell: float = 5.0,
        filename: str = "snapshots.xyz",
    ) -> Path:
        base = [
            ("O", (0.000, 0.000, 0.000)),
            ("H", (0.757, 0.586, 0.000)),
            ("H", (-0.757, 0.586, 0.000)),
        ]
        lattice = f"{xyz_cell} 0.0 0.0 0.0 {xyz_cell} 0.0 0.0 0.0 {xyz_cell}"
        lines: list[str] = []
        for frame in range(n_frames):
            shift = 0.01 * frame
            lines.append(str(len(base)))
            lines.append(f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"')
            for symbol, (x, y, z) in base:
                lines.append(f"{symbol} {x + shift:.6f} {y:.6f} {z:.6f}")
        path = directory / filename
        path.write_text("\n".join(lines) + "\n")
        return path

    return _write


# ----------------------------------------------------------------------
# Recorded binary-inspection evidence (MPI capability detection)
# ----------------------------------------------------------------------
# Transcribed from ``nm -D`` and ``ldd`` run on a Debian box carrying an
# OpenMPI cmake build of Quantum ESPRESSO 7.4, a serial wannier90 build, and
# an Intel-MPI wannier90.x under /usr/local/bin.

# ``nm -D`` on an Intel-MPI pw.x, which calls MPI from the executable itself.
NM_CALLS_MPI_INIT = """\
                 U mpi_abort_
                 U mpi_allreduce_
                 U mpi_init_
                 U mpi_initialized_
"""

# ``nm -D`` on the GNU cmake build of pw.x. The MPI Fortran common blocks are
# copy-relocated into the executable (type B, so defined, not undefined) but
# no MPI call is: those live in libqe_modules.
NM_NO_MPI_CALL = """\
0000000000004380 B mpi_fortran_argvs_null_
00000000000043a0 B mpi_fortran_bottom_
                 U _gfortran_set_args
"""

# ``nm -D`` on a serial wannier90.x: no MPI entry at all.
NM_SERIAL = """\
0000000000004140 B __command_line_options_MOD_command_line
                 U _gfortran_st_write
"""

# ``nm -D`` on a program that asks whether MPI is running without starting it.
# ``MPI_Init`` is a substring of every one of these names.
NM_MPI_INITIALIZED_ONLY = """\
                 U mpi_initialized_
                 U MPI_Initialized
"""

# The libraries of a QE binary, keyed by soname: the MPI calls are in
# libqe_modules, and the OpenMPI Fortran bindings carry undefined PMPI_Init
# entries of their own.
LIBS_QE = {
    "libqe_pw.so.7": NM_SERIAL,
    "libqe_modules.so.7": "                 U mpi_init_\n                 U mpi_initialized_\n",
    "libmpi_mpifh.so.40": "                 U PMPI_Init\n                 U PMPI_Init_thread\n",
}

# The libraries of a serial program linked with ``-lmpi``: the runtime is
# there, but nothing outside it calls MPI.
LIBS_SERIAL_LINKING_MPI = {
    "libmpi.so.40": "0000000000080bb0 T MPI_Init\n",
    "libmpi_mpifh.so.40": "                 U PMPI_Init\n                 U PMPI_Init_thread\n",
    "libopen-pal.so.40": NM_SERIAL,
    "libc.so.6": NM_SERIAL,
}

# The libraries of a serial program linked against a parallel build of HDF5.
# HDF5's own MPI calls are indistinguishable from the program's, so this is
# read as parallel — the documented residual false positive.
LIBS_PARALLEL_HDF5 = {
    "libhdf5.so.1000": "                 U MPI_Init\n                 U MPI_Initialized\n",
    "libc.so.6": NM_SERIAL,
}

# The libraries of a genuinely serial wannier90.x.
LIBS_SERIAL = {
    "libwannier90.so.4": NM_SERIAL,
    "libopenblas.so.0": NM_SERIAL,
}

# ----------------------------------------------------------------------
# A serial binary whose OpenMPI libraries the loader actually resolves
# ----------------------------------------------------------------------
# Recorded from ``int main(void) { return 0; }`` compiled through the OpenMPI
# wrapper on a Debian box. The wrapper records libmpi_mpifh in DT_NEEDED, and
# that library carries three undefined PMPI_Init entries — so anything that
# walks it reads this program as parallel.
#
# The Intel-MPI wannier90.x under /usr/local/bin cannot stand in for this: its
# libmpi.so.12 is unresolvable here, so ``linked_libraries`` drops it and the
# binary reads serial whether or not the runtime is excluded.

# ``ldd`` on that program: libmpi_mpifh and libmpi both resolve to real files.
LDD_SERIAL_LINKING_OPENMPI = """\
\tlinux-vdso.so.1 (0x00007ffcd5dac000)
\tlibmpi_mpifh.so.40 => /lib/x86_64-linux-gnu/libmpi_mpifh.so.40 (0x00007b656e545000)
\tlibmpi.so.40 => /lib/x86_64-linux-gnu/libmpi.so.40 (0x00007b656e40e000)
\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007b656e000000)
\tlibopen-pal.so.40 => /lib/x86_64-linux-gnu/libopen-pal.so.40 (0x00007b656e35b000)
\tlibopen-rte.so.40 => /lib/x86_64-linux-gnu/libopen-rte.so.40 (0x00007b656e29c000)
\tlibm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007b656df19000)
\tlibhwloc.so.15 => /lib/x86_64-linux-gnu/libhwloc.so.15 (0x00007b656e240000)
\t/lib64/ld-linux-x86-64.so.2 (0x00007b656e5cb000)
\tlibz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007b656dec8000)
"""

# ``nm -D`` on that program: weak libc references and nothing else.
NM_SERIAL_MPIF90 = """\
                 w __cxa_finalize@GLIBC_2.2.5
                 w __gmon_start__
                 w _ITM_deregisterTMCloneTable
                 w _ITM_registerTMCloneTable
                 U __libc_start_main@GLIBC_2.34
"""

# ``nm -D`` on OpenMPI's Fortran bindings: the trap, since these are undefined.
NM_LIBMPI_MPIFH = """\
                 U PMPI_Init
                 U PMPI_Initialized
                 U PMPI_Init_thread
"""

# ``nm -D`` on the OpenMPI core, which defines MPI_Init rather than calling it.
NM_LIBMPI = """\
0000000000080bb0 T MPI_Init
0000000000080d20 T MPI_Init_thread
00000000000603e0 T ompi_init_preconnect_mpi
"""

# That program's whole closure, keyed by the path ``ldd`` resolved. Everything
# outside the OpenMPI libraries is free of any MPI entry.
SERIAL_OPENMPI_CLOSURE = {
    "/lib/x86_64-linux-gnu/libmpi_mpifh.so.40": NM_LIBMPI_MPIFH,
    "/lib/x86_64-linux-gnu/libmpi.so.40": NM_LIBMPI,
    "/lib/x86_64-linux-gnu/libopen-pal.so.40": NM_SERIAL,
    "/lib/x86_64-linux-gnu/libopen-rte.so.40": NM_SERIAL,
    "/lib/x86_64-linux-gnu/libc.so.6": NM_SERIAL,
    "/lib/x86_64-linux-gnu/libm.so.6": NM_SERIAL,
    "/lib/x86_64-linux-gnu/libhwloc.so.15": NM_SERIAL,
    "/lib64/ld-linux-x86-64.so.2": NM_SERIAL,
    "/lib/x86_64-linux-gnu/libz.so.1": NM_SERIAL,
}


@pytest.fixture
def binary_probe() -> Callable[..., Any]:
    """Return a factory building a ``BinaryProbe`` from recorded probe output.

    The factory signature is ``(dynamic_symbols="", library_symbols=None,
    raw_strings="") -> BinaryProbe``.
    """
    from koopmans.aiida.setup.codes import BinaryProbe

    def _build(
        dynamic_symbols: str = "",
        library_symbols: dict[str, str] | None = None,
        raw_strings: str = "",
    ) -> Any:
        return BinaryProbe(dynamic_symbols, library_symbols or {}, raw_strings)

    return _build


@pytest.fixture
def replay_probes(monkeypatch: Any, stub_executable: Callable[[str], Path]) -> Callable[..., Path]:
    """Return a factory replaying recorded ``nm``/``ldd``/``strings`` output.

    The factory signature is ``(dynamic_symbols, ldd_output="",
    library_symbols=None, raw_strings="") -> Path``, where ``library_symbols``
    is keyed by the path ``ldd`` resolved. It replaces
    :func:`koopmans.aiida.setup.codes._run_probe` and returns a real file to
    probe, so ``collect_mpi_evidence`` runs its whole pipeline — ldd parsing,
    the runtime skip, the library walk — over the recording.
    """

    def _install(
        dynamic_symbols: str,
        ldd_output: str = "",
        library_symbols: dict[str, str] | None = None,
        raw_strings: str = "",
    ) -> Path:
        from koopmans.aiida.setup import codes

        executable = stub_executable("recorded.x")
        libraries = library_symbols or {}

        def _replay(command: list[str], path: str) -> str:
            if command[0] == "ldd":
                return ldd_output
            if command[0] == "strings":
                return raw_strings
            if path == str(executable):
                return dynamic_symbols
            return libraries.get(path, "")

        monkeypatch.setattr(codes, "_run_probe", _replay)
        return executable

    return _install


# ----------------------------------------------------------------------
# Compiled probe corpus
# ----------------------------------------------------------------------
# Sources for four ELF binaries, one per way the MPI probe can decide. They
# call ``MPI_Init`` without any MPI implementation: a stub ``libmpi.so.40``
# defines the symbol, which is all the linker and the probe ask for. None of
# them includes a header, so they compile ``-ffreestanding``.

PROBE_SOURCES: dict[str, str] = {
    # The MPI runtime: defines MPI_Init.
    "libmpi.so.40": "int MPI_Init(int *argc, char ***argv) { return argc == 0 && argv == 0; }\n",
    # The runtime's Fortran bindings: an MPI library that itself calls
    # MPI_Init, as OpenMPI's libmpi_mpifh does.
    "libmpi_mpifh.so.40": (
        "int MPI_Init(int *, char ***);\nint fortran_binding(void) { return MPI_Init(0, 0); }\n"
    ),
    # A library that is not part of the runtime and calls MPI_Init, as
    # Quantum ESPRESSO's libqe_modules does.
    "libworker.so": (
        "int MPI_Init(int *, char ***);\nint worker(void) { return MPI_Init(0, 0); }\n"
    ),
    "serial.x": "int main(void) { return 0; }\n",
    "calls_mpi.x": ("int MPI_Init(int *, char ***);\nint main(void) { return MPI_Init(0, 0); }\n"),
    "library_calls_mpi.x": "int worker(void);\nint main(void) { return worker(); }\n",
    "runtime_only.x": (
        "int fortran_binding(void);\nint main(void) { return fortran_binding(); }\n"
    ),
}

# What each binary links, in link order.
PROBE_LINKS: dict[str, tuple[str, ...]] = {
    "libmpi_mpifh.so.40": ("libmpi.so.40",),
    "libworker.so": ("libmpi.so.40",),
    "calls_mpi.x": ("libmpi.so.40",),
    "library_calls_mpi.x": ("libworker.so", "libmpi.so.40"),
    "runtime_only.x": ("libmpi_mpifh.so.40",),
}


@pytest.fixture(scope="session")
def compiled_binaries(tmp_path_factory: Any) -> dict[str, Path]:
    """Return ELF binaries built from C, keyed by the verdict each must get.

    Keys are ``serial.x`` (no MPI anywhere), ``calls_mpi.x`` (the executable
    calls MPI_Init), ``library_calls_mpi.x`` (a library it links does) and
    ``runtime_only.x`` (the only caller is the MPI runtime). Skips when the
    toolchain to build or inspect them is missing.
    """
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("no C compiler to build the probe binaries")
    for tool in ("nm", "ldd", "strings"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} is not installed")

    directory = tmp_path_factory.mktemp("mpi-probe-binaries")
    built = {}
    for name, source in PROBE_SOURCES.items():
        source_path = directory / f"{name.partition('.')[0]}.c"
        source_path.write_text(source)
        output = directory / name
        command = [compiler, "-ffreestanding"]
        if name.endswith(".so") or ".so." in name:
            command += ["-shared", "-fPIC"]
        command += ["-o", str(output), str(source_path), f"-L{directory}"]
        command += [f"-l:{library}" for library in PROBE_LINKS.get(name, ())]
        command += [f"-Wl,-rpath,{directory}"]
        result = subprocess.run(command, capture_output=True, text=True)  # noqa: S603
        if result.returncode != 0:
            raise RuntimeError(f"could not build {name}: {result.stderr.strip()}")
        built[name] = output
    return built


# ----------------------------------------------------------------------
# Broken-upstream-fixture overrides
# ----------------------------------------------------------------------
# The deprecated ``aiida.manage.tests.pytest_fixtures`` chain calls a
# removed ``Profile.clear_profile()`` during teardown; override with no-ops
# so tests that don't need an isolated DB aren't tripped. Tests that *do*
# need isolation should request ``aiida_profile_clean`` directly.


@pytest.fixture(scope="function")
def clear_database_after_test(aiida_profile: Any) -> Iterator[Any]:
    """Override the deprecated-and-broken upstream fixture with a no-op yield."""
    yield aiida_profile


@pytest.fixture(scope="function")
def clear_database(clear_database_after_test: Any) -> Iterator[None]:
    """Alias override for ``clear_database``."""
    yield


# ----------------------------------------------------------------------
# Codes + pseudos for dispatcher tests that build (but do not run) workgraphs
# ----------------------------------------------------------------------


@pytest.fixture
def stub_executable(tmp_path: Path) -> Callable[[str], Path]:
    """Return a factory writing an executable shell script that carries no MPI evidence.

    The factory signature is ``(name) -> Path``. A shell script has no dynamic
    symbols and no linked libraries, so the MPI probe finds nothing and the
    code registers serial.
    """

    def _write(name: str = "pw.x") -> Path:
        path = tmp_path / name
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
        return path

    return _write


@pytest.fixture
def code_without_mpi_flag(stub_executable: Callable[[str], Path]) -> Callable[..., Any]:
    """Return a factory storing a code whose ``with_mpi`` was never set.

    The factory signature is ``(label, plugin, computer) -> InstalledCode``.
    It models a code registered before koopmans inspected binaries: aiida-core
    then takes the ``withmpi`` default of whichever CalcJob is submitted.
    """

    def _store(label: str, plugin: str | None, computer: Any) -> Any:
        from aiida.orm import InstalledCode

        return InstalledCode(
            label=label,
            computer=computer,
            filepath_executable=str(stub_executable(f"{label}.x")),
            default_calc_job_plugin=plugin,
            with_mpi=None,
        ).store()

    return _store


@pytest.fixture
def localhost_computer(aiida_computer_local: Any) -> Any:
    """Return a computer whose label is literally ``localhost``.

    aiida-core's ``aiida_localhost`` fixture now suffixes its computer label
    with the pytest-xdist worker id (``localhost-master``, ...), but the
    dispatcher resolves codes as ``<name>@localhost`` — the label real
    profiles use — so the dummy codes must live on a literal one.
    """
    return aiida_computer_local(label="localhost")


@pytest.fixture
def localhost_code(localhost_computer: Any) -> Any:
    """Return a get-or-create factory for dummy codes on the literal ``localhost``.

    Unlike aiida-core's ``aiida_code_installed`` factory, the lookup matches
    label *and* computer — a same-labelled code another test left on a
    different computer (e.g. ``test_code_setup``'s ``pw`` on the suffixed
    ``aiida_localhost``) must not shadow the one the dispatcher resolves as
    ``<label>@localhost``.
    """
    from aiida.common.exceptions import NotExistent
    from aiida.orm import InstalledCode, load_code

    def factory(label: str, entry_point: str) -> Any:
        """Return the ``<label>@localhost`` code, creating it if absent."""
        try:
            return load_code(f"{label}@{localhost_computer.label}")
        except NotExistent:
            return InstalledCode(
                label=label,
                computer=localhost_computer,
                default_calc_job_plugin=entry_point,
                filepath_executable="/bin/true",
            ).store()

    return factory


@pytest.fixture
def installed_pw_code(localhost_code: Any) -> Any:
    """Register a dummy ``pw@localhost`` code so ``load_code`` succeeds."""
    return localhost_code("pw", "quantumespresso.pw")


@pytest.fixture
def installed_kcp_code(localhost_code: Any) -> Any:
    """Register a dummy ``kcp@localhost`` code so ``load_code`` succeeds."""
    return localhost_code("kcp", "koopmans.kcp")


@pytest.fixture
def installed_kcw_code(localhost_code: Any) -> Any:
    """Register a dummy ``kcw@localhost`` code so ``load_code`` succeeds."""
    return localhost_code("kcw", "koopmans.kcw_wann2kc")


@pytest.fixture
def installed_ph_code(localhost_code: Any) -> Any:
    """Register a dummy ``ph@localhost`` code so ``load_code`` succeeds."""
    return localhost_code("ph", "quantumespresso.ph")


@pytest.fixture
def installed_wannier_codes(localhost_code: Any) -> dict[str, Any]:
    """Register dummy ``wannier90`` / ``pw2wannier90`` codes for DFPT builds."""
    return {
        "wannier90": localhost_code("wannier90", "wannier90.wannier90"),
        "pw2wannier90": localhost_code("pw2wannier90", "quantumespresso.pw2wannier90"),
    }


@pytest.fixture
def installed_fold_codes(localhost_code: Any) -> dict[str, Any]:
    """Register dummy ``wann2kcp`` / ``merge_evc`` codes for the fold path."""
    return {
        "wann2kcp": localhost_code("wann2kcp", "koopmans.wann2kcp"),
        "merge_evc": localhost_code("merge_evc", "koopmans.merge_evc"),
    }


# Header excerpts transcribed from real pseudopotentials, which the synthetic
# streams below cannot stand in for: generators disagree on how to spell a UPF
# boolean, PAW files set the ultrasoft flag as well as their own, and the v1
# layout is not XML at all.

# PSlibrary's ultrasoft silicon (Si.pbe-n-rrkjus_psl.1.0.0.UPF): "true", not
# "T", and ``pseudo_type`` reads USPP rather than US.
UPF_V2_ULTRASOFT_HEADER = """\
<UPF version="2.0.1">
  <PP_INFO>
Pseudopotential type: USPP
  </PP_INFO>
  <PP_HEADER
     element="Si"
     pseudo_type="USPP"
     relativistic="scalar"
     is_ultrasoft="true"
     is_paw="false"
     z_valence="4.000000000000E+000"/>
</UPF>
"""

# PSlibrary's PAW silicon (Si.pbe-n-kjpaw_psl.1.0.0.UPF), which sets both
# flags: reading ``is_ultrasoft`` alone would call this one ultrasoft.
UPF_V2_PAW_HEADER = """\
<UPF version="2.0.1">
  <PP_HEADER
     element="Si"
     pseudo_type="PAW"
     is_ultrasoft="true"
     is_paw="true"
     z_valence="4.000000000000E+000"/>
</UPF>
"""

# A UPF v1 ultrasoft carbon (aiida-core's C_pbe_v1.2.uspp.F.UPF): no
# ``<UPF version=...>`` wrapper and a fixed-format header whose third line
# carries the type. Every field of the block is transcribed, because a v1
# header is read as a whole: the fields are positional, so a reader that
# stopped early would be reading them off a file it could not otherwise use.
UPF_V1_ULTRASOFT_HEADER = """\
<PP_INFO>
Generated using Vanderbilt code, version   7  3  6
</PP_INFO>
<PP_HEADER>
   0                   Version Number
  C                    Element
   US                  Ultrasoft pseudopotential
    T                  Nonlinear Core Correction
SLA  PW   PBE  PBE     PBE  Exchange-Correlation functional
    4.00000000000      Z valence
  -10.81268860050      Total energy
    0.00000    0.00000 Suggested cutoff for wfc and rho
    1                  Max angular momentum component
  721                  Number of points in mesh
    2    4             Number of Wavefunctions, Number of Projectors
 Wavefunctions         nl  l   occ
                       2S  0  2.00
                       2P  1  2.00
</PP_HEADER>
"""

# The same block with the other type Quantum ESPRESSO accepts on that line.
# Its reader takes US, PAW, NC or 1/r there (upflib/read_upf_v1.f90), so a v1
# PAW file is a file koopmans must refuse; this one is built rather than
# transcribed, no v1 PAW pseudopotential being at hand.
UPF_V1_PAW_HEADER = """\
<PP_HEADER>
   0                   Version Number
  C                    Element
   PAW                 Projector augmented-wave
    T                  Nonlinear Core Correction
SLA  PW   PBE  PBE     PBE  Exchange-Correlation functional
    4.00000000000      Z valence
  -10.81268860050      Total energy
    0.00000    0.00000 Suggested cutoff for wfc and rho
    1                  Max angular momentum component
  721                  Number of points in mesh
    2    4             Number of Wavefunctions, Number of Projectors
 Wavefunctions         nl  l   occ
                       2S  0  2.00
                       2P  1  2.00
</PP_HEADER>
"""

# Every PSlibrary pseudopotential that embeds its generation input carries a
# Fortran namelist inside PP_INFO, whose bare ``&`` makes the file invalid
# XML. An XML parser rejects the whole file; the header still reads.
UPF_V2_ULTRASOFT_WITH_NAMELIST = """\
<UPF version="2.0.1">
  <PP_INFO>
<PP_INPUTFILE>
 &input
   title='O',
   config='[He] 2s2 2p4 3d-2',
 /
</PP_INPUTFILE>
  </PP_INFO>
  <PP_HEADER
     element="O"
     pseudo_type="USPP"
     is_ultrasoft="true"
     is_paw="false"
     z_valence="6.000000000000E+000"/>
</UPF>
"""

# A header that flags itself ultrasoft without naming a ``pseudo_type``, which
# is the only thing the boolean flags decide: every other file states both.
UPF_V2_FLAGGED_BUT_UNNAMED = """\
<UPF version="2.0.1">
  <PP_HEADER
     element="Si"
     is_ultrasoft="true"
     is_paw="false"
     z_valence="4.000000000000E+000"/>
</UPF>
"""

# An ultrasoft header on a file that stops partway through its first data
# block, as an interrupted copy does. Reading the whole file raises; the header
# is intact and says what the pseudopotential is.
UPF_V2_ULTRASOFT_WITH_UNREADABLE_BODY = """\
<UPF version="2.0.1">
  <PP_HEADER
     element="Si"
     pseudo_type="USPP"
     is_ultrasoft="true"
     is_paw="false"
     z_valence="4.000000000000E+000"/>
  <PP_LOCAL type="real" size="4" columns="4">
 -1.0000000000E+00 -2.0000000000E+00
"""

# SG15's ONCV silicon (Si_ONCV_PBE-1.2.upf), the norm-conserving control.
UPF_V2_NORM_CONSERVING_HEADER = """\
<UPF version="2.0.1">
  <PP_HEADER
     element="Si"
     pseudo_type="NC"
     relativistic="scalar"
     is_ultrasoft="F"
     is_paw="F"
     z_valence="4.000000000000E+000"/>
</UPF>
"""


def fake_upf_content(
    element: str,
    z_valence: float,
    has_so: bool | None = False,
    info: str | None = None,
    pseudo_type: str | None = None,
    number_of_wfc: int | None = 2,
) -> str:
    """Return a synthetic UPF v2 stream for the fake test pseudos.

    Shaped for the line-based block extractors in aiida-wannier90-workflows'
    pseudo utilities: ``<PP_HEADER`` and its ``/>`` sit on their own lines and
    ``PP_PSWFC`` provides an s+p valence (4 projectors per atom) so projection
    counting works. ``has_so`` must be present for that machinery — real UPF
    generators always write it, and an attribute-bearing header without it
    makes the upstream sniffing crash, which the dispatcher converts into an
    error naming the pseudo. ``has_so=None`` omits the flag to exercise
    exactly that guard. ``info`` fills the ``PP_INFO`` block real generators
    write, which gives two otherwise identical streams content of their own.
    ``pseudo_type`` writes the header attribute real generators use to say
    what kind of pseudopotential this is ("NC", "US", "PAW"), along with the
    ``is_ultrasoft``/``is_paw`` flags that agree with it; omitted by default,
    which is the header that says nothing.
    ``number_of_wfc`` is the header's count of ``PP_PSWFC`` wavefunctions
    (2, matching the s+p block); ``0`` writes an empty-valence pseudo with
    the block dropped, ``None`` omits the attribute while keeping the block.
    """
    has_so_line = "" if has_so is None else f'has_so="{"T" if has_so else "F"}"\n'
    info_block = "" if info is None else f"<PP_INFO>\n{info}\n</PP_INFO>\n"
    if pseudo_type is None:
        type_lines = ""
    else:
        ultrasoft = "T" if pseudo_type.upper() in {"US", "USPP"} else "F"
        paw = "T" if pseudo_type.upper() == "PAW" else "F"
        type_lines = f'pseudo_type="{pseudo_type}"\nis_ultrasoft="{ultrasoft}"\nis_paw="{paw}"\n'
    wfc_line = "" if number_of_wfc is None else f'number_of_wfc="{number_of_wfc}"\n'
    pswfc_block = (
        ""
        if number_of_wfc == 0
        else '<PP_PSWFC>\n<PP_CHI.1 l="0"/>\n<PP_CHI.2 l="1"/>\n</PP_PSWFC>\n'
    )
    return (
        f'<UPF version="2.0.1">\n'
        f"{info_block}"
        f'<PP_HEADER\nelement="{element}"\n'
        f'z_valence="{z_valence}"\n{type_lines}{wfc_line}{has_so_line}/>\n'
        f"{pswfc_block}"
        f"</UPF>\n"
    )


# One member per (element, revision, relativistic variant), laid out flat under
# a single directory as the published SG15 tarball is. The coverage mirrors the
# real archive's: silicon is fully relativistic only at 1.1 and oxygen only at
# 1.0, so a family holding both is one that composed 1.1 over 1.0. Every file
# names its own revision in ``PP_INFO``, which is what lets a test say which of
# them an installed pseudopotential is.
SG15_ARCHIVE_MEMBERS: dict[str, tuple[str, float, bool, str]] = {
    # 1.1 before 1.0 deliberately: overlay precedence must come from the
    # revision, not from the order the tarball happens to list its members.
    "sg15_oncv_upf_2020-02-06/Si_ONCV_PBE-1.1.upf": ("Si", 4.0, False, "1.1"),
    "sg15_oncv_upf_2020-02-06/Si_ONCV_PBE-1.0.upf": ("Si", 4.0, False, "1.0"),
    "sg15_oncv_upf_2020-02-06/O_ONCV_PBE-1.0.upf": ("O", 6.0, False, "1.0"),
    "sg15_oncv_upf_2020-02-06/Si_ONCV_PBE-1.2.upf": ("Si", 4.0, False, "1.2"),
    "sg15_oncv_upf_2020-02-06/O_ONCV_PBE-1.2.upf": ("O", 6.0, False, "1.2"),
    "sg15_oncv_upf_2020-02-06/O_ONCV_PBE_FR-1.0.upf": ("O", 6.0, True, "1.0"),
    "sg15_oncv_upf_2020-02-06/Si_ONCV_PBE_FR-1.1.upf": ("Si", 4.0, True, "1.1"),
}


@pytest.fixture
def offline_sg15_archive(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Serve a synthetic SG15 tarball from ``urlopen``, pinned to its own checksum.

    Returns each member's UPF stream keyed by filename, so a test can assert
    which revision an installed pseudopotential came from. The published
    archive is never downloaded.
    """
    import hashlib
    import tarfile
    import urllib.request

    from koopmans.aiida.setup.pseudos import _sg15

    contents = {
        Path(name).name: fake_upf_content(
            element, z_valence, has_so=has_so, info=f"SG15 ONCV revision {revision}"
        )
        for name, (element, z_valence, has_so, revision) in SG15_ARCHIVE_MEMBERS.items()
    }

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        # The real tarball carries a directory entry and a README the
        # installer's member walk must step over, so the fixture does too.
        directory = tarfile.TarInfo("sg15/")
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)
        readme = b"SG15 ONCV potentials"
        info = tarfile.TarInfo("sg15/README")
        info.size = len(readme)
        tar.addfile(info, io.BytesIO(readme))
        for name in SG15_ARCHIVE_MEMBERS:
            payload = contents[Path(name).name].encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    archive = buffer.getvalue()

    monkeypatch.setattr(_sg15, "ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(urllib.request, "urlopen", lambda url: io.BytesIO(archive))
    return contents


@pytest.fixture
def installed_decompose_code(localhost_code: Any) -> Any:
    """Register a dummy ``pw2wannier90@localhost`` code for the decompose pass."""
    return localhost_code("pw2wannier90", "quantumespresso.pw2wannier90")


def _install_fake_family(
    label: str,
    elements: dict[str, float],
    cutoffs: bool = False,
    has_so: bool = False,
    recommended_cutoffs: bool = True,
    pseudo_type: str | None = None,
    number_of_wfc: int | None = 2,
) -> Any:
    """Install (or fetch) a fake pseudopotential family with synthetic UPF streams.

    The streams are enough for ``UpfData`` validation, not physically
    meaningful pseudos. ``cutoffs=True`` builds a
    ``CutoffsPseudoPotentialFamily`` with recommended cutoffs — needed by
    a build that states none of its own; ``cutoffs=False`` builds a plain
    ``PseudoPotentialFamily``, the shape both ``aiida-pseudo install family``
    and ``_sg15.install`` produce.
    ``recommended_cutoffs=False`` leaves the cutoffs family with no stringency
    defined, the shape ``-F pseudo.family.cutoffs`` produces on its own.
    ``has_so=True`` marks every pseudo fully relativistic.
    ``pseudo_type`` writes that kind into every pseudo's header.
    ``number_of_wfc=0`` gives every pseudo a header reporting no ``PP_PSWFC``
    atomic wavefunctions.
    """
    from aiida.common.exceptions import NotExistent
    from aiida_pseudo.data.pseudo.upf import UpfData
    from aiida_pseudo.groups.family import (
        CutoffsPseudoPotentialFamily,
        PseudoPotentialFamily,
    )

    cls = CutoffsPseudoPotentialFamily if cutoffs else PseudoPotentialFamily
    try:
        return cls.collection.get(label=label)
    except NotExistent:
        pass

    family = cls(label=label, description=f"fake {label} family for tests")
    family.store()
    pseudos = []
    for element, z_valence in elements.items():
        content = fake_upf_content(
            element, z_valence, has_so=has_so, pseudo_type=pseudo_type, number_of_wfc=number_of_wfc
        )
        upf = UpfData(io.BytesIO(content.encode("utf-8")), filename=f"{element}.upf")
        pseudos.append(upf.store())
    family.add_nodes(pseudos)
    if cutoffs and recommended_cutoffs:
        family.set_cutoffs(
            {element: {"cutoff_wfc": 30.0, "cutoff_rho": 240.0} for element in elements},
            stringency="normal",
        )
    return family


@pytest.fixture
def fake_sg15_pseudo_family(aiida_profile: Any) -> Any:
    """Install a minimal fake ``SG15/1.2/PBE/SR`` family (H, O and Si pseudos)."""
    return _install_fake_family("SG15/1.2/PBE/SR", {"H": 1.0, "O": 6.0, "Si": 4.0})


@pytest.fixture
def fake_sg15_cutoffs_family(aiida_profile: Any) -> Any:
    """Install a minimal fake ``SG15/1.0/PBE/SR`` cutoffs family (O and Si).

    A different version label from ``fake_sg15_pseudo_family`` so both can
    coexist in one session profile.
    """
    return _install_fake_family("SG15/1.0/PBE/SR", {"O": 6.0, "Si": 4.0}, cutoffs=True)


@pytest.fixture
def fake_sg15_family_without_cutoffs(aiida_profile: Any) -> Any:
    """Install ``SG15/1.1/PBE/FR`` as a cutoffs family with no stringency defined.

    The half-configured shape a user reaches by passing
    ``-F pseudo.family.cutoffs`` and never running ``aiida-pseudo family
    cutoffs set``: it can recommend no cutoffs. A label of its own so it
    coexists with the other SG15 fixtures in one session profile.
    """
    return _install_fake_family(
        "SG15/1.1/PBE/FR", {"Si": 4.0}, cutoffs=True, recommended_cutoffs=False
    )


def count_pw_bands_runs(wg: Any) -> int:
    """Count the graph's pw steps that declare ``calculation = 'bands'``.

    Counting tasks *named* ``bands`` is vacuous: aiida-workgraph uniquifies
    colliding task names, so a duplicated run shows up as ``bands1`` and
    the name count stays at 1. The declared ``CONTROL.calculation`` on the
    step's own ``pw`` namespace cannot be disguised that way.
    """
    count = 0
    for graph_task in wg.tasks:
        try:
            parameters = graph_task.inputs["pw"]["parameters"].value
        except (AttributeError, KeyError, TypeError):
            continue
        if parameters is None:
            continue
        parameters = parameters.get_dict() if hasattr(parameters, "get_dict") else dict(parameters)
        if parameters.get("CONTROL", {}).get("calculation") == "bands":
            count += 1
    return count


def path_labels(kpoints: Any) -> list[str]:
    """Return the labels of an explicit k-path node, in path order."""
    assert kpoints is not None, "no k-path node reached the calculation"
    return [label for _, label in kpoints.labels]


@pytest.fixture
def fake_family_without_pswfc(aiida_profile: Any) -> Any:
    """Install a cutoffs family whose Si pseudo carries no ``PP_PSWFC`` block.

    The shape projwfc.x cannot project onto: the header reports
    ``number_of_wfc="0"`` and the block is absent, so the projected DOS
    must be skipped with a warning rather than attempted.
    """
    return _install_fake_family("MyPseudos/no-pswfc", {"Si": 4.0}, cutoffs=True, number_of_wfc=0)


@pytest.fixture
def fake_ultrasoft_family(aiida_profile: Any) -> Any:
    """Install a self-built family whose Si pseudopotential is ultrasoft.

    The label says nothing about the kind of pseudopotential inside, which is
    the whole point: only the header does.
    """
    return _install_fake_family("MyPseudos/ultrasoft", {"Si": 4.0}, cutoffs=True, pseudo_type="US")


@pytest.fixture
def fake_coulomb_family(aiida_profile: Any) -> Any:
    """Install a self-built family whose Si pseudopotential is a bare Coulomb potential."""
    return _install_fake_family("MyPseudos/coulomb", {"Si": 4.0}, cutoffs=True, pseudo_type="1/r")


@pytest.fixture
def fake_paw_family(aiida_profile: Any) -> Any:
    """Install a self-built family whose Si pseudopotential is PAW."""
    return _install_fake_family("MyPseudos/paw", {"Si": 4.0}, cutoffs=True, pseudo_type="PAW")


@pytest.fixture
def fake_declared_nc_family(aiida_profile: Any) -> Any:
    """Install a self-built family whose Si pseudopotential declares itself NC."""
    return _install_fake_family("MyPseudos/nc", {"Si": 4.0}, cutoffs=True, pseudo_type="NC")


@pytest.fixture
def fake_user_built_family(aiida_profile: Any) -> Any:
    """Install a plain family, the shape ``aiida-pseudo install family`` produces.

    No cutoff stringencies are even representable on this family class.
    """
    return _install_fake_family("MyPseudos/local", {"Si": 4.0})


@pytest.fixture
def fake_sg15_fr_cutoffs_family(aiida_profile: Any) -> Any:
    """Install a fake fully-relativistic ``SG15/1.0/PBE/FR`` cutoffs family (O and Si)."""
    return _install_fake_family("SG15/1.0/PBE/FR", {"O": 6.0, "Si": 4.0}, cutoffs=True, has_so=True)


@pytest.fixture
def fake_pseudodojo_lda_family(aiida_profile: Any) -> Any:
    """Install a minimal fake ``PseudoDojo/0.4/LDA/SR/standard/upf`` family.

    Zn (z=20) and O (z=6) — enough for the dispatcher's electron counting
    (ZnO: nelec 52, nocc 26).
    """
    return _install_fake_family("PseudoDojo/0.4/LDA/SR/standard/upf", {"Zn": 20.0, "O": 6.0})


def silicon_pw_input(
    *,
    pseudo_library: str = "SG15/1.2/PBE/SR",
    parallelization: dict[str, Any] | None = None,
    calculator_parameters: dict[str, Any] | None = None,
    kpoints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a minimal silicon ``dft_bands`` input dict for the wiring tests.

    ``calculator_parameters`` replaces the default ``{"ecutwfc": 20.0}`` block
    outright, so a caller can leave the cutoffs out entirely.
    """
    d: dict[str, Any] = {
        "workflow": {"task": "dft_bands", "pseudo_library": pseudo_library},
        "atoms": {
            "cell_parameters": {"periodic": True, "ibrav": 2, "celldms": {"1": 10.2622}},
            "atomic_positions": {
                "units": "crystal",
                "positions": [["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25]],
            },
        },
        "kpoints": kpoints or {"grid": [2, 2, 2], "offset": [0, 0, 0]},
        "calculator_parameters": (
            {"ecutwfc": 20.0} if calculator_parameters is None else calculator_parameters
        ),
    }
    if parallelization is not None:
        d["parallelization"] = parallelization
    return d


def write_koopmans_input(directory: Path, name: str = "si.yaml") -> Path:
    """Write :func:`silicon_pw_input` into ``directory`` and return its path."""
    import yaml

    path = directory / name
    path.write_text(yaml.safe_dump(silicon_pw_input()))
    return path


def skip_profile_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for ``load_koopmans_profile``: the test profile is already loaded.

    ``koopmans.cli`` always loads the profile named "koopmans" by name,
    which does not exist under the throwaway test profile the AiiDA
    pytest fixtures set up.
    """
    import koopmans.cli as cli_mod

    monkeypatch.setattr(cli_mod, "load_koopmans_profile", lambda: None)


def pw_step_from_overrides(code: Any, structure: Any, overrides: dict[str, Any]) -> Any:
    """Return the ``pw`` sub-builder one scf or nscf step assembles from ``overrides``.

    Mirrors ``aiida_koopmans.workgraphs.pw.assemble_pw_base_step``, which hands
    a route's per-step override entry to this builder when the step runs. Use
    it on an entry taken off a built graph to see what the pw.x calculation
    receives.
    """
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

    return PwBaseWorkChain.get_builder_from_protocol(
        code=code, structure=structure, overrides=overrides
    ).pw


def make_process(
    process_type: str = "",
    caller: Any = None,
    link_label: str = "step",
    label: str = "",
    exit_status: int = 0,
    exit_message: str | None = None,
    calcjob: bool = False,
    calcfunction: bool = False,
    workfunction: bool = False,
    computer: Any = None,
    process_label: str | None = None,
    inputs: dict[str, Any] | None = None,
) -> Any:
    """Return a stored, finished process node of the given ``process_type``.

    ``process_type`` is the formal AiiDA process type string (what
    resolvers like ``koopmans.plotting`` key off); ``process_label`` is
    the display name the live progress table and ``koopmans status``
    read instead (``node.process_label``, left unset by default). Link
    ``caller`` in as this node's parent via a real ``CALL_WORK``/
    ``CALL_CALC`` link, under ``link_label``, so callers that walk the
    process tree by its call links see this node as a child. ``inputs``
    links data nodes as the process's own inputs, keyed by link label
    (``__`` separating namespace levels, e.g. ``pw__parameters``), so
    resolvers that key off a run's declared inputs (rather than its
    process type) have something to read.

    ``calcjob``, ``calcfunction`` and ``workfunction`` are mutually
    exclusive and pick a ``CalcJobNode``/``CalcFunctionNode``/
    ``WorkFunctionNode`` instead of the default ``WorkflowNode`` — the
    distinction a dumped tree's bookkeeping prune keys off
    (:func:`koopmans.aiida.dumping._write_step_io`): a
    ``CalcFunctionNode`` is a plain pyfunction, and a ``WorkFunctionNode``
    is a ``@workfunction`` (python code that only ever hands back
    *existing* Data, e.g. ``resolve_pseudo_family_task``) — neither is
    ever treated as a genuine calculation or workflow step. A
    ``PythonJob`` helper is a different case again — it needs a code to
    run on, so it is a real ``CalcJobNode`` like any domain CalcJob;
    build one with ``calcjob=True`` and a process_type naming
    ``aiida_pythonjob``'s own generic runner (see
    ``TestStepIoListing.PYTHONJOB`` in ``tests/test_dumping.py``), since
    ``_is_calcjob_step`` excludes it by comparing ``process_class``, not
    by node type.
    """
    from aiida import orm
    from aiida.common.links import LinkType
    from plumpy.process_states import ProcessState

    if calcjob:
        node: Any = orm.CalcJobNode()
    elif calcfunction:
        node = orm.CalcFunctionNode()
    elif workfunction:
        node = orm.WorkFunctionNode()
    else:
        node = orm.WorkflowNode()
    node.process_type = process_type
    node.label = label
    if calcjob:
        node.computer = computer
        node.set_option("resources", {"num_machines": 1})
    if caller is not None:
        link_type = LinkType.CALL_CALC if (calcjob or calcfunction) else LinkType.CALL_WORK
        node.base.links.add_incoming(caller, link_type=link_type, link_label=link_label)
    for name, data in (inputs or {}).items():
        input_type = LinkType.INPUT_CALC if (calcjob or calcfunction) else LinkType.INPUT_WORK
        node.base.links.add_incoming(data.store(), link_type=input_type, link_label=name)
    node.store()
    if process_label is not None:
        node.set_process_label(process_label)
    node.set_process_state(ProcessState.FINISHED)
    node.set_exit_status(exit_status)
    if exit_message is not None:
        node.set_exit_message(exit_message)
    return node


def attach(node: Any, socket: str, data: Any) -> Any:
    """Link ``data`` as an output of ``node`` under the link label ``socket``.

    A calculation (``CalcJobNode``/``CalcFunctionNode``) creates its
    outputs, so ``data`` must still be unstored; a workflow only returns
    data that already exists, so ``data`` is stored first.
    """
    from aiida import orm
    from aiida.common.links import LinkType

    if isinstance(node, (orm.CalcJobNode, orm.CalcFunctionNode)):
        data.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label=socket)
        return data.store()
    data.store()
    data.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label=socket)
    return data


def si_external_projector_tables() -> dict[str, list[dict[str, Any]]]:
    """Return the tables the dispatcher synthesizes from the fixture's ``Si.dat``.

    s + p per atom, so a two-atom cell carries 8 projectors; every entry
    is explicitly unfrozen.
    """
    return {"Si": [{"l": angular_momentum, "frozen": False} for angular_momentum in [0, 1]]}


@pytest.fixture
def si_external_projector_dir(tmp_path: Path) -> Path:
    """Write a silicon external projector directory.

    One ``Si.dat`` in pw2wannier90's radial-projector format: a leading
    comment, the ``<ngrid> <nproj>`` header, and the angular momenta (s +
    p). The radial table that would follow is never read at build time, so
    it is omitted.
    """
    directory = tmp_path / "projectors"
    directory.mkdir()
    (directory / "Si.dat").write_text("# radial projector table stand-in\n4 2\n0 1\n")
    return directory


# ----------------------------------------------------------------------
# WorkGraph → stable dict for snapshot regressions
# ----------------------------------------------------------------------


_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_SCRUB_KEYS = {
    "uuid",
    "pk",
    "id",
    "ctime",
    "mtime",
    "identifier",
    "remote_path",
    "computer",
    # Machine-specific / version-drift fields that aren't structural.
    "file_path",
    "package_version",
    "platform_version",
    "hash",
    "process",  # pyyaml block scalar of 'null\n...\n' varies by version
}


def _scrub(value: Any) -> Any:  # noqa: C901
    """Recursively replace non-deterministic fields (UUIDs, PKs, paths) with placeholders.

    Also turns AiiDA ``Node`` instances into stable ``<class:key>``
    placeholders (for ``Dict`` nodes we include the dict contents, with
    UUIDs scrubbed) so YAML serialisation of the WorkGraph dict doesn't
    trip over live ORM objects.

    Mirrors the intent of aiida-qe's ``serialize_builder``: diff structure,
    not volatile run metadata.
    """
    try:
        from aiida import orm
    except ImportError:  # pragma: no cover — aiida is always present here
        orm = None  # type: ignore[assignment]

    # Unwrap node_graph's TaggedValue (``wrapt.ObjectProxy``) so downstream
    # isinstance checks see the real underlying type.
    wrapped = getattr(value, "__wrapped__", None)
    if wrapped is not None and wrapped is not value:
        return _scrub(wrapped)

    if orm is not None:
        if isinstance(value, orm.Dict):
            return {"__aiida_dict__": _scrub(value.get_dict())}  # type: ignore[no-untyped-call]
        if isinstance(value, orm.StructureData):
            return {"__aiida_structure__": value.get_formula()}  # type: ignore[no-untyped-call]
        if isinstance(value, orm.AbstractCode):
            return {"__aiida_code__": value.full_label}
        if isinstance(value, orm.Node):
            return {"__aiida_node__": type(value).__name__}

    if isinstance(value, dict):
        return {
            key: (f"<scrubbed:{key}>" if key in _SCRUB_KEYS else _scrub(val))
            for key, val in value.items()
        }
    if isinstance(value, list | tuple):
        scrubbed = [_scrub(v) for v in value]
        # ``WorkGraph.to_dict()`` collects some namespaces by iterating
        # an unordered dict, so two runs of the same build can emit the
        # same list of port names in different orders. Sort lists of
        # plain strings to make the snapshot stable; lists holding
        # structured items (dicts, sub-lists) are left in place — their
        # order tends to carry semantic meaning (e.g. socket-connection
        # order).
        #
        # The DFPT route breaks that assumption: its ``links`` entries for
        # ``codes.pw`` / ``codes.wannier90`` swap places between processes,
        # so three builds of one input give three orderings. Nothing
        # snapshots a DFPT graph today; whoever first does will need those
        # entries sorted by ``(from_socket, to_socket)`` here, or the
        # snapshot will be flaky.
        if scrubbed and all(isinstance(v, str) for v in scrubbed):
            scrubbed.sort()
        return scrubbed
    if isinstance(value, str):
        return _UUID_RE.sub("<uuid>", value)
    # node-graph socket objects (``SocketAny`` etc.) sometimes appear in
    # the serialised workgraph payload when one ``@task.graph``'s output
    # is wired into another's input. YAML can't represent them; collapse
    # to a stable placeholder so ``data_regression`` works.
    type_name = type(value).__name__
    if type_name.startswith("Socket"):
        return f"<{type_name}>"
    return value


@pytest.fixture
def serialize_workgraph() -> Callable[..., dict[str, Any]]:
    """Return a callable that serializes a ``WorkGraph`` into a stable dict.

    The returned dict records the task name list, the task count, and a
    scrubbed version of ``WorkGraph.to_dict()`` (UUIDs / PKs / paths
    replaced with stable placeholders). Suitable for ``data_regression``.
    """
    from aiida_workgraph import WorkGraph

    def _serialize(workgraph: WorkGraph) -> dict[str, Any]:
        raw = workgraph.to_dict()
        task_names = sorted(workgraph.get_task_names())
        return {
            "workgraph_name": workgraph.name,
            "task_names": task_names,
            "n_tasks": len(task_names),
            "raw": _scrub(raw),
        }

    return _serialize
