"""Tests for AiiDA code registration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures import (
    LDD_SERIAL_LINKING_OPENMPI,
    LIBS_PARALLEL_HDF5,
    LIBS_QE,
    LIBS_SERIAL,
    LIBS_SERIAL_LINKING_MPI,
    NM_CALLS_MPI_INIT,
    NM_MPI_INITIALIZED_ONLY,
    NM_NO_MPI_CALL,
    NM_SERIAL,
    NM_SERIAL_MPIF90,
    SERIAL_OPENMPI_CLOSURE,
)

# ``ldd`` output, kept here because only the parser tests read it.
LDD_OUTPUT = """\
\tlinux-vdso.so.1 (0x00007ffc355c0000)
\tlibmpi_mpifh.so.40 => /lib/x86_64-linux-gnu/libmpi_mpifh.so.40 (0x000077c29429d000)
\tlibmkl_core.so.2 => not found
\t/lib64/ld-linux-x86-64.so.2 (0x000077c2942c1000)
"""


class TestExecutableCoverage:
    """Every code the dispatcher can load must be registrable."""

    def test_dispatchable_names_are_registered(self) -> None:
        """The dispatcher's vocabulary is the labels the installer registers.

        Every route names its codes through this vocabulary, so a name it
        does not hold cannot reach a profile lookup at all.
        """
        from koopmans.aiida.setup.codes import code_specs
        from koopmans.aiida.workflows import code_executables

        specs = code_specs()
        # ``wannierjl`` is registered by its own plugin helper, not the QE scan.
        assert set(code_executables()) == set(specs) | {"wannierjl"}
        assert all(
            executable == specs[name][0]
            for name, executable in code_executables().items()
            if name in specs
        )

    def test_an_unregistered_name_is_rejected(self) -> None:
        """Loading a name outside the vocabulary says so, rather than searching.

        Nothing but the dispatcher's own source names a code, so a name no
        installer registers is a typo, and a profile lookup would report it as
        a missing installation.
        """
        from koopmans.aiida.workflows import load_code
        from koopmans.input_file.workflow import Task

        with pytest.raises(ValueError, match=r"'pw90' is not a code koopmans registers"):
            load_code("pw90", Task.DFT_BANDS)


def _input_for(task: str, **workflow: Any) -> Any:
    """Return a minimal silicon input for ``task``, for the declaration pins."""
    from koopmans.input_file import KoopmansInput

    return KoopmansInput.model_validate(
        {
            "workflow": {"task": task, "pseudo_library": "SG15/1.2/PBE/SR", **workflow},
            "atoms": {
                "cell_parameters": {"periodic": True, "ibrav": 2, "celldms": {"1": 10.2622}},
                "atomic_positions": {
                    "units": "crystal",
                    "positions": [["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25]],
                },
            },
            "kpoints": {"grid": [2, 2, 2], "offset": [0, 0, 0]},
            "calculator_parameters": {"ecutwfc": 20.0, "nbnd": 8},
        }
    )


class TestRequiredCodes:
    """Each route states the whole code set its chain runs, up front.

    The sets are spelt out here rather than read off the declarations, so
    that dropping a name from one fails this test.
    """

    @pytest.mark.parametrize(
        ("task", "workflow", "expected"),
        [
            ("dft_bands", {}, {"pw"}),
            ("dft_eps", {}, {"pw", "ph"}),
            ("wannierize", {}, {"pw", "pw2wannier90", "wannier90", "projwfc"}),
            (
                "wannierize",
                {"block_wannierization_threshold": 0.1},
                {"pw", "pw2wannier90", "wannier90", "projwfc", "wannierjl"},
            ),
            (
                "singlepoint",
                {"screening_method": "dscf"},
                {"pw", "kcp", "wannier90", "pw2wannier90", "wann2kcp", "merge_evc"},
            ),
            (
                "singlepoint",
                {"screening_method": "dfpt"},
                {"pw", "kcw", "wannier90", "pw2wannier90", "ph"},
            ),
            (
                "trajectory",
                {},
                {"pw", "kcp", "wannier90", "pw2wannier90", "wann2kcp", "merge_evc"},
            ),
        ],
    )
    def test_route_states_its_whole_chain(
        self, task: str, workflow: dict[str, Any], expected: set[str]
    ) -> None:
        """The declared set is exactly the codes the chain can reach."""
        from koopmans.aiida.workflows import route_for
        from koopmans.input_file.workflow import Task

        koopmans_input = _input_for(task, **workflow)
        route = route_for(Task(task))
        assert set(route.required_codes(koopmans_input)) == expected

    def test_only_a_split_run_asks_for_the_julia_code(self) -> None:
        """The threshold is what adds Wannier.jl, and it is all that adds it.

        Both halves are needed. Without the with-threshold case a
        declaration that never names the julia code would pass; without the
        without-threshold case one that always names it would, and every
        Wannierize user would need a Julia install.
        """
        from koopmans.aiida.workflows import route_for
        from koopmans.input_file.workflow import Task

        route = route_for(Task.WANNIERIZE)
        plain = route.required_codes(_input_for("wannierize"))
        split = route.required_codes(_input_for("wannierize", block_wannierization_threshold=0.1))
        assert "wannierjl" not in plain
        assert "wannierjl" in split
        assert set(split) - set(plain) == {"wannierjl"}

    def test_the_rest_of_the_wannierize_set_ignores_its_flags(self) -> None:
        """Only Wannier.jl moves with the input; the QE codes are unconditional.

        They come out of one build, so narrowing them would let a user get
        part-way through a run before learning one is missing.
        """
        from koopmans.aiida.workflows import route_for
        from koopmans.input_file.workflow import Task

        route = route_for(Task.WANNIERIZE)
        plain = route.required_codes(_input_for("wannierize"))
        automatic = route.required_codes(_input_for("wannierize", auto_projections=True))
        assert plain == automatic

    def test_missing_code_names_the_task_and_the_fix(
        self, aiida_profile_clean: Any, localhost_computer: Any
    ) -> None:
        """A code no profile holds is reported before anything is built."""
        from koopmans.aiida.workflows import load_codes_for_task

        with pytest.raises(ValueError) as excinfo:
            load_codes_for_task(_input_for("dft_eps"))
        assert str(excinfo.value) == (
            "The `dft_eps` task runs pw.x, but this AiiDA profile has no `pw@localhost` "
            "code. Run `koopmans install` to register every koopmans code this machine has."
        )

    def test_the_julia_code_points_at_its_own_installer(
        self, aiida_profile_clean: Any, installed_pw_code: Any, localhost_code: Any
    ) -> None:
        """Wannier.jl is not something ``koopmans install`` can find.

        ``koopmans install`` scans for Quantum ESPRESSO binaries, so
        pointing a user at it for the julia code would send them in a
        circle.
        """
        from koopmans.aiida.workflows import load_codes_for_task

        for label, entry_point in (
            ("pw2wannier90", "quantumespresso.pw2wannier90"),
            ("wannier90", "wannier90.wannier90"),
            ("projwfc", "quantumespresso.projwfc"),
        ):
            localhost_code(label, entry_point)
        split = _input_for("wannierize", block_wannierization_threshold=0.1)
        with pytest.raises(ValueError, match=r"setup_julia_environment") as excinfo:
            load_codes_for_task(split)
        assert "`wannierjl@localhost`" in str(excinfo.value)
        assert "koopmans install" not in str(excinfo.value)


class TestMpiSymbolNames:
    """Only MPI's initialization entry points count as an MPI call."""

    @pytest.mark.parametrize(
        "symbol",
        ["MPI_Init", "mpi_init_", "mpi_init__", "MPI_Init_thread", "PMPI_Init", "MPI_Init@@v1"],
    )
    def test_initialization_entry_points_match(self, symbol: str) -> None:
        """Every C, Fortran and profiling spelling of MPI_Init is recognized."""
        from koopmans.aiida.setup.codes import is_mpi_init

        assert is_mpi_init(symbol)

    @pytest.mark.parametrize(
        "symbol",
        ["MPI_Initialized", "mpi_initialized_", "ompi_init_preconnect_mpi", "MPI_Comm_rank"],
    )
    def test_other_mpi_names_do_not_match(self, symbol: str) -> None:
        """MPI_Init is a substring of MPI_Initialized, which starts nothing."""
        from koopmans.aiida.setup.codes import is_mpi_init

        assert not is_mpi_init(symbol)

    def test_only_undefined_symbols_are_read_as_calls(self) -> None:
        """A defined MPI_Init means the file provides MPI, not that it calls it."""
        from koopmans.aiida.setup.codes import calls_mpi_init

        assert calls_mpi_init("0000000000080bb0 T MPI_Init\n") is None
        assert calls_mpi_init("                 U MPI_Init\n") == "MPI_Init"


class TestLddParsing:
    """``ldd`` output becomes the list of libraries there is a file to inspect."""

    def test_unresolved_and_virtual_entries_are_dropped(self) -> None:
        """A "not found" library and linux-vdso have nothing to run nm on."""
        from koopmans.aiida.setup.codes import linked_libraries

        assert linked_libraries(LDD_OUTPUT) == [
            ("libmpi_mpifh.so.40", "/lib/x86_64-linux-gnu/libmpi_mpifh.so.40"),
            ("ld-linux-x86-64.so.2", "/lib64/ld-linux-x86-64.so.2"),
        ]


class TestMpiEvidence:
    """A binary is promoted only where an MPI_Init call can be found."""

    def test_a_call_in_the_binary_is_evidence(self, binary_probe: Callable[..., Any]) -> None:
        """A binary whose own dynamic symbols call mpi_init_ is MPI-capable."""
        from koopmans.aiida.setup.codes import declares_mpi, mpi_evidence

        probe = binary_probe(NM_CALLS_MPI_INIT, LIBS_SERIAL)
        assert declares_mpi(probe)
        assert mpi_evidence(probe) == "calls mpi_init_"

    def test_a_call_in_a_linked_library_is_evidence(self, binary_probe: Callable[..., Any]) -> None:
        """The GNU cmake build of QE calls MPI only from libqe_modules.

        pw.x and kcw.x carry no MPI call of their own, so the executable's
        symbols alone would register the whole of Quantum ESPRESSO as serial.
        """
        from koopmans.aiida.setup.codes import mpi_evidence

        probe = binary_probe(NM_NO_MPI_CALL, LIBS_QE)
        assert mpi_evidence(probe) == "links libqe_modules.so.7, which calls mpi_init_"

    def test_linking_the_runtime_without_calling_it_is_not_evidence(
        self, binary_probe: Callable[..., Any]
    ) -> None:
        """A serial program built with mpif90 links libmpi and stays serial.

        Anything compiled through an MPI wrapper records libmpi in its
        DT_NEEDED list whether or not an MPI call survives, and the runtime's
        own Fortran bindings call PMPI_Init — so neither the linkage nor the
        runtime's symbols may promote the program.
        """
        from koopmans.aiida.setup.codes import declares_mpi

        assert not declares_mpi(binary_probe(NM_NO_MPI_CALL, LIBS_SERIAL_LINKING_MPI))

    def test_parallel_hdf5_is_a_known_false_positive(
        self, binary_probe: Callable[..., Any]
    ) -> None:
        """A serial program linking parallel HDF5 inherits HDF5's MPI_Init.

        HDF5's calls are indistinguishable from the program's own, so this
        reads as parallel. Register such a code with --serial.
        """
        from koopmans.aiida.setup.codes import mpi_evidence

        probe = binary_probe(NM_SERIAL, LIBS_PARALLEL_HDF5)
        assert mpi_evidence(probe) == "links libhdf5.so.1000, which calls MPI_Init"

    def test_asking_whether_mpi_runs_is_not_evidence(
        self, binary_probe: Callable[..., Any]
    ) -> None:
        """MPI_Initialized alone does not start MPI, in the binary or a library."""
        from koopmans.aiida.setup.codes import declares_mpi

        assert not declares_mpi(
            binary_probe(NM_MPI_INITIALIZED_ONLY, {"libfoo.so.1": NM_MPI_INITIALIZED_ONLY})
        )

    def test_strings_catch_a_stripped_static_build(self, binary_probe: Callable[..., Any]) -> None:
        """A statically linked MPI with no symbols left is caught by its strings."""
        from koopmans.aiida.setup.codes import mpi_evidence

        probe = binary_probe(NM_SERIAL, LIBS_SERIAL, "MPI_Comm_rank\nMPI_Init\n")
        assert mpi_evidence(probe) == "contains an MPI_Init symbol name"

    def test_a_banner_mentioning_mpi_init_is_not_evidence(
        self, binary_probe: Callable[..., Any]
    ) -> None:
        """Prose naming MPI_Init is not a symbol table.

        A substring search over ``strings`` promotes any binary whose help
        text or build banner mentions the routine.
        """
        from koopmans.aiida.setup.codes import declares_mpi

        banner = "Configured serial: MPI_Init is never called\n"
        assert not declares_mpi(binary_probe(NM_SERIAL, LIBS_SERIAL, banner))

    def test_serial_binary_stays_serial(self, binary_probe: Callable[..., Any]) -> None:
        """A binary with no MPI evidence anywhere is serial."""
        from koopmans.aiida.setup.codes import declares_mpi

        assert not declares_mpi(binary_probe(NM_SERIAL, LIBS_SERIAL))

    def test_no_evidence_at_all_is_serial(self, binary_probe: Callable[..., Any]) -> None:
        """A probe that collected nothing resolves to serial rather than raising."""
        from koopmans.aiida.setup.codes import declares_mpi

        assert not declares_mpi(binary_probe())

    def test_a_missing_binary_is_not_probed(self, tmp_path: Path, monkeypatch: Any) -> None:
        """No inspection command is run against a path that is not a regular file.

        Without the guard, nm/ldd/strings are spawned three times per missing
        executable — and ``strings -a`` would be pointed at whatever the path
        turns out to be.
        """
        from koopmans.aiida.setup import codes

        commands = []

        def _record(command: list[str], path: str) -> str:
            commands.append(command[0])
            return ""

        monkeypatch.setattr(codes, "_run_probe", _record)

        probe = codes.collect_mpi_evidence(str(tmp_path / "does_not_exist.x"))
        assert commands == []
        assert probe == codes.BinaryProbe("", {}, "")
        assert not codes.declares_mpi(probe)


class TestRuntimeExclusionEndToEnd:
    """A serial binary whose MPI runtime the loader resolves stays serial.

    ``TestMpiEvidence`` feeds ``mpi_evidence`` a hand-built library dict, so it
    never exercises the ldd parse or the skip inside the closure walk. And the
    Intel-MPI wannier90.x on this box cannot stand in: its libmpi.so.12 does
    not resolve, so it reads serial with or without the exclusion.
    """

    def test_the_runtime_is_not_walked(self, replay_probes: Callable[..., Path]) -> None:
        """The OpenMPI libraries are skipped before nm is ever run on them.

        Walking them would find libmpi_mpifh's undefined PMPI_Init entries.
        """
        from koopmans.aiida.setup.codes import collect_mpi_evidence

        exe = replay_probes(NM_SERIAL_MPIF90, LDD_SERIAL_LINKING_OPENMPI, SERIAL_OPENMPI_CLOSURE)
        walked = set(collect_mpi_evidence(str(exe)).library_symbols)

        assert not walked & {
            "libmpi_mpifh.so.40",
            "libmpi.so.40",
            "libopen-pal.so.40",
            "libopen-rte.so.40",
        }
        assert "libhwloc.so.15" in walked, "the non-runtime libraries must still be walked"

    def test_the_binary_is_registered_serial(self, replay_probes: Callable[..., Path]) -> None:
        """The whole decision, from the recorded probes down, comes out serial."""
        from koopmans.aiida.setup.codes import decide_with_mpi

        exe = replay_probes(NM_SERIAL_MPIF90, LDD_SERIAL_LINKING_OPENMPI, SERIAL_OPENMPI_CLOSURE)
        decision = decide_with_mpi("wannier90", str(exe))

        assert decision.with_mpi is False, decision.reason


class TestCompiledBinaries:
    """The whole probe over real ELF files, built from C by the fixture.

    Every other probe test replaces ``_run_probe``, so these are the only
    ones in which ``nm``, ``ldd`` and ``strings`` actually run: they pin the
    command lines and the output formats the parsers read.
    """

    def test_a_binary_with_no_mpi_call_is_serial(self, compiled_binaries: dict[str, Path]) -> None:
        """A binary that mentions MPI nowhere is registered serial."""
        from koopmans.aiida.setup.codes import decide_with_mpi

        decision = decide_with_mpi("probe", str(compiled_binaries["serial.x"]))
        assert decision.with_mpi is False, decision.reason

    def test_a_binary_that_calls_mpi_init_is_parallel(
        self, compiled_binaries: dict[str, Path]
    ) -> None:
        """An undefined MPI_Init in the executable's own symbols is enough."""
        from koopmans.aiida.setup.codes import decide_with_mpi

        decision = decide_with_mpi("probe", str(compiled_binaries["calls_mpi.x"]))
        assert decision.with_mpi is True, decision.reason
        assert decision.reason == "calls MPI_Init"

    def test_a_linked_library_that_calls_mpi_init_is_parallel(
        self, compiled_binaries: dict[str, Path]
    ) -> None:
        """The closure walk finds MPI_Init in a library, as it must for pw.x."""
        from koopmans.aiida.setup.codes import decide_with_mpi

        decision = decide_with_mpi("probe", str(compiled_binaries["library_calls_mpi.x"]))
        assert decision.with_mpi is True, decision.reason
        assert decision.reason == "links libworker.so, which calls MPI_Init"

    def test_linking_only_the_mpi_runtime_is_serial(
        self, compiled_binaries: dict[str, Path]
    ) -> None:
        """A binary whose only MPI_Init caller is an MPI library stays serial.

        This is the wannier90.x case: the runtime is linked, its Fortran
        bindings do call MPI_Init, and the program itself does not. Without
        the runtime exclusion the walk reaches libmpi_mpifh and says parallel.
        """
        from koopmans.aiida.setup.codes import decide_with_mpi

        decision = decide_with_mpi("probe", str(compiled_binaries["runtime_only.x"]))
        assert decision.with_mpi is False, decision.reason


class TestMpiDecision:
    """Precedence between the always-serial list, the overrides and the probe."""

    def test_serial_codes_beat_a_positive_probe(self, monkeypatch: Any) -> None:
        """wann2kcp stays serial even when its binary calls MPI."""
        from koopmans.aiida.setup import codes

        monkeypatch.setattr(
            codes, "collect_mpi_evidence", lambda path: codes.BinaryProbe(NM_CALLS_MPI_INIT, {})
        )

        decision = codes.decide_with_mpi("wann2kcp", "/somewhere/wann2kcp.x")
        assert decision.with_mpi is False
        assert decision.reason == "always serial: races on its buffer scratch"

    def test_explicit_serial_beats_a_positive_probe(self, monkeypatch: Any) -> None:
        """--serial overrules a binary that calls MPI."""
        from koopmans.aiida.setup import codes

        monkeypatch.setattr(
            codes, "collect_mpi_evidence", lambda path: codes.BinaryProbe(NM_CALLS_MPI_INIT, {})
        )

        decision = codes.decide_with_mpi("pw", "/somewhere/pw.x", serial_labels={"pw"})
        assert decision.with_mpi is False
        assert decision.reason == "requested by --serial"

    def test_explicit_parallel_beats_a_negative_probe(self, monkeypatch: Any) -> None:
        """--parallel promotes a binary whose MPI support could not be found."""
        from koopmans.aiida.setup import codes

        monkeypatch.setattr(codes, "collect_mpi_evidence", lambda path: codes.BinaryProbe("", {}))

        decision = codes.decide_with_mpi(
            "wannier90", "/somewhere/wannier90.x", parallel_labels={"wannier90"}
        )
        assert decision.with_mpi is True
        assert decision.reason == "requested by --parallel"

    def test_parallel_on_an_always_serial_code_is_refused(self) -> None:
        """--parallel wann2kcp raises instead of being honoured or dropped.

        SERIAL_CODES records a property of the program, not of the build, so
        there is nothing for an override to overrule.
        """
        from koopmans.aiida.setup import codes

        with pytest.raises(ValueError, match="property of the program") as excinfo:
            codes.decide_with_mpi("wann2kcp", "/somewhere/wann2kcp.x", parallel_labels={"wann2kcp"})
        assert "races on its buffer scratch" in str(excinfo.value)


class TestMpiRegistration:
    """The stored ``with_mpi`` follows what the binary declares."""

    def test_binary_without_mpi_registers_serial(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
    ) -> None:
        """A wannier90.x with no MPI evidence is registered serial.

        Registration used to key on the label alone, so any code outside
        SERIAL_CODES was flagged MPI whatever its build.
        """
        from koopmans.aiida.setup.codes import setup_code

        exe = stub_executable("wannier90.x")
        code = setup_code("wannier90.x", str(exe), "wannier90.wannier90", aiida_localhost)

        assert code is not None
        assert code.with_mpi is False

    def test_explicit_parallel_reaches_the_scan(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
    ) -> None:
        """--parallel registers an evidence-free binary as MPI."""
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import scan_and_register_codes

        exe = stub_executable("wannier90.x")
        found, _, decisions = scan_and_register_codes(
            {"wannier90": ("wannier90.x", "wannier90.wannier90")},
            aiida_localhost,
            explicit_codes={"wannier90": str(exe)},
            parallel_labels=["wannier90"],
        )

        assert found == ["wannier90"]
        assert decisions[0].with_mpi is True
        assert load_code(f"wannier90@{aiida_localhost.label}").with_mpi is True

    def test_overrides_apply_to_every_code_in_one_scan(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
    ) -> None:
        """A generator of override labels is honoured for the second code too.

        Membership is tested once per code, so an argument that can only be
        iterated once loses every override the first test consumed. The
        generator lists the codes in the opposite order to the scan, so
        looking up the first code drains it entirely.
        """
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import scan_and_register_codes

        exe = stub_executable("stub.x")
        found, _, decisions = scan_and_register_codes(
            {
                "pw": ("pw.x", "quantumespresso.pw"),
                "dos": ("dos.x", "quantumespresso.dos"),
            },
            aiida_localhost,
            explicit_codes={"pw": str(exe), "dos": str(exe)},
            parallel_labels=(label for label in ["dos", "pw"]),
        )

        assert sorted(found) == ["dos", "pw"]
        assert all(decision.with_mpi is True for decision in decisions)
        for label in ("pw", "dos"):
            assert load_code(f"{label}@{aiida_localhost.label}").with_mpi is True


class TestEffectiveWithMpi:
    """How a registered code actually runs, given an unset ``with_mpi``."""

    def test_an_unset_flag_falls_through_to_the_plugin_default(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        code_without_mpi_flag: Callable[..., Any],
    ) -> None:
        """The two koopmans CalcJobs that disagree resolve to their own defaults.

        wann2kcp's CalcJob defaults ``withmpi`` to True and merge_evc's to
        False, so an unset flag cannot be read as "serial" for both.
        """
        from koopmans.aiida.setup.codes import effective_with_mpi

        wann2kcp = code_without_mpi_flag("wann2kcp", "koopmans.wann2kcp", aiida_localhost)
        merge_evc = code_without_mpi_flag("merge_evc", "koopmans.merge_evc", aiida_localhost)

        assert effective_with_mpi(wann2kcp) is True
        assert effective_with_mpi(merge_evc) is False

    def test_a_code_with_no_default_plugin_is_undecidable(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        code_without_mpi_flag: Callable[..., Any],
    ) -> None:
        """kcw.x backs three CalcJobs, so its behaviour is not on the node."""
        from koopmans.aiida.setup.codes import effective_with_mpi

        assert effective_with_mpi(code_without_mpi_flag("kcw", None, aiida_localhost)) is None


class TestMpiFlagMigration:
    """Codes that run the wrong way are replaced; the rest keep their cache."""

    def test_an_unset_flag_matching_the_plugin_default_is_left_alone(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        code_without_mpi_flag: Callable[..., Any],
    ) -> None:
        """merge_evc already runs serial through its plugin, so it is not replaced.

        Comparing against the stored value rather than the behaviour would
        replace every code whose flag was never set — on one live profile
        that orphaned ~2000 cached calculations without changing a thing.
        """
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import migrate_code_mpi_flags

        original = code_without_mpi_flag("merge_evc", "koopmans.merge_evc", aiida_localhost)

        assert migrate_code_mpi_flags(["merge_evc"], aiida_localhost) == []
        assert load_code(f"merge_evc@{aiida_localhost.label}").pk == original.pk

    def test_an_unset_flag_contradicting_the_plugin_default_is_replaced(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        code_without_mpi_flag: Callable[..., Any],
    ) -> None:
        """wann2kcp runs under mpirun through its plugin, so it is replaced."""
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import migrate_code_mpi_flags

        code_without_mpi_flag("wann2kcp", "koopmans.wann2kcp", aiida_localhost)

        migrations = migrate_code_mpi_flags(["wann2kcp"], aiida_localhost)

        assert [m.label for m in migrations] == ["wann2kcp"]
        assert load_code(f"wann2kcp@{aiida_localhost.label}").with_mpi is False

    def test_an_undecidable_code_is_left_alone(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        code_without_mpi_flag: Callable[..., Any],
    ) -> None:
        """A code with neither a flag nor a default plugin cannot be shown wrong."""
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import migrate_code_mpi_flags

        original = code_without_mpi_flag("kcw", None, aiida_localhost)

        assert migrate_code_mpi_flags(["kcw"], aiida_localhost) == []
        assert load_code(f"kcw@{aiida_localhost.label}").pk == original.pk

    def test_a_failed_replacement_leaves_the_label_resolvable(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
        monkeypatch: Any,
    ) -> None:
        """If storing the replacement raises, the original keeps its label.

        Retiring first would rename the only node under that label, so
        ``load_code('wannier90@localhost')`` would then raise NotExistent and
        nothing would point at the retired ``wannier90_mpi_pre``.
        """
        from aiida.orm import load_code

        from koopmans.aiida.setup import codes

        exe = stub_executable("wannier90.x")
        original = codes.setup_code(
            "wannier90.x", str(exe), "wannier90.wannier90", aiida_localhost, with_mpi=True
        )
        assert original is not None

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("storage is full")

        monkeypatch.setattr(codes, "_store_code", _boom)

        with pytest.raises(RuntimeError, match="storage is full"):
            codes.migrate_code_mpi_flags(["wannier90"], aiida_localhost)

        assert load_code(f"wannier90@{aiida_localhost.label}").pk == original.pk

    def test_a_failed_retire_leaves_one_node_holding_the_label(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
        monkeypatch: Any,
    ) -> None:
        """If retiring the superseded code raises, the replacement is undone.

        The replacement is stored under the live label before the original is
        renamed off it, so both nodes hold that label in between. Leaving them
        there makes ``load_code`` raise MultipleObjectsError for good, and the
        next install stores a third node under it.
        """
        from aiida.common.exceptions import MultipleObjectsError
        from aiida.orm import load_code

        from koopmans.aiida.setup import codes

        exe = stub_executable("wannier90.x")
        original = codes.setup_code(
            "wannier90.x", str(exe), "wannier90.wannier90", aiida_localhost, with_mpi=True
        )
        assert original is not None

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("the database went away")

        monkeypatch.setattr(codes, "retire_code", _boom)

        with pytest.raises(RuntimeError, match="the database went away"):
            codes.migrate_code_mpi_flags(["wannier90"], aiida_localhost)

        full_label = f"wannier90@{aiida_localhost.label}"
        try:
            survivor = load_code(full_label)
        except MultipleObjectsError as exc:
            raise AssertionError(f"the label was left resolving to two codes: {exc}") from exc
        assert survivor.pk == original.pk

    def test_an_undecidable_code_is_reported(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        code_without_mpi_flag: Callable[..., Any],
        capsys: Any,
    ) -> None:
        """Leaving a code alone because it cannot be read says so.

        kcw stores no with_mpi and has no default plugin, so a --serial/
        --parallel naming it cannot take effect; skipping in silence leaves
        the user believing it did.
        """
        from koopmans.aiida.setup.codes import migrate_code_mpi_flags

        code_without_mpi_flag("kcw", None, aiida_localhost)

        assert migrate_code_mpi_flags(["kcw"], aiida_localhost, serial_labels=["kcw"]) == []

        message = capsys.readouterr().out
        assert "kcw" in message
        assert "--serial kcw" in message

    def test_overrides_apply_to_every_code_in_one_run(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
    ) -> None:
        """A generator of override labels is honoured for the second code too.

        Membership is tested once per code, so an argument that can only be
        iterated once loses every override the first test consumed. The
        generator lists the codes in the opposite order to the migration, so
        looking up the first code drains it entirely.
        """
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import migrate_code_mpi_flags, setup_code

        exe = stub_executable("stub.x")
        for label, plugin in (("pw", "quantumespresso.pw"), ("dos", "quantumespresso.dos")):
            code = setup_code(f"{label}.x", str(exe), plugin, aiida_localhost, label=label)
            assert code is not None

        migrations = migrate_code_mpi_flags(
            ["pw", "dos"], aiida_localhost, parallel_labels=(label for label in ["dos", "pw"])
        )

        assert sorted(m.label for m in migrations) == ["dos", "pw"]
        for label in ("pw", "dos"):
            assert load_code(f"{label}@{aiida_localhost.label}").with_mpi is True

    def test_wrong_flag_is_replaced_under_the_original_label(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
    ) -> None:
        """A serial binary registered as MPI gains a serial replacement."""
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import migrate_code_mpi_flags, setup_code

        exe = stub_executable("wannier90.x")
        stale = setup_code(
            "wannier90.x",
            str(exe),
            "wannier90.wannier90",
            aiida_localhost,
            with_mpi=True,
        )
        assert stale is not None

        migrations = migrate_code_mpi_flags(["wannier90"], aiida_localhost)

        assert [m.label for m in migrations] == ["wannier90"]
        assert migrations[0].retired_label == "wannier90_mpi_pre"
        replacement = load_code(f"wannier90@{aiida_localhost.label}")
        assert replacement.with_mpi is False
        assert replacement.pk != stale.pk
        assert load_code(f"wannier90_mpi_pre@{aiida_localhost.label}").pk == stale.pk

    def test_a_correct_flag_is_left_alone(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
    ) -> None:
        """Rerunning the migration over an already-correct code is a no-op.

        Without this the installer would retire and replace every code on
        every run, orphaning the cache each time.
        """
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import migrate_code_mpi_flags, setup_code

        exe = stub_executable("wannier90.x")
        original = setup_code("wannier90.x", str(exe), "wannier90.wannier90", aiida_localhost)
        assert original is not None

        assert migrate_code_mpi_flags(["wannier90"], aiida_localhost) == []
        assert load_code(f"wannier90@{aiida_localhost.label}").pk == original.pk

    def test_a_vanished_binary_is_left_alone(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
    ) -> None:
        """A code whose executable is gone cannot be probed, so it is untouched."""
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import migrate_code_mpi_flags, setup_code

        exe = stub_executable("wannier90.x")
        original = setup_code(
            "wannier90.x", str(exe), "wannier90.wannier90", aiida_localhost, with_mpi=True
        )
        assert original is not None
        exe.unlink()

        assert migrate_code_mpi_flags(["wannier90"], aiida_localhost) == []
        assert load_code(f"wannier90@{aiida_localhost.label}").pk == original.pk

    def test_replacement_changes_the_code_hash(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
    ) -> None:
        """The replacement hashes differently, so cached results are not reused.

        This is what the migration message warns about; if the two nodes
        hashed alike the warning would be wrong.
        """
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import migrate_code_mpi_flags, setup_code

        exe = stub_executable("wannier90.x")
        stale = setup_code(
            "wannier90.x", str(exe), "wannier90.wannier90", aiida_localhost, with_mpi=True
        )
        assert stale is not None
        stale_hash = stale.base.caching.compute_hash()

        migrate_code_mpi_flags(["wannier90"], aiida_localhost)

        replacement = load_code(f"wannier90@{aiida_localhost.label}")
        assert replacement.base.caching.compute_hash() != stale_hash


class TestDuplicateLabels:
    """Two codes under one label is an error, never an absence."""

    def test_a_duplicated_label_is_not_reported_as_absent(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
    ) -> None:
        """``code_exists`` raises on a duplicate rather than answering False.

        Every caller here treats False as "store one", so swallowing the error
        turns a pair of nodes into three on the next install.
        """
        from aiida.common.exceptions import MultipleObjectsError
        from aiida.orm import InstalledCode

        from koopmans.aiida.setup.codes import code_exists

        for _ in range(2):
            InstalledCode(
                label="wannier90",
                computer=aiida_localhost,
                filepath_executable=str(stub_executable("wannier90.x")),
                default_calc_job_plugin="wannier90.wannier90",
            ).store()

        with pytest.raises(MultipleObjectsError):
            code_exists(f"wannier90@{aiida_localhost.label}")

    def test_an_absent_label_is_still_reported_as_absent(
        self, aiida_profile_clean: Any, aiida_localhost: Any
    ) -> None:
        """Only the duplicate case propagates; a missing code is still False."""
        from koopmans.aiida.setup.codes import code_exists

        assert not code_exists(f"nothing_here@{aiida_localhost.label}")


class TestCodeLabelValidation:
    """``--serial``/``--parallel`` reject a label koopmans never registers."""

    def test_unknown_label_is_rejected(self) -> None:
        """A misspelled code name fails the install rather than being ignored."""
        import click

        from koopmans.cli import _validate_code_labels

        try:
            _validate_code_labels(("wannier",), "--serial")
        except click.BadParameter as exc:
            assert "wannier" in str(exc)
            assert "wannier90" in str(exc)
        else:
            raise AssertionError("an unknown code label was accepted")

    def test_known_labels_pass_through(self) -> None:
        """Registered labels are returned unchanged."""
        from koopmans.cli import _validate_code_labels

        assert _validate_code_labels(("wannier90", "pw"), "--serial") == {"wannier90", "pw"}

    def test_parallel_on_an_always_serial_code_is_rejected(self) -> None:
        """``koopmans install --parallel wann2kcp`` fails at the command line.

        The install would otherwise get as far as registering codes before
        raising, or — as it did — honour the request and race on scratch.
        """
        import click

        from koopmans.cli import _reject_parallel_on_serial_codes

        with pytest.raises(click.BadParameter) as excinfo:
            _reject_parallel_on_serial_codes({"wann2kcp"})
        assert "races on its buffer scratch" in str(excinfo.value)

    def test_parallel_on_an_ordinary_code_is_accepted(self) -> None:
        """Only the always-serial labels are refused."""
        from koopmans.cli import _reject_parallel_on_serial_codes

        _reject_parallel_on_serial_codes({"pw", "wannier90"})

    def test_the_install_command_refuses_before_it_touches_anything(self, monkeypatch: Any) -> None:
        """``koopmans install --parallel wann2kcp`` stops at argument parsing.

        The helper above is tested in isolation, so nothing caught the command
        forgetting to call it — in which case the install would create a
        profile and only then decide what to do about --parallel.
        """
        from click.testing import CliRunner

        from koopmans import cli

        def _too_far(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the install began before --parallel was checked")

        monkeypatch.setattr(cli, "setup_profile", _too_far)

        result = CliRunner().invoke(cli.cli, ["install", "--parallel", "wann2kcp"])

        assert result.exit_code != 0
        assert "races on its buffer scratch" in result.output


class TestMigrateFlag:
    """``koopmans install --no-migrate`` keeps codes that run the wrong way."""

    @staticmethod
    def _stub_install(monkeypatch: Any, computer: Any) -> None:
        """Point ``setup_computers`` at a test computer and skip the PATH scan."""
        from koopmans.aiida.setup import orchestrate

        monkeypatch.setattr(orchestrate, "get_localhost_computer", lambda **kwargs: computer)
        monkeypatch.setattr(
            orchestrate, "scan_and_register_codes", lambda *args, **kwargs: ([], [], [])
        )

    def test_no_migrate_keeps_a_code_that_runs_the_wrong_way(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
        monkeypatch: Any,
    ) -> None:
        """The stale code keeps its node, so its cached calculations stay reusable."""
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import setup_code
        from koopmans.aiida.setup.orchestrate import setup_computers

        self._stub_install(monkeypatch, aiida_localhost)
        exe = stub_executable("wannier90.x")
        stale = setup_code(
            "wannier90.x", str(exe), "wannier90.wannier90", aiida_localhost, with_mpi=True
        )
        assert stale is not None

        setup_computers(migrate=False)

        assert load_code(f"wannier90@{aiida_localhost.label}").pk == stale.pk

    def test_the_default_install_does_migrate(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
        monkeypatch: Any,
    ) -> None:
        """Without --no-migrate the same code is replaced, so the flag decides."""
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import setup_code
        from koopmans.aiida.setup.orchestrate import setup_computers

        self._stub_install(monkeypatch, aiida_localhost)
        exe = stub_executable("wannier90.x")
        stale = setup_code(
            "wannier90.x", str(exe), "wannier90.wannier90", aiida_localhost, with_mpi=True
        )
        assert stale is not None

        setup_computers()

        replacement = load_code(f"wannier90@{aiida_localhost.label}")
        assert replacement.pk != stale.pk
        assert replacement.with_mpi is False


class TestForcedCodeReinstall:
    """Forced reinstalls must not collide on the retired ``<label>_old`` label."""

    def test_repeated_force_reinstall_uniquifies_old_labels(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
    ) -> None:
        """Two forced reinstalls retire two codes with distinct labels."""
        from aiida import orm

        from koopmans.aiida.setup.codes import setup_code

        exe = stub_executable("pw.x")

        for _ in range(3):
            setup_code("pw.x", str(exe), "quantumespresso.pw", aiida_localhost, force=True)

        labels = {
            code.label
            for (code,) in orm.QueryBuilder().append(orm.InstalledCode).iterall()
            if code.label.startswith("pw")
        }
        assert labels == {"pw", "pw_old", "pw_old2"}

    def test_a_failed_reinstall_leaves_the_label_resolvable(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
        monkeypatch: Any,
    ) -> None:
        """A forced reinstall that cannot store its replacement changes nothing."""
        from aiida.orm import load_code

        from koopmans.aiida.setup import codes

        exe = stub_executable("pw.x")
        original = codes.setup_code("pw.x", str(exe), "quantumespresso.pw", aiida_localhost)
        assert original is not None

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("storage is full")

        monkeypatch.setattr(codes, "_store_code", _boom)

        with pytest.raises(RuntimeError, match="storage is full"):
            codes.setup_code("pw.x", str(exe), "quantumespresso.pw", aiida_localhost, force=True)

        assert load_code(f"pw@{aiida_localhost.label}").pk == original.pk


class TestLabelKeyedRegistration:
    """The install scan is keyed by code label, not executable name."""

    def test_registered_labels_split_from_missing(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
    ) -> None:
        """An already-registered label lands in existing, the rest in to-find."""
        from koopmans.aiida.setup.codes import get_codes_to_register, setup_code

        exe = stub_executable("pw.x")
        setup_code("pw.x", str(exe), "quantumespresso.pw", aiida_localhost)

        existing, to_find = get_codes_to_register(aiida_localhost)
        assert existing == ["pw"]
        assert "pw" not in to_find
        assert to_find["pw2wannier90"] == ("pw2wannier90.x", "quantumespresso.pw2wannier90")

    def test_explicit_path_wins_over_the_scan(
        self,
        aiida_profile_clean: Any,
        aiida_localhost: Any,
        stub_executable: Callable[[str], Path],
    ) -> None:
        """An explicit per-label path registers that binary; absent ones report missing."""
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import scan_and_register_codes

        exe = stub_executable("special_pw2wannier90.x")

        found, missing, _ = scan_and_register_codes(
            {
                "pw2wannier90": ("pw2wannier90.x", "quantumespresso.pw2wannier90"),
                "kcp": ("definitely_not_on_path.x", "koopmans.kcp"),
            },
            aiida_localhost,
            explicit_codes={"pw2wannier90": str(exe)},
        )
        assert found == ["pw2wannier90"]
        assert missing == ["kcp"]
        code = load_code(f"pw2wannier90@{aiida_localhost.label}")
        assert str(code.filepath_executable) == str(exe)
