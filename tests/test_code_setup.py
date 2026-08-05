"""Tests for AiiDA code registration."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures import (
    LIBS_PARALLEL_HDF5,
    LIBS_QE,
    LIBS_SERIAL,
    LIBS_SERIAL_LINKING_MPI,
    NM_CALLS_MPI_INIT,
    NM_MPI_INITIALIZED_ONLY,
    NM_NO_MPI_CALL,
    NM_SERIAL,
)

# ``ldd`` output, kept here because only the parser tests read it.
LDD_OUTPUT = """\
\tlinux-vdso.so.1 (0x00007ffc355c0000)
\tlibmpi_mpifh.so.40 => /lib/x86_64-linux-gnu/libmpi_mpifh.so.40 (0x000077c29429d000)
\tlibmkl_core.so.2 => not found
\t/lib64/ld-linux-x86-64.so.2 (0x000077c2942c1000)
"""

# The executables this developer's box carries, and the verdict each must
# get. Every path is skipped when absent, so the check is a no-op on CI.
REAL_BINARIES = {
    "/usr/local/bin/wannier90.x": False,
    "/home/linsco_e/code/wannier90/build/wannier90.x": False,
    "/home/linsco_e/code/q-e/build/bin/kcw.x": True,
    "/home/linsco_e/code/q-e/build/bin/pw.x": True,
}


def _workflows_source() -> str:
    """Concatenate the workflows package's module sources for call-site scans."""
    import koopmans.aiida.workflows as workflows

    return "".join(
        path.read_text() for path in sorted(Path(workflows.__file__).parent.glob("*.py"))
    )


class TestExecutableCoverage:
    """Every code the dispatcher can load must be registrable."""

    def test_dispatched_codes_have_executable_entries(self) -> None:
        """Each ``load_code`` executable is a key in ``QE_EXECUTABLES``."""
        from koopmans.aiida.setup.codes import QE_EXECUTABLES

        source = _workflows_source()
        executables = set(re.findall(r'(?<!\.)\bload_code\(\s*"[^"]+"\s*,\s*"([^"]+)"', source))

        assert executables, "regex matched no load_code call sites"
        # Codes registered outside the QE install scan (their second argument
        # is a human-readable name, not an executable on PATH).
        non_qe = {name for name in executables if not name.endswith(".x")}
        assert non_qe <= {"julia (Wannier.jl)"}, f"unexpected non-QE load sites: {non_qe}"
        missing = (executables - non_qe) - set(QE_EXECUTABLES)
        assert not missing, f"dispatcher loads {missing} with no QE_EXECUTABLES entry"

    def test_dispatched_labels_are_registered(self) -> None:
        """Each ``load_code`` label is a label the installer registers."""
        from koopmans.aiida.setup.codes import code_specs

        source = _workflows_source()
        labels = set(re.findall(r'(?<!\.)\bload_code\(\s*"([^"]+)"\s*,\s*"[^"]+"', source))

        assert labels, "regex matched no load_code call sites"
        # ``wannierjl`` is registered by its own plugin helper, not the QE scan.
        missing = labels - set(code_specs()) - {"wannierjl"}
        assert not missing, f"dispatcher loads {missing} with no registration entry"

    def test_no_non_literal_load_code_calls(self) -> None:
        """Every ``load_code`` call passes two string literals.

        The coverage test above only sees literal arguments; a call built
        from variables would escape it and could load an unregistered code.
        """
        source = _workflows_source()
        all_calls = len(re.findall(r"(?<!\.)\bload_code\((?!\s*self)", source))
        literal_calls = len(re.findall(r'load_code\(\s*"[^"]+"\s*,\s*"[^"]+"', source))
        definitions = len(re.findall(r"def load_code\(", source))

        assert all_calls - definitions == literal_calls, (
            "a load_code call site uses non-literal arguments and escapes "
            "the executable-coverage test"
        )

    def test_load_code_labels_match_executables(self) -> None:
        """Each ``load_code`` pair matches how that label is registered.

        Registration pairs a label with an executable, so a call site that
        names a different executable for a label would pass the coverage
        test yet load a code built from the wrong binary.
        """
        from koopmans.aiida.setup.codes import code_specs

        source = _workflows_source()
        pairs = re.findall(r'(?<!\.)\bload_code\(\s*"([^"]+)"\s*,\s*"([^"]+)"', source)
        specs = code_specs()

        assert pairs, "regex matched no load_code call sites"
        mismatched = [
            (label, executable)
            for label, executable in pairs
            if executable.endswith(".x") and specs.get(label, (None,))[0] != executable
        ]
        assert not mismatched, f"label/executable mismatch at call sites: {mismatched}"


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


class TestRealBinaries:
    """The decision made about the executables actually installed on this box."""

    @pytest.mark.parametrize(("path", "expected"), sorted(REAL_BINARIES.items()))
    def test_installed_binary_gets_the_expected_verdict(self, path: str, expected: bool) -> None:
        """Each reference build is judged the way its source says it behaves."""
        from koopmans.aiida.setup.codes import decide_with_mpi

        if not Path(path).is_file():
            pytest.skip(f"{path} is not installed here")

        decision = decide_with_mpi("probe", path)
        assert decision.with_mpi is expected, decision.reason


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
