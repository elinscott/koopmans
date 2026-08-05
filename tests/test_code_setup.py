"""Tests for AiiDA code registration."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Probe output recorded from real binaries on a Debian box.

# ``nm -D`` on an Intel-MPI Quantum ESPRESSO pw.x.
NM_MPI = """\
                 U mpi_abort_
                 U mpi_allreduce_
                 U mpi_init_
                 U mpi_initialized_
"""

# ``nm -D`` on a serial wannier90.x: no MPI entry at all.
NM_SERIAL = """\
0000000000004140 B __command_line_options_MOD_command_line
0000000000004040 B __command_line_options_MOD_input_file_
"""

# ``ldd`` on an OpenMPI Quantum ESPRESSO pw.x.
LDD_MPI = """\
\tlibmpi_mpifh.so.40 => /lib/x86_64-linux-gnu/libmpi_mpifh.so.40 (0x000077c29429d000)
\tlibmpi.so.40 => /lib/x86_64-linux-gnu/libmpi.so.40 (0x000077c291679000)
\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x000071272e600000)
"""

# ``ldd`` on a serial wannier90.x.
LDD_SERIAL = """\
\tlinux-vdso.so.1 (0x00007ffc355c0000)
\tlibwannier90.so.4 => /home/user/wannier90/build/libwannier90.so.4 (0x000071272ee00000)
\tlibgfortran.so.5 => /lib/x86_64-linux-gnu/libgfortran.so.5 (0x000071272ea00000)
\tlibopenblas.so.0 => /lib/x86_64-linux-gnu/libopenblas.so.0 (0x000071272c1b0000)
"""

# ``ldd`` on a serial binary built against libraries whose *names and paths*
# spell "mpi" while no MPI runtime is linked.
LDD_MPI_SUBSTRING_ONLY = """\
\tlibopenblas.so.0 => /usr/lib/x86_64-linux-gnu/openmpi/lib/libopenblas.so.0 (0x000071272c1b0)
\tlibscalapack-openmpi.so.2.1 => /usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so.2.1 (0x7f00)
"""


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


class TestMpiEvidence:
    """``declares_mpi`` promotes a binary only on evidence of MPI itself."""

    def test_mpi_init_symbol_is_evidence(self) -> None:
        """A binary whose dynamic symbols name mpi_init_ is MPI-capable."""
        from koopmans.aiida.setup.codes import declares_mpi, mpi_evidence

        assert declares_mpi(NM_MPI, LDD_SERIAL, "")
        assert mpi_evidence(NM_MPI, LDD_SERIAL, "") == "declares mpi_init_"

    def test_linked_mpi_runtime_is_evidence(self) -> None:
        """A binary calling MPI from a shared library is caught by its sonames.

        QE's pw.x exports no MPI symbol of its own — the calls live in the
        libraries it links — so the symbol probe alone would miss it.
        """
        from koopmans.aiida.setup.codes import mpi_evidence

        assert mpi_evidence(NM_SERIAL, LDD_MPI, "") == "links libmpi_mpifh.so.40"

    def test_strings_catch_a_stripped_static_build(self) -> None:
        """A statically linked MPI with no symbols left is caught by its strings."""
        from koopmans.aiida.setup.codes import mpi_evidence

        evidence = mpi_evidence(NM_SERIAL, LDD_SERIAL, "MPI_Init\nMPI_Comm_rank\n")
        assert evidence == "contains the MPI_Init string"

    def test_mpi_substring_alone_is_not_evidence(self) -> None:
        """An OpenBLAS or ScaLAPACK name spelling "mpi" does not promote a binary."""
        from koopmans.aiida.setup.codes import declares_mpi

        assert not declares_mpi(NM_SERIAL, LDD_MPI_SUBSTRING_ONLY, "openmpi/lib/libopenblas.so")

    def test_serial_binary_stays_serial(self) -> None:
        """A binary with no MPI evidence anywhere is serial."""
        from koopmans.aiida.setup.codes import declares_mpi

        assert not declares_mpi(NM_SERIAL, LDD_SERIAL, "")

    def test_no_evidence_at_all_is_serial(self) -> None:
        """A probe that collected nothing resolves to serial rather than raising."""
        from koopmans.aiida.setup.codes import declares_mpi

        assert not declares_mpi("", "", "")

    def test_unreadable_binary_collects_nothing(self, tmp_path: Path) -> None:
        """A path that is not a file yields empty evidence instead of an error."""
        from koopmans.aiida.setup.codes import collect_mpi_evidence, declares_mpi

        evidence = collect_mpi_evidence(str(tmp_path / "does_not_exist.x"))
        assert evidence == ("", "", "")
        assert not declares_mpi(*evidence)


class TestMpiDecision:
    """Precedence between the always-serial list, the overrides and the probe."""

    def test_serial_codes_beat_a_positive_probe(self, monkeypatch: Any) -> None:
        """wann2kcp stays serial even when its binary declares MPI."""
        from koopmans.aiida.setup import codes

        monkeypatch.setattr(codes, "collect_mpi_evidence", lambda path: (NM_MPI, LDD_MPI, ""))

        decision = codes.decide_with_mpi("wann2kcp", "/somewhere/wann2kcp.x")
        assert decision.with_mpi is False
        assert decision.reason == "always serial: races on its buffer scratch"

    def test_explicit_serial_beats_a_positive_probe(self, monkeypatch: Any) -> None:
        """--serial overrules a binary that declares MPI."""
        from koopmans.aiida.setup import codes

        monkeypatch.setattr(codes, "collect_mpi_evidence", lambda path: (NM_MPI, LDD_MPI, ""))

        decision = codes.decide_with_mpi("pw", "/somewhere/pw.x", serial_labels=["pw"])
        assert decision.with_mpi is False
        assert decision.reason == "requested by --serial"

    def test_explicit_parallel_beats_the_always_serial_list(self, monkeypatch: Any) -> None:
        """--parallel overrules both the probe and SERIAL_CODES."""
        from koopmans.aiida.setup import codes

        monkeypatch.setattr(codes, "collect_mpi_evidence", lambda path: ("", "", ""))

        decision = codes.decide_with_mpi(
            "wann2kcp", "/somewhere/wann2kcp.x", parallel_labels=["wann2kcp"]
        )
        assert decision.with_mpi is True
        assert decision.reason == "requested by --parallel"


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


class TestMpiFlagMigration:
    """Codes registered with the wrong flag are replaced, not left in place."""

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
