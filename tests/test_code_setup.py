"""Tests for AiiDA code registration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class TestExecutableCoverage:
    """Every code the dispatcher can load must be registrable."""

    def test_dispatched_codes_have_executable_entries(self) -> None:
        """Each ``_load_code`` executable is a key in ``QE_EXECUTABLES``."""
        import koopmans.aiida.workflows as workflows
        from koopmans.aiida.setup.codes import QE_EXECUTABLES

        source = Path(workflows.__file__).read_text()
        executables = set(re.findall(r'_load_code\(\s*"[^"]+"\s*,\s*"([^"]+)"', source))

        assert executables, "regex matched no _load_code call sites"
        missing = executables - set(QE_EXECUTABLES)
        assert not missing, f"dispatcher loads {missing} with no QE_EXECUTABLES entry"


class TestPseudoNameNonFatal:
    """The decompose pseudo-name must not break the PATH scan or install."""

    def test_pseudo_name_maps_to_decompose_plugin(self) -> None:
        """The pseudo-name is registered against the decompose CalcJob."""
        from koopmans.aiida.setup.codes import QE_EXECUTABLES

        assert QE_EXECUTABLES["pw2wannier90_decompose.x"] == "koopmans.pw2wannier_decompose"

    def test_missing_pseudo_name_scans_as_non_fatal(self) -> None:
        """An unresolved pseudo-name lands in ``missing`` without registering."""
        from koopmans.aiida.setup.codes import scan_and_register_codes

        # No explicit override and no such binary on PATH: the scan must route
        # it to ``missing`` without ever calling ``setup_code`` (so ``computer``
        # is never touched) and without raising.
        found, missing = scan_and_register_codes(
            {"pw2wannier90_decompose.x": "koopmans.pw2wannier_decompose"},
            computer=None,  # never reached: nothing is found to register
        )

        assert found == []
        assert missing == ["pw2wannier90_decompose.x"]

    def test_missing_pseudo_name_not_flagged_essential(self, capsys: Any) -> None:
        """The summary treats only pw.x as essential, so the pseudo-name is OK."""
        from koopmans.aiida.setup.codes import print_setup_summary

        print_setup_summary(
            existing_codes=["pw.x"],
            found_codes=[],
            missing_codes=["pw2wannier90_decompose.x", "wann2kcp.x", "merge_evc.x"],
        )

        out = capsys.readouterr().out
        assert "All essential executables found" in out
        assert "Essential executable(s) not found" not in out


class TestForcedCodeReinstall:
    """Forced reinstalls must not collide on the retired ``<label>_old`` label."""

    def test_repeated_force_reinstall_uniquifies_old_labels(
        self, aiida_profile_clean: Any, aiida_localhost: Any, tmp_path: Any
    ) -> None:
        """Two forced reinstalls retire two codes with distinct labels."""
        from aiida import orm

        from koopmans.aiida.setup.codes import setup_code

        exe = tmp_path / "pw.x"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)

        for _ in range(3):
            setup_code("pw.x", str(exe), "quantumespresso.pw", aiida_localhost, force=True)

        labels = {
            code.label
            for (code,) in orm.QueryBuilder().append(orm.InstalledCode).iterall()
            if code.label.startswith("pw")
        }
        assert labels == {"pw", "pw_old", "pw_old2"}
