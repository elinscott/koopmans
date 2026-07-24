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

    def test_no_non_literal_load_code_calls(self) -> None:
        """Every ``_load_code`` call passes two string literals.

        The coverage test above only sees literal arguments; a call built
        from variables would escape it and could load an unregistered code.
        """
        import koopmans.aiida.workflows as workflows

        source = Path(workflows.__file__).read_text()
        all_calls = len(re.findall(r"_load_code\((?!\s*self)", source))
        literal_calls = len(re.findall(r'_load_code\(\s*"[^"]+"\s*,\s*"[^"]+"', source))
        definitions = len(re.findall(r"def _load_code\(", source))

        assert all_calls - definitions == literal_calls, (
            "a _load_code call site uses non-literal arguments and escapes "
            "the executable-coverage test"
        )

    def test_load_code_labels_match_executables(self) -> None:
        """Each ``_load_code`` label equals its executable minus ``.x``.

        Registration derives the code label from the executable name, so a
        mismatched pair would pass the coverage test yet fail at runtime.
        """
        import koopmans.aiida.workflows as workflows

        source = Path(workflows.__file__).read_text()
        pairs = re.findall(r'_load_code\(\s*"([^"]+)"\s*,\s*"([^"]+)"', source)

        assert pairs, "regex matched no _load_code call sites"
        mismatched = [(n, e) for n, e in pairs if e != f"{n}.x"]
        assert not mismatched, f"label/executable mismatch at call sites: {mismatched}"


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
