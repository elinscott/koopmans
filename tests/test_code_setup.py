"""Tests for AiiDA code registration."""

from __future__ import annotations

from typing import Any


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
