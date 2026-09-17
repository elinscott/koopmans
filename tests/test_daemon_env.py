"""Unit tests for the daemon-launch environment setup."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest


def test_ensure_daemon_env_resolves_verdi_to_own_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After the fix, ``verdi`` resolves under this interpreter's own bin dir.

    Strips the interpreter's bin dir from ``PATH`` first, so a caller
    launching ``koopmans`` by full path from a shell without the venv
    activated reproduces the reported failure mode.
    """
    from koopmans.aiida.setup import daemon as daemon_mod
    from koopmans.aiida.setup import hq as hq_mod
    from koopmans.aiida.setup import profile as profile_mod

    interpreter_bin_dir = Path(sys.executable).parent
    verdi_path = interpreter_bin_dir / "verdi"
    if not verdi_path.is_file():
        pytest.skip("test interpreter has no sibling verdi executable")

    monkeypatch.setattr(profile_mod, "koopmans_dir", lambda: tmp_path)
    monkeypatch.setattr(hq_mod, "koopmans_dir", lambda: tmp_path)

    stripped_path = os.pathsep.join(
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry not in (str(interpreter_bin_dir), str(tmp_path / "bin"))
    )
    monkeypatch.setenv("PATH", stripped_path)
    monkeypatch.setenv("HQ_SERVER_DIR", "sentinel")
    monkeypatch.setenv("SQLITE_TMPDIR", "sentinel")

    daemon_mod._ensure_daemon_env()

    resolved = shutil.which("verdi")
    assert resolved is not None
    assert Path(resolved) == verdi_path


def test_ensure_daemon_env_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Calling ``_ensure_daemon_env`` twice leaves one copy of each dir at the front."""
    from koopmans.aiida.setup import daemon as daemon_mod
    from koopmans.aiida.setup import hq as hq_mod
    from koopmans.aiida.setup import profile as profile_mod

    monkeypatch.setattr(profile_mod, "koopmans_dir", lambda: tmp_path)
    monkeypatch.setattr(hq_mod, "koopmans_dir", lambda: tmp_path)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setenv("HQ_SERVER_DIR", "sentinel")
    monkeypatch.setenv("SQLITE_TMPDIR", "sentinel")

    daemon_mod._ensure_daemon_env()
    path_after_first_call = os.environ["PATH"]
    daemon_mod._ensure_daemon_env()
    path_after_second_call = os.environ["PATH"]

    assert path_after_second_call == path_after_first_call
    interpreter_bin_dir = str(Path(sys.executable).parent)
    entries = path_after_second_call.split(os.pathsep)
    assert entries.count(interpreter_bin_dir) == 1
    assert entries.count(str(tmp_path / "bin")) == 1


def test_ensure_daemon_env_exports_sqlite_tmpdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The daemon environment points SQLITE_TMPDIR at a created koopmans dir."""
    from koopmans.aiida.setup import daemon as daemon_mod
    from koopmans.aiida.setup import hq as hq_mod
    from koopmans.aiida.setup import profile as profile_mod

    monkeypatch.setattr(profile_mod, "koopmans_dir", lambda: tmp_path)
    monkeypatch.setattr(hq_mod, "koopmans_dir", lambda: tmp_path)
    # Register PATH and the exported variables with monkeypatch so the
    # function's os.environ writes are undone at teardown.
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setenv("HQ_SERVER_DIR", "sentinel")
    monkeypatch.setenv("SQLITE_TMPDIR", "sentinel")

    daemon_mod._ensure_daemon_env()

    sqlite_tmpdir = tmp_path / "sqlite-tmp"
    assert os.environ["SQLITE_TMPDIR"] == str(sqlite_tmpdir)
    assert sqlite_tmpdir.is_dir()
    assert os.environ["HQ_SERVER_DIR"] == str(tmp_path / "hq-server-dir")
    assert str(tmp_path / "bin") in os.environ["PATH"].split(os.pathsep)
