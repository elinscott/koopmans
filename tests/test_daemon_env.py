"""Unit tests for the daemon-launch environment setup."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


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
