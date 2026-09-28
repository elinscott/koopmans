"""Unit tests for the daemon-launch environment setup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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


def test_start_daemon_reads_the_configured_worker_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aiida_profile: Any
) -> None:
    """``start_daemon`` passes the profile's ``daemon.default_workers``.

    ``DaemonClient.start_daemon`` defaults to one worker on its own; the
    option is only applied by the ``verdi daemon start`` CLI command, which
    koopmans bypasses by calling the client directly.
    """
    from aiida.manage import get_config

    from koopmans.aiida.setup import daemon as daemon_mod
    from koopmans.aiida.setup import hq as hq_mod
    from koopmans.aiida.setup import profile as profile_mod

    monkeypatch.setattr(profile_mod, "koopmans_dir", lambda: tmp_path)
    monkeypatch.setattr(hq_mod, "koopmans_dir", lambda: tmp_path)

    config = get_config()
    config.set_option(  # type: ignore[no-untyped-call]
        "daemon.default_workers", 4, scope=aiida_profile.name
    )

    class _FakeDaemonClient:
        def __init__(self) -> None:
            self.profile = aiida_profile
            self.is_daemon_running = False
            self.start_calls: list[int] = []

        def start_daemon(self, number_workers: int) -> object:
            self.start_calls.append(number_workers)
            return object()

    fake_client = _FakeDaemonClient()
    monkeypatch.setattr("aiida.engine.daemon.client.get_daemon_client", lambda: fake_client)

    assert daemon_mod.start_daemon(wait=False) is True
    assert fake_client.start_calls == [4]
