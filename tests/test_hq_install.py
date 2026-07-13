"""Tests for the HyperQueue install + server/worker lifecycle.

These tests exercise the koopmans-managed HQ binary install and the
server/worker process helpers in :mod:`koopmans.aiida.setup.hq`. We
mock the network download so the suite stays self-contained.

The lifecycle test is gated on the actual ``hq`` binary being installed
(via the bundled installer). It boots a real HQ server + worker, checks
they appear in :func:`is_hq_server_running` / :func:`is_hq_worker_running`,
and tears them down. Skipped on non-Linux platforms.
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import tarfile
from pathlib import Path
from typing import Any, Literal

import pytest

pytestmark = pytest.mark.usefixtures("aiida_profile")


def _fake_hq_binary_archive() -> tuple[bytes, str]:
    """Build a minimal tar.gz containing a stub ``hq`` shell script.

    Returns the archive bytes + its sha256 hex digest. The stub binary
    just exits 0 — fine for ``install_hq_binary`` which only checks that
    the file is extracted and the digest matches.
    """
    stub = b"#!/bin/sh\nexit 0\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="hq-v0.0.0-linux-x64/hq")
        info.size = len(stub)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(stub))
    archive = buf.getvalue()
    return archive, hashlib.sha256(archive).hexdigest()


@pytest.fixture
def patched_hq(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Patch the koopmans-managed dir + the HQ download to use a local stub."""
    from koopmans.aiida.setup import hq as hq_mod
    from koopmans.aiida.setup import profile as profile_mod

    # Redirect the koopmans-managed dir into the tmp path so we don't
    # pollute the real AiiDA config dir with test fixtures.
    monkeypatch.setattr(profile_mod, "koopmans_dir", lambda: tmp_path)
    monkeypatch.setattr(hq_mod, "koopmans_dir", lambda: tmp_path)

    archive_bytes, digest = _fake_hq_binary_archive()
    monkeypatch.setattr(hq_mod, "HQ_ARCHIVE_SHA256", digest)

    class _FakeResponse:
        def __init__(self, data: bytes) -> None:
            """Wrap the canned archive bytes."""
            self._data = data

        def read(self) -> bytes:
            """Return the canned archive bytes."""
            return self._data

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> Literal[False]:
            return False

    def fake_urlopen(url: str) -> _FakeResponse:
        """Serve the canned archive instead of downloading."""
        return _FakeResponse(archive_bytes)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.delenv("KOOPMANS_HQ_BINARY", raising=False)
    return tmp_path


@pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"},
    reason="install_hq_binary only fetches a Linux x86_64 prebuilt",
)
def test_install_hq_binary_downloads_and_extracts(patched_hq: Path) -> None:
    """``install_hq_binary`` writes an executable ``hq`` to the bundled path."""
    from koopmans.aiida.setup.hq import hq_bin_path, install_hq_binary

    bin_path = install_hq_binary()
    assert bin_path is not None
    assert bin_path == hq_bin_path()
    assert bin_path.is_file()
    assert os.access(bin_path, os.X_OK)


@pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"},
    reason="install_hq_binary only fetches a Linux x86_64 prebuilt",
)
def test_install_hq_binary_is_idempotent(patched_hq: Path) -> None:
    """Re-running ``install_hq_binary`` reuses the cached binary."""
    from koopmans.aiida.setup.hq import install_hq_binary

    first = install_hq_binary()
    assert first is not None
    mtime = first.stat().st_mtime

    second = install_hq_binary()
    assert second == first
    assert second.stat().st_mtime == mtime


def test_install_hq_binary_fails_on_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mismatched checksum aborts install and raises ``ClickException``."""
    import click

    from koopmans.aiida.setup import hq as hq_mod
    from koopmans.aiida.setup import profile as profile_mod

    monkeypatch.setattr(profile_mod, "koopmans_dir", lambda: tmp_path)
    monkeypatch.setattr(hq_mod, "koopmans_dir", lambda: tmp_path)
    monkeypatch.setattr(hq_mod, "HQ_ARCHIVE_SHA256", "0" * 64)
    monkeypatch.delenv("KOOPMANS_HQ_BINARY", raising=False)

    archive_bytes, _ = _fake_hq_binary_archive()

    class _FakeResponse:
        def __init__(self, data: bytes) -> None:
            """Wrap the canned archive bytes."""
            self._data = data

        def read(self) -> bytes:
            """Return the canned archive bytes."""
            return self._data

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> Literal[False]:
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda url: _FakeResponse(archive_bytes))

    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        pytest.skip("checksum check is only reached on linux-x64")

    with pytest.raises(click.ClickException, match="checksum mismatch"):
        hq_mod.install_hq_binary()


@pytest.mark.slow
def test_hq_lifecycle_with_real_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """End-to-end smoke test: download real HQ, start server + worker, stop both.

    Marked slow because it hits the internet and spins up two child
    processes. Skipped on non-Linux platforms.
    """
    from koopmans.aiida.setup import hq as hq_mod
    from koopmans.aiida.setup import profile as profile_mod

    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        pytest.skip("HQ prebuilt only available on linux-x64")

    # Real download into a tmp koopmans-managed dir.
    monkeypatch.setattr(profile_mod, "koopmans_dir", lambda: tmp_path)
    monkeypatch.setattr(hq_mod, "koopmans_dir", lambda: tmp_path)
    monkeypatch.delenv("KOOPMANS_HQ_BINARY", raising=False)

    bin_path = hq_mod.install_hq_binary()
    if bin_path is None:
        pytest.skip("HQ download failed (likely no network)")

    assert hq_mod.start_hq_server(wait=True)
    try:
        assert hq_mod.is_hq_server_running()
        assert hq_mod.start_hq_worker(wait=True, resources=1)
        assert hq_mod.is_hq_worker_running()
    finally:
        hq_mod.stop_hq()
    assert not hq_mod.is_hq_server_running()
    assert not hq_mod.is_hq_worker_running()


def test_get_localhost_computer_uses_hyperqueue(
    monkeypatch: pytest.MonkeyPatch, aiida_profile_clean: Any
) -> None:
    """``get_localhost_computer`` always registers with the ``hyperqueue`` scheduler."""
    from koopmans.aiida.setup import computer as computer_mod

    computer = computer_mod.get_localhost_computer(nprocs=1)
    assert computer.scheduler_type == "hyperqueue"
