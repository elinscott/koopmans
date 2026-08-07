"""Tests for the HyperQueue install + server/worker lifecycle.

These tests exercise the koopmans-managed HQ binary install and the
server/worker process helpers in :mod:`koopmans.aiida.setup.hq`. We
mock the network download so the suite stays self-contained.

The worker-lifecycle tests run against ``fake_hq``, a stub ``hq``
executable in a tmp dir that records the argv koopmans builds and serves
a canned ``worker list``. Nothing here touches the live HQ server or the
``koopmans`` profile.

The lifecycle test is gated on the actual ``hq`` binary being installed
(via the bundled installer). It boots a real HQ server + worker, checks
they appear in :func:`is_hq_server_running` / :func:`is_hq_worker_running`,
and tears them down. Skipped on non-Linux platforms.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import tarfile
import time
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
        assert hq_mod.start_hq_worker(wait=True, cpus=1)
        assert hq_mod.is_hq_worker_running()
    finally:
        hq_mod.stop_hq()
    assert not hq_mod.is_hq_server_running()
    assert not hq_mod.is_hq_worker_running()


FAKE_HQ = '''#!/usr/bin/env python3
"""Stand-in for the ``hq`` binary: records its argv, serves a canned worker list."""
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["KOOPMANS_TEST_HQ_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")

workers_path = os.environ["KOOPMANS_TEST_HQ_WORKERS"]
server_path = os.environ["KOOPMANS_TEST_HQ_SERVER"]
ignored_path = os.environ["KOOPMANS_TEST_HQ_IGNORED"]
verb = [arg for arg in args if not arg.startswith("-")][:2]


def server_up():
    with open(server_path) as handle:
        return handle.read().strip() == "1"


def ignored():
    with open(ignored_path) as handle:
        return " ".join(verb) in json.load(handle)


if verb == ["server", "info"]:
    sys.exit(0 if server_up() else 1)
if verb[:1] == ["server"] and verb[1:2] in (["start"], ["stop"]):
    # Workers do not outlive their server: HQ's default on_server_lost is stop.
    with open(server_path, "w") as handle:
        handle.write("1" if verb[1] == "start" else "0")
    with open(workers_path, "w") as handle:
        json.dump([], handle)
    sys.exit(0)

# Every worker subcommand needs a live server, as the real hq does.
if verb[:1] == ["worker"] and not server_up():
    sys.stderr.write("No running instance of HQ found\\n")
    sys.exit(1)

# A command the test asked the stub to swallow: hq accepts it and the worker
# list is unchanged, which is what a worker that dies on startup, or one that
# outlives its stop, looks like from koopmans' side.
if ignored():
    sys.exit(0)

if verb == ["worker", "list"]:
    with open(workers_path) as handle:
        sys.stdout.write(handle.read())
elif verb == ["worker", "stop"]:
    with open(workers_path, "w") as handle:
        json.dump([], handle)
elif verb == ["worker", "start"]:
    cpus = int(args[args.index("--cpus") + 1])
    with open(workers_path) as handle:
        workers = json.load(handle)
    workers.append(
        {
            "id": max([w["id"] for w in workers], default=0) + 1,
            "configuration": {
                "resources": {
                    "resources": [
                        {"kind": "range", "name": "cpus", "start": 0, "end": cpus - 1}
                    ]
                }
            },
        }
    )
    with open(workers_path, "w") as handle:
        json.dump(workers, handle)
sys.exit(0)
'''


class FakeHq:
    """A stub ``hq`` binary plus the state the tests read back from it."""

    def __init__(self, log: Path, workers: Path, server: Path, ignored: Path) -> None:
        """Record where the stub logs its argv and keeps its server state."""
        self._log = log
        self._workers = workers
        self._server = server
        self._ignored = ignored

    def stop_server(self) -> None:
        """Make the stub behave as a machine whose HQ server is down."""
        self._server.write_text("0")

    def ignore(self, *commands: str) -> None:
        """Make the stub accept the given commands and change nothing.

        ``ignore("worker start")`` is a worker that dies before it registers;
        ``ignore("worker stop")`` is one that outlives the stop it was sent.
        """
        self._ignored.write_text(json.dumps(list(commands)))

    @property
    def worker_pools(self) -> list[int]:
        """CPU pool of each worker the stub currently reports."""
        return [
            r["end"] - r["start"] + 1
            for w in json.loads(self._workers.read_text())
            for r in w["configuration"]["resources"]["resources"]
            if r["name"] == "cpus"
        ]

    @property
    def commands(self) -> list[list[str]]:
        """Every argv the stub was invoked with, in order."""
        if not self._log.exists():
            return []
        return [json.loads(line) for line in self._log.read_text().splitlines()]

    def set_workers(self, *cpus: int) -> None:
        """Make the stub report one running worker per given CPU pool."""
        self._workers.write_text(
            json.dumps(
                [
                    {
                        "id": index + 1,
                        "configuration": {
                            "resources": {
                                "resources": [
                                    {"kind": "sum", "name": "mem", "size": 1},
                                    {
                                        "kind": "range",
                                        "name": "cpus",
                                        "start": 0,
                                        "end": pool - 1,
                                    },
                                ]
                            }
                        },
                    }
                    for index, pool in enumerate(cpus)
                ]
            )
        )


@pytest.fixture
def fake_hq(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeHq:
    """Point the HQ helpers at a stub binary and a tmp koopmans-managed dir.

    The stub is a real executable reached through ``subprocess``, so the tests
    exercise the argv koopmans builds and the JSON it parses back, not a
    stand-in for :func:`~koopmans.aiida.setup.hq._run_hq`.
    """
    from koopmans.aiida.setup import hq as hq_mod
    from koopmans.aiida.setup import profile as profile_mod

    monkeypatch.setattr(profile_mod, "koopmans_dir", lambda: tmp_path)
    monkeypatch.setattr(hq_mod, "koopmans_dir", lambda: tmp_path)

    binary = tmp_path / "fake-hq"
    binary.write_text(FAKE_HQ)
    binary.chmod(0o755)
    monkeypatch.setenv("KOOPMANS_HQ_BINARY", str(binary))

    workers = tmp_path / "workers.json"
    workers.write_text("[]")
    server = tmp_path / "server-up"
    server.write_text("1")
    ignored = tmp_path / "ignored.json"
    ignored.write_text("[]")
    monkeypatch.setenv("KOOPMANS_TEST_HQ_WORKERS", str(workers))
    monkeypatch.setenv("KOOPMANS_TEST_HQ_SERVER", str(server))
    monkeypatch.setenv("KOOPMANS_TEST_HQ_IGNORED", str(ignored))
    monkeypatch.setenv("KOOPMANS_TEST_HQ_LOG", str(tmp_path / "argv.log"))
    monkeypatch.delenv("KOOPMANS_MAX_PROCS", raising=False)
    # The poll loops sleep in real time; the stub responds instantly.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    return FakeHq(log=tmp_path / "argv.log", workers=workers, server=server, ignored=ignored)


class TestWorkerDetection:
    """Whether a worker is running is HQ's answer, not a pidfile's."""

    @staticmethod
    def test_worker_koopmans_did_not_spawn_is_reported_running(fake_hq: FakeHq) -> None:
        """A hand-started worker has no pidfile, and must still be seen.

        This is the reported bug: ``koopmans backend status`` printed
        ``✗ hq.worker`` while a worker started with ``hq worker start --cpus 24``
        had been up for hours running jobs.
        """
        from koopmans.aiida.setup.hq import is_hq_worker_running, running_hq_workers

        fake_hq.set_workers(24)

        assert is_hq_worker_running()
        assert [(w.id, w.cpus) for w in running_hq_workers()] == [(1, 24)]

    @staticmethod
    def test_pidfile_without_a_worker_is_reported_not_running(
        fake_hq: FakeHq, tmp_path: Path
    ) -> None:
        """A pidfile naming a live process does not make a worker exist.

        The converse of the bug above, and what stops the fix from being "always
        true": the pid here is this test process, which is certainly alive.
        """
        from koopmans.aiida.setup.hq import is_hq_worker_running

        (tmp_path / "hq.worker.pid").write_text(f"{os.getpid()}\n")

        assert not is_hq_worker_running()

    @staticmethod
    def test_server_koopmans_did_not_spawn_is_reported_running(fake_hq: FakeHq) -> None:
        """A server with no pidfile is seen."""
        from koopmans.aiida.setup.hq import is_hq_server_running

        assert is_hq_server_running()

    @staticmethod
    def test_starting_a_running_server_leaves_its_state_dir_alone(
        fake_hq: FakeHq, tmp_path: Path
    ) -> None:
        """``start_hq_server`` does not clear a live server's state dir.

        It clears the dir whenever the server reads as absent, so a server with
        no pidfile used to have its access files deleted underneath it. The
        access files are what ``hq`` clients read to reach the server.
        """
        from koopmans.aiida.setup.hq import _hq_server_dir, start_hq_server

        server_dir = _hq_server_dir()
        server_dir.mkdir(parents=True)
        (server_dir / "access.json").write_text("{}")

        assert start_hq_server(wait=False)
        assert (server_dir / "access.json").exists()


class TestPoolDefault:
    """``worker_cpus`` resolves the pool the docstring promises."""

    @staticmethod
    def test_env_var_sets_the_pool(fake_hq: FakeHq, monkeypatch: pytest.MonkeyPatch) -> None:
        """``KOOPMANS_MAX_PROCS`` applies when no count is passed."""
        from koopmans.aiida.setup.hq import worker_cpus

        monkeypatch.setenv("KOOPMANS_MAX_PROCS", "9")

        assert worker_cpus() == 9
        assert worker_cpus(3) == 3

    @staticmethod
    def test_unparseable_env_var_falls_back_to_the_core_count(
        fake_hq: FakeHq, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A junk ``KOOPMANS_MAX_PROCS`` is ignored, not propagated as a crash."""
        from koopmans.aiida.setup.cores import detect_num_cores
        from koopmans.aiida.setup.hq import worker_cpus

        monkeypatch.setenv("KOOPMANS_MAX_PROCS", "lots")

        assert worker_cpus() == detect_num_cores()


class TestWorkerListParsing:
    """Output koopmans cannot read means "no worker", never a traceback."""

    @staticmethod
    def test_non_json_output_yields_no_workers(
        fake_hq: FakeHq, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A future ``hq`` that stops emitting JSON must not crash the CLI."""
        from koopmans.aiida.setup.hq import running_hq_workers

        (tmp_path / "workers.json").write_text("not json at all")

        assert running_hq_workers() == []

    @staticmethod
    def test_an_entry_without_a_cpu_count_is_skipped(fake_hq: FakeHq, tmp_path: Path) -> None:
        """One unreadable entry does not hide the workers alongside it."""
        from koopmans.aiida.setup.hq import running_hq_workers

        fake_hq.set_workers(24)
        listed = json.loads((tmp_path / "workers.json").read_text())
        listed.append({"id": 99, "configuration": {"resources": {"resources": []}}})
        (tmp_path / "workers.json").write_text(json.dumps(listed))

        assert [(w.id, w.cpus) for w in running_hq_workers()] == [(1, 24)]


class TestHqCannotAnswer:
    """An ``hq`` that cannot be run reads as "nothing running", never a crash."""

    @staticmethod
    def test_no_binary_reports_nothing_running(
        fake_hq: FakeHq, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no ``hq`` on the machine the helpers answer, rather than raise.

        Every caller of these helpers runs before the binary is guaranteed to
        exist — ``koopmans install`` itself does — so an unresolvable binary
        has to be an answer and not an exception.
        """
        from koopmans.aiida.setup.hq import is_hq_server_running, running_hq_workers

        fake_hq.set_workers(24)
        monkeypatch.delenv("KOOPMANS_HQ_BINARY")
        monkeypatch.setattr("shutil.which", lambda name: None)

        assert not is_hq_server_running()
        assert running_hq_workers() == []

    @staticmethod
    def test_an_hq_that_never_returns_reports_nothing_running(
        fake_hq: FakeHq, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hung ``hq`` times out into "nothing running", not a traceback.

        Discriminates against reading the subprocess result unconditionally:
        with the timeout raised instead of returned, this test is the only one
        that fails, since every other stub call answers instantly.
        """
        import subprocess

        from koopmans.aiida.setup.hq import is_hq_server_running, running_hq_workers

        fake_hq.set_workers(24)

        def hang(*args: Any, **kwargs: Any) -> None:
            """Behave as an ``hq`` that never answers within the timeout."""
            raise subprocess.TimeoutExpired(cmd="hq", timeout=5)

        monkeypatch.setattr(subprocess, "run", hang)

        assert not is_hq_server_running()
        assert running_hq_workers() == []


class TestWorkerCommands:
    """``koopmans backend hq`` drives the worker after installation."""

    @staticmethod
    def _invoke(args: list[str]) -> Any:
        from click.testing import CliRunner

        from koopmans import cli

        return CliRunner().invoke(cli.cli, args)

    def test_start_passes_its_cpu_count_through(self, fake_hq: FakeHq) -> None:
        """``--max-procs 7`` reaches ``hq worker start --cpus 7``.

        Establishes that the pool the user asks for is the pool the worker
        advertises, rather than being defaulted away to the core count.
        """
        result = self._invoke(["backend", "hq", "start", "--max-procs", "7"])

        assert result.exit_code == 0, result.output
        assert ["worker", "start", "--cpus", "7"] in fake_hq.commands

    def test_start_against_a_live_server_with_no_worker(self, fake_hq: FakeHq) -> None:
        """The state a worker leaves behind when it ends: server up, no worker.

        Starting must add a worker to the running server rather than restarting
        the server — which would discard its queue — and must say what it did.
        """
        result = self._invoke(["backend", "hq", "start", "--max-procs", "12"])

        assert result.exit_code == 0, result.output
        assert ["worker", "start", "--cpus", "12"] in fake_hq.commands
        assert not [c for c in fake_hq.commands if c[:2] == ["server", "start"]]
        assert "HyperQueue worker started." in result.output
        assert "pool of 12 CPU(s)" in result.output

    def test_status_with_no_worker_names_the_command_that_fixes_it(self, fake_hq: FakeHq) -> None:
        """Zero workers is a state to report, not an error to raise."""
        result = self._invoke(["backend", "hq", "status"])

        assert result.exit_code == 0, result.output
        assert "HyperQueue server is running." in result.output
        assert "No HyperQueue worker is running." in result.output
        assert "koopmans backend hq start" in result.output

    def test_stop_asks_hq_rather_than_signalling_a_pid(
        self, fake_hq: FakeHq, tmp_path: Path
    ) -> None:
        """A worker koopmans did not spawn is stopped, not just seen.

        The pidfile here names a dead process, as it does whenever a
        koopmans-spawned worker has been replaced by a hand-started one. Sending
        that pid a signal would stop nothing.
        """
        from koopmans.aiida.setup.hq import is_hq_worker_running

        fake_hq.set_workers(24)
        (tmp_path / "hq.worker.pid").write_text("999999999\n")

        result = self._invoke(["backend", "hq", "stop"])

        assert result.exit_code == 0, result.output
        assert ["worker", "stop", "all"] in fake_hq.commands
        assert not is_hq_worker_running()

    def test_restart_stops_before_it_starts(self, fake_hq: FakeHq) -> None:
        """Restart resizes the pool, which needs the old worker gone first."""
        fake_hq.set_workers(24)

        result = self._invoke(["backend", "hq", "restart", "--max-procs", "12"])

        assert result.exit_code == 0, result.output
        issued = [c for c in fake_hq.commands if c[:1] == ["worker"] and c[1] != "list"]
        assert issued == [["worker", "stop", "all"], ["worker", "start", "--cpus", "12"]]

    def test_restart_leaves_the_server_up(self, fake_hq: FakeHq) -> None:
        """Restarting the worker must not discard the queue the server holds."""
        fake_hq.set_workers(24)

        assert self._invoke(["backend", "hq", "restart"]).exit_code == 0
        assert not [c for c in fake_hq.commands if c[:2] == ["server", "stop"]]

    def test_start_without_a_binary_points_at_the_installer(
        self, fake_hq: FakeHq, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no ``hq`` to run, say so rather than blaming an empty log."""
        monkeypatch.delenv("KOOPMANS_HQ_BINARY")
        monkeypatch.setattr("shutil.which", lambda name: None)

        result = self._invoke(["backend", "hq", "start"])

        assert result.exit_code != 0
        assert "Run 'koopmans install'" in result.output

    def test_stop_refuses_when_several_workers_are_running(self, fake_hq: FakeHq) -> None:
        """With two workers, stopping is ambiguous, so say so and do nothing.

        koopmans manages one worker; several mean someone arranged them by
        hand. Stopping both would shrink the machine's capacity silently.
        """
        fake_hq.set_workers(24, 8)

        result = self._invoke(["backend", "hq", "stop"])

        assert result.exit_code != 0
        assert "2 HyperQueue workers are running" in result.output
        assert "1 (24 CPUs), 2 (8 CPUs)" in result.output
        assert not [c for c in fake_hq.commands if c[:2] == ["worker", "stop"]]

    def test_restart_refuses_when_several_workers_are_running(self, fake_hq: FakeHq) -> None:
        """Restarting two workers into one would change the pool unasked."""
        fake_hq.set_workers(24, 8)

        result = self._invoke(["backend", "hq", "restart", "--max-procs", "12"])

        assert result.exit_code != 0
        assert not [c for c in fake_hq.commands if c[1:2] in (["stop"], ["start"])]

    def test_bare_restart_keeps_the_pool_it_had(
        self, fake_hq: FakeHq, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Restarting without --max-procs must not resize the pool.

        The default would be the core count, so a bare restart of a worker
        sized by hand used to shrink it and report that as success.
        """
        from koopmans.aiida.setup import hq as hq_mod

        monkeypatch.setattr(hq_mod, "detect_num_cores", lambda: 14)
        fake_hq.set_workers(24)

        result = self._invoke(["backend", "hq", "restart"])

        assert result.exit_code == 0, result.output
        assert ["worker", "start", "--cpus", "24"] in fake_hq.commands
        assert fake_hq.worker_pools == [24]

    def test_restart_starts_the_server_when_it_is_down(self, fake_hq: FakeHq) -> None:
        """After a reboot the server is gone too, and restart must handle that.

        ``start`` brought the server up and ``restart`` did not, so the same
        machine state made one command work and the other time out.
        """
        fake_hq.stop_server()

        result = self._invoke(["backend", "hq", "restart", "--max-procs", "12"])

        assert result.exit_code == 0, result.output
        assert ["server", "start"] in fake_hq.commands
        assert fake_hq.worker_pools == [12]

    def test_zero_procs_is_rejected_before_the_worker_is_stopped(self, fake_hq: FakeHq) -> None:
        """``--max-procs 0`` must not take the machine down to nothing.

        Real ``hq worker start --cpus 0`` panics, so a zero that reaches HQ
        stops the running worker and leaves none behind.
        """
        fake_hq.set_workers(24)

        result = self._invoke(["backend", "hq", "restart", "--max-procs", "0"])

        assert result.exit_code != 0
        assert not [c for c in fake_hq.commands if c[:2] == ["worker", "stop"]]
        assert fake_hq.worker_pools == [24]

    def test_stop_and_status_without_a_binary_name_the_installer(
        self, fake_hq: FakeHq, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Never report absence when the truth is that nothing was asked.

        Both commands answer by running ``hq``. Without it they used to say
        the worker or server was not running while one was.
        """
        fake_hq.set_workers(24)
        monkeypatch.delenv("KOOPMANS_HQ_BINARY")
        monkeypatch.setattr("shutil.which", lambda name: None)

        for command in (["backend", "hq", "stop"], ["backend", "hq", "status"]):
            result = self._invoke(command)
            assert result.exit_code != 0, command
            assert "Run 'koopmans install'" in result.output, command
            assert "not running" not in result.output, command

    def test_status_reports_the_pool_and_the_default_calc_size(
        self, fake_hq: FakeHq, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Status shows both numbers, so a pool too small to schedule is visible.

        A pool below the default MPI ranks per calculation leaves every
        default-sized calculation queued forever rather than failing.
        """
        from koopmans.aiida.setup import orchestrate

        fake_hq.set_workers(12)
        monkeypatch.setattr(orchestrate, "_default_procs_per_calc", lambda: 14)

        result = self._invoke(["backend", "hq", "status"])

        assert result.exit_code == 0, result.output
        assert "pool of 12 CPU(s)" in result.output
        assert "14 MPI rank(s) by default" in result.output

    def test_status_with_the_server_down_names_the_command_that_fixes_it(
        self, fake_hq: FakeHq
    ) -> None:
        """A machine fresh from a reboot has no server, and status must say so.

        Without a server there is nothing to ask about workers, so the report
        stops at the server rather than adding "no worker is running", which
        would read as a second thing to fix.
        """
        fake_hq.stop_server()

        result = self._invoke(["backend", "hq", "status"])

        assert result.exit_code == 0, result.output
        assert "HyperQueue server is not running." in result.output
        assert "koopmans backend hq start" in result.output
        assert "worker" not in result.output

    def test_status_omits_the_default_calc_size_before_a_computer_exists(
        self, fake_hq: FakeHq, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Report no default rank count while there is no Computer to read it off.

        The number is the localhost Computer's, so a profile without one — the
        state between ``koopmans install`` creating the profile and registering
        the Computer — leaves the pool to be reported on its own.
        """
        from koopmans.aiida.setup import orchestrate

        fake_hq.set_workers(12)
        monkeypatch.setattr(orchestrate, "profile_exists", lambda: True)
        monkeypatch.setattr(orchestrate, "load_koopmans_profile", lambda: None)
        monkeypatch.setattr(orchestrate, "computer_exists", lambda: False)

        result = self._invoke(["backend", "hq", "status"])

        assert result.exit_code == 0, result.output
        assert "pool of 12 CPU(s)" in result.output
        assert "by default" not in result.output

    def test_start_reports_a_worker_that_never_registers(self, fake_hq: FakeHq) -> None:
        """A worker that dies on startup is a failure, not a silent success.

        The spawn itself always succeeds — koopmans detaches the process and
        never reads its exit status — so the only evidence is that no worker
        ever appears in ``hq worker list``.
        """
        fake_hq.ignore("worker start")

        result = self._invoke(["backend", "hq", "start", "--max-procs", "12"])

        assert result.exit_code != 0
        assert "Failed to start the HyperQueue worker" in result.output
        assert "hq.worker.log" in result.output

    def test_restart_starts_nothing_when_the_old_worker_will_not_stop(
        self, fake_hq: FakeHq
    ) -> None:
        """A stop that does not take must not be followed by a second worker.

        Two workers would then share the machine's cores between them, each
        sized for the whole of it.
        """
        fake_hq.set_workers(24)
        fake_hq.ignore("worker stop")

        result = self._invoke(["backend", "hq", "restart", "--max-procs", "12"])

        assert result.exit_code != 0
        assert "Failed to restart the HyperQueue worker" in result.output
        assert not [c for c in fake_hq.commands if c[:2] == ["worker", "start"]]
        assert fake_hq.worker_pools == [24]


class TestBackendStatusRows:
    """``koopmans backend status`` answers the HQ rows by asking ``hq``."""

    @staticmethod
    @pytest.fixture
    def stubbed_profile(monkeypatch: pytest.MonkeyPatch) -> None:
        """Report a profile with no Computer, without touching a real one."""
        from koopmans.aiida.setup import orchestrate

        monkeypatch.setattr(orchestrate, "profile_exists", lambda: True)
        monkeypatch.setattr(orchestrate, "load_koopmans_profile", lambda: None)
        monkeypatch.setattr(orchestrate, "computer_exists", lambda: False)
        monkeypatch.setattr(orchestrate, "is_daemon_running", lambda: False)

    @staticmethod
    def test_a_running_server_and_worker_are_reported(
        fake_hq: FakeHq, stubbed_profile: None
    ) -> None:
        """Both HQ rows follow ``hq``, so a hand-started worker shows up."""
        from koopmans.aiida.setup.orchestrate import verify_installation

        fake_hq.set_workers(24)

        status = verify_installation()

        assert status["hq.binary"]
        assert status["hq.server"]
        assert status["hq.worker"]

    @staticmethod
    def test_without_a_binary_the_server_and_worker_are_not_probed(
        fake_hq: FakeHq, stubbed_profile: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing ``hq`` is its own row, and stops the other two being guessed.

        A worker may well be running; koopmans simply has no way to look. The
        rows are booleans, so the binary row is what carries "unknown" — and
        the stub records that nothing was asked of it.
        """
        from koopmans.aiida.setup.orchestrate import verify_installation

        fake_hq.set_workers(24)
        monkeypatch.delenv("KOOPMANS_HQ_BINARY")
        monkeypatch.setattr("shutil.which", lambda name: None)

        status = verify_installation()

        assert not status["hq.binary"]
        assert not status["hq.server"]
        assert not status["hq.worker"]
        assert fake_hq.commands == []


def test_install_sizes_the_pool_and_the_calc_separately(
    fake_hq: FakeHq, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--max-procs`` sets the worker pool; ``--procs-per-calc`` the calc size.

    Pins the separation so a later edit cannot quietly route one flag into both:
    the assertion fails if either number reaches the wrong consumer.
    """
    from click.testing import CliRunner

    from koopmans import cli

    recorded: dict[str, Any] = {}

    monkeypatch.setattr(cli, "setup_profile", lambda **kwargs: None)
    monkeypatch.setattr(cli, "install_hq_binary", lambda: None)
    monkeypatch.setattr(cli, "_start_daemon_with_caching", lambda cache: None)
    monkeypatch.setattr(cli, "setup_computers", lambda **kwargs: recorded.update(kwargs))

    result = CliRunner().invoke(cli.cli, ["install", "--procs-per-calc", "4", "--max-procs", "12"])

    assert result.exit_code == 0, result.output
    assert recorded["nprocs"] == 4
    assert ["worker", "start", "--cpus", "12"] in fake_hq.commands


def test_get_localhost_computer_uses_hyperqueue(
    monkeypatch: pytest.MonkeyPatch, aiida_profile_clean: Any
) -> None:
    """``get_localhost_computer`` always registers with the ``hyperqueue`` scheduler."""
    from koopmans.aiida.setup import computer as computer_mod

    computer = computer_mod.get_localhost_computer(nprocs=1)
    assert computer.scheduler_type == "hyperqueue"
