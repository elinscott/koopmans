"""HyperQueue (HQ) integration for the koopmans backend.

``koopmans install`` bundles HyperQueue automatically: it downloads the
HQ binary, starts a server + worker pair, and registers the localhost
Computer with the ``hyperqueue`` scheduler so AiiDA's CalcJobs are
core-aware. This replaces the ``core.direct`` fire-and-forget submission
that would otherwise let the per-orbital DSCF fan-out oversubscribe the
machine.

The integration is invisible to end users by design. Customisation is
opt-in via env vars (no CLI flags beyond ``koopmans install
--max-procs``):

* ``KOOPMANS_HQ_BINARY`` — absolute path to a pre-installed ``hq``
  binary; if set we skip the auto-download.
* ``KOOPMANS_MAX_PROCS`` — integer cap on the worker's advertised CPU
  pool. Same as ``koopmans install --max-procs``. Defaults to the box's
  physical core count.
* ``KOOPMANS_HQ_PORT`` — override the HQ server's default port.

HQ is required for the localhost backend — there is no ``core.direct``
fallback. Install failure (unsupported platform, network error, checksum
mismatch) raises and surfaces to the user as a clear ``koopmans install``
error. The HQ wiring exists only because localhost itself has no native
queue; once koopmans grows remote-Computer support against a real
scheduler (Slurm/PBS/etc.), that path will not involve HQ at all.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import click

from .cores import detect_num_cores
from .profile import koopmans_dir

logger = logging.getLogger(__name__)

# HyperQueue binary release pin. Update sha256 when bumping HQ_VERSION.
HQ_VERSION = "0.19.0"
HQ_ARCHIVE_SHA256 = "80a72cd53a265967650a10c8072ed73ad0efe5484278bcbd12b4168c3f9017c3"
HQ_ARCHIVE_URL_TEMPLATE = (
    "https://github.com/It4innovations/hyperqueue/releases/download/"
    "v{version}/hq-v{version}-linux-x64.tar.gz"
)


# ----------------------------------------------------------------------
# Binary location & install
# ----------------------------------------------------------------------


def hq_bin_path() -> Path:
    """Path where the bundled HQ binary lives."""
    return koopmans_dir() / "bin" / "hq"


def hq_binary() -> Path | None:
    """Return the path to a usable ``hq`` binary, or None if unavailable.

    Resolution order:

    1. ``KOOPMANS_HQ_BINARY`` env var (lets sysadmins point at a system
       install or an alternate cache).
    2. The koopmans-managed bundled binary.
    3. Any ``hq`` on ``PATH``.
    """
    override = os.environ.get("KOOPMANS_HQ_BINARY")
    if override:
        candidate = Path(override)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    bundled = hq_bin_path()
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return bundled
    on_path = shutil.which("hq")
    if on_path:
        return Path(on_path)
    return None


def install_hq_binary(force: bool = False) -> Path:
    """Download and install the bundled HQ binary.

    Idempotent: if the cached binary is present and ``force=False`` we
    reuse it. Raises ``click.ClickException`` on unsupported platform,
    download error, or checksum mismatch — the koopmans backend has no
    fallback when HQ is unavailable.
    """
    import hashlib
    import platform
    import tarfile
    import urllib.error
    import urllib.request

    override = os.environ.get("KOOPMANS_HQ_BINARY")
    if override:
        candidate = Path(override)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            click.echo(f"  Using HQ binary from KOOPMANS_HQ_BINARY: {candidate}")
            return candidate
        raise click.ClickException(f"KOOPMANS_HQ_BINARY={override} is not a usable executable.")

    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise click.ClickException(
            f"HyperQueue prebuilt binary is only published for linux-x64 "
            f"(detected: {platform.system()}/{platform.machine()}). "
            "Install a compatible ``hq`` manually and point at it via "
            "KOOPMANS_HQ_BINARY."
        )

    bin_path = hq_bin_path()
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    if bin_path.is_file() and not force:
        click.echo(f"  HQ v{HQ_VERSION} already installed at {bin_path}.")
        return bin_path

    url = HQ_ARCHIVE_URL_TEMPLATE.format(version=HQ_VERSION)
    click.echo(f"  Downloading HyperQueue v{HQ_VERSION} from {url}")

    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310
            archive_bytes = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise click.ClickException(f"Failed to download HQ binary from {url}: {exc}") from exc

    digest = hashlib.sha256(archive_bytes).hexdigest()
    if digest != HQ_ARCHIVE_SHA256:
        raise click.ClickException(
            f"HQ archive checksum mismatch: got {digest}, expected {HQ_ARCHIVE_SHA256}."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "hq.tar.gz"
        archive_path.write_bytes(archive_bytes)
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if Path(member.name).name == "hq" and member.isfile():
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    bin_path.write_bytes(extracted.read())
                    bin_path.chmod(0o755)
                    break
            else:
                raise click.ClickException("HQ archive did not contain an ``hq`` binary.")

    click.echo(f"  Installed HQ v{HQ_VERSION} -> {bin_path}")
    return bin_path


# ----------------------------------------------------------------------
# Server + worker process management
# ----------------------------------------------------------------------


def _hq_server_dir() -> Path:
    """Subdir HQ uses for its server-state file (``--server-dir``)."""
    return koopmans_dir() / "hq-server-dir"


def _hq_pidfile(role: str) -> Path:
    return koopmans_dir() / f"hq.{role}.pid"


def _hq_logfile(role: str) -> Path:
    return koopmans_dir() / f"hq.{role}.log"


def _read_pidfile(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _hq_env() -> dict[str, str]:
    """Subprocess env that points HQ at the koopmans-managed server dir."""
    env = os.environ.copy()
    env["HQ_SERVER_DIR"] = str(_hq_server_dir())
    return env


def is_hq_server_running() -> bool:
    """Return True if the koopmans HQ server is running."""
    pid = _read_pidfile(_hq_pidfile("server"))
    if pid is None or not _process_alive(pid):
        return False
    binary = hq_binary()
    if binary is None:
        return False
    try:
        result = subprocess.run(  # noqa: S603 - binary path validated by hq_binary()
            [str(binary), "server", "info"],
            capture_output=True,
            text=True,
            timeout=5,
            env=_hq_env(),
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0


def is_hq_worker_running() -> bool:
    """Return True if the koopmans HQ worker is running."""
    pid = _read_pidfile(_hq_pidfile("worker"))
    if pid is None:
        return False
    return _process_alive(pid)


def _spawn_hq_process(args: list[str], role: str) -> int:
    """Launch a detached HQ subprocess, recording its pid and log path."""
    log_path = _hq_logfile(role)
    pid_path = _hq_pidfile(role)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_fh = open(log_path, "ab")
    try:
        proc = subprocess.Popen(  # noqa: S603 - args constructed from validated paths
            args,
            stdout=log_fh,
            stderr=log_fh,
            stdin=subprocess.DEVNULL,
            env=_hq_env(),
            start_new_session=True,
        )
    finally:
        log_fh.close()
    pid_path.write_text(str(proc.pid) + "\n")
    return proc.pid


def start_hq_server(wait: bool = True) -> bool:
    """Start the HQ server if it isn't already running."""
    if is_hq_server_running():
        return True

    binary = hq_binary()
    if binary is None:
        return False

    server_dir = _hq_server_dir()
    if server_dir.exists():
        # Stale state from a prior crashed server confuses HQ.
        shutil.rmtree(server_dir, ignore_errors=True)
    server_dir.mkdir(parents=True, exist_ok=True)

    args = [str(binary), "server", "start"]
    port = os.environ.get("KOOPMANS_HQ_PORT")
    if port:
        args.extend(["--port", str(port)])

    _spawn_hq_process(args, role="server")

    if not wait:
        return True
    for _ in range(60):
        if is_hq_server_running():
            return True
        time.sleep(0.5)
    return False


def stop_hq_server() -> bool:  # noqa: C901
    """Stop the HQ server (and any worker spawned against it)."""
    pid_path = _hq_pidfile("server")
    if not is_hq_server_running():
        if pid_path.exists():
            pid_path.unlink()
        return True

    binary = hq_binary()
    if binary is not None:
        try:
            subprocess.run(  # noqa: S603
                [str(binary), "server", "stop"],
                capture_output=True,
                timeout=10,
                env=_hq_env(),
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("hq server stop failed: %s", exc)

    pid = _read_pidfile(pid_path)
    if pid is not None and _process_alive(pid):
        try:
            os.kill(pid, 15)  # SIGTERM
        except OSError:
            pass
        for _ in range(20):
            if not _process_alive(pid):
                break
            time.sleep(0.25)

    if pid_path.exists():
        pid_path.unlink()
    worker_pid_path = _hq_pidfile("worker")
    if worker_pid_path.exists():
        worker_pid_path.unlink()
    return not is_hq_server_running()


def start_hq_worker(wait: bool = True, resources: int | None = None) -> bool:  # noqa: C901
    """Start the HQ worker process advertising ``resources`` CPUs.

    Args:
        wait: poll until the worker registers with the server.
        resources: CPU count to advertise. Defaults to ``KOOPMANS_MAX_PROCS``
            env var, falling back to the box's physical core count.
    """
    if is_hq_worker_running():
        return True

    binary = hq_binary()
    if binary is None:
        return False

    if resources is None:
        env_override = os.environ.get("KOOPMANS_MAX_PROCS")
        if env_override:
            try:
                resources = int(env_override)
            except ValueError:
                logger.warning(
                    "KOOPMANS_MAX_PROCS=%r is not an integer; ignoring",
                    env_override,
                )
                resources = None
    if resources is None:
        resources = detect_num_cores()

    args = [str(binary), "worker", "start", "--cpus", str(resources)]
    _spawn_hq_process(args, role="worker")

    if not wait:
        return True
    for _ in range(60):
        if is_hq_worker_running():
            try:
                result = subprocess.run(  # noqa: S603
                    [str(binary), "worker", "list", "--filter", "running"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    env=_hq_env(),
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return True
            except (subprocess.SubprocessError, OSError):
                pass
        time.sleep(0.5)
    return is_hq_worker_running()


def stop_hq_worker() -> bool:
    """Stop the HQ worker process."""
    pid_path = _hq_pidfile("worker")
    if not is_hq_worker_running():
        if pid_path.exists():
            pid_path.unlink()
        return True

    pid = _read_pidfile(pid_path)
    if pid is None:
        return True

    try:
        os.kill(pid, 15)
    except OSError:
        pass
    for _ in range(20):
        if not _process_alive(pid):
            break
        time.sleep(0.25)
    if pid_path.exists():
        pid_path.unlink()
    return not is_hq_worker_running()


def ensure_hq_running(resources: int | None = None) -> bool:
    """Ensure the HQ server + worker are up.

    Assumes :func:`install_hq_binary` has already placed a usable ``hq``
    binary; returns False if the server or worker fail to start.
    """
    if hq_binary() is None:
        logger.warning("HQ binary unavailable; call install_hq_binary first.")
        return False
    server_ok = start_hq_server(wait=True)
    if not server_ok:
        logger.warning("HQ server failed to start.")
        return False
    worker_ok = start_hq_worker(wait=True, resources=resources)
    if not worker_ok:
        logger.warning("HQ worker failed to start.")
        return False
    return True


def stop_hq() -> None:
    """Tear down the HQ worker + server."""
    stop_hq_worker()
    stop_hq_server()
