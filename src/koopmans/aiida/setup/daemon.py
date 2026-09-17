"""AiiDA daemon lifecycle helpers."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def is_daemon_running() -> bool:
    """Check if the AiiDA daemon is running."""
    # aiida.engine.daemon.client resolves and caches the verdi executable at
    # import time, so the PATH fix must land before this first import of it.
    _ensure_daemon_env()
    from aiida.engine.daemon.client import get_daemon_client

    try:
        client = get_daemon_client()
        return client.is_daemon_running
    except Exception:
        return False


def _ensure_daemon_env() -> None:
    """Set the environment the AiiDA daemon worker needs before it forks.

    The daemon inherits environment from whoever launches it. Without
    this, three failures happen:

    1. ``bash: line 1: hq: command not found`` at submit time — bundled
       ``hq`` lives at ``${AIIDA_CONFIG}/koopmans/bin/hq``, not on
       system PATH.
    2. ``hq submit`` finds the binary but defaults to looking for the
       server at ``$HOME/.hq-server``; ours lives under the koopmans
       config dir.
    3. ``sqlite3.OperationalError: unable to open database file`` when
       a large transaction (e.g. the link inserts of a wide fan-out)
       spills to a temporary file: sqlite tries ``/var/tmp`` first,
       which sandboxed environments may not grant.
    4. ``Unable to find 'verdi' in the path``, or a ``verdi`` from a
       different Python resolved instead, when ``aiida.engine.daemon.client``
       looks up ``verdi`` with ``shutil.which`` at import time: whichever
       ``verdi`` is first on the caller's ``PATH`` at that moment is cached
       for the life of the process, regardless of the koopmans venv this
       code is running from.

    Fixed by prepending this interpreter's own ``bin`` directory and the
    bundled hq bin dir to ``PATH``, and exporting ``HQ_SERVER_DIR`` and
    ``SQLITE_TMPDIR`` to koopmans-managed directories. All are scoped to
    *this* Python process (and its forks — the daemon worker), not the
    user's shell. Every caller that may import ``aiida.engine.daemon.client``
    calls this first, so the fix lands before that module's first import
    anywhere in the process.
    """
    from .hq import _hq_server_dir, hq_bin_path
    from .profile import koopmans_dir

    current_path = os.environ.get("PATH", "")
    existing_entries = current_path.split(os.pathsep) if current_path else []

    # interpreter_bin_dir goes first: shutil.which("verdi") must resolve to
    # this venv's verdi, whatever verdi the caller's PATH already contains.
    interpreter_bin_dir = str(Path(sys.executable).parent)
    hq_dir = str(hq_bin_path().parent)
    new_entries = [interpreter_bin_dir, hq_dir]

    path_entries = new_entries + [entry for entry in existing_entries if entry not in new_entries]
    os.environ["PATH"] = os.pathsep.join(path_entries)

    os.environ["HQ_SERVER_DIR"] = str(_hq_server_dir())

    sqlite_tmpdir = koopmans_dir() / "sqlite-tmp"
    sqlite_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["SQLITE_TMPDIR"] = str(sqlite_tmpdir)


def start_daemon(wait: bool = True, cache: bool = True) -> bool:
    """Start the AiiDA daemon if it's not already running.

    Args:
        wait: If True, wait for the daemon to be fully started.
        cache: If True, enable AiiDA caching for calculations.
    """
    _ensure_daemon_env()
    from aiida.engine.daemon.client import get_daemon_client
    from aiida.manage import get_config

    config = get_config()
    config.set_option("caching.default_enabled", cache)  # type: ignore[no-untyped-call]
    config.store()  # type: ignore[no-untyped-call]

    if is_daemon_running():
        return True

    try:
        client = get_daemon_client()
        # DaemonClient.start_daemon is annotated ``-> None`` upstream, but the
        # non-wait path historically inspected its return value; preserve that.
        response = client.start_daemon()  # type: ignore[func-returns-value]

        if wait:
            for _ in range(30):
                if client.is_daemon_running:
                    return True
                time.sleep(1)
            return False

        return response is not None
    except Exception as e:
        logger.warning("Failed to start daemon: %s", e)
        return False


def stop_daemon() -> bool:
    """Stop the AiiDA daemon."""
    _ensure_daemon_env()
    from aiida.engine.daemon.client import get_daemon_client

    if not is_daemon_running():
        return True

    try:
        client = get_daemon_client()
        client.stop_daemon(wait=True)
        return True
    except Exception as e:
        logger.warning("Failed to stop daemon: %s", e)
        return False


def ensure_daemon_running() -> None:
    """Ensure the AiiDA daemon is running, starting it if necessary."""
    if is_daemon_running():
        return

    click.echo("Starting AiiDA daemon...")
    if start_daemon(wait=True):
        click.echo("Daemon started successfully.")
    else:
        raise click.ClickException(
            "Failed to start the AiiDA daemon. "
            "This may be because RabbitMQ is not available. "
            "Please check your installation with 'koopmans backend status'."
        )
