"""AiiDA daemon lifecycle helpers."""

from __future__ import annotations

import logging
import os
import time

import click

logger = logging.getLogger(__name__)


def is_daemon_running() -> bool:
    """Check if the AiiDA daemon is running."""
    from aiida.engine.daemon.client import get_daemon_client

    try:
        client = get_daemon_client()
        return client.is_daemon_running
    except Exception:
        return False


def _ensure_hq_env() -> None:
    """Make ``hq`` and our HQ server discoverable to the AiiDA daemon worker.

    The daemon inherits environment from whoever launches it. Without
    this, two failures happen at submit time:

    1. ``bash: line 1: hq: command not found`` — bundled ``hq`` lives at
       ``${AIIDA_CONFIG}/koopmans/bin/hq``, not on system PATH.
    2. ``hq submit`` finds the binary but defaults to looking for the
       server at ``$HOME/.hq-server``; ours lives under the koopmans
       config dir.

    Fixed by prepending the bundled bin dir to ``PATH`` and exporting
    ``HQ_SERVER_DIR`` to point at the koopmans-managed server-dir.
    Both are scoped to *this* Python process (and its forks — the
    daemon worker), not the user's shell.
    """
    from .hq import _hq_server_dir, hq_bin_path

    bin_dir = str(hq_bin_path().parent)
    current_path = os.environ.get("PATH", "")
    if bin_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current_path}" if current_path else bin_dir

    os.environ["HQ_SERVER_DIR"] = str(_hq_server_dir())


def start_daemon(wait: bool = True, cache: bool = True) -> bool:
    """Start the AiiDA daemon if it's not already running.

    Args:
        wait: If True, wait for the daemon to be fully started.
        cache: If True, enable AiiDA caching for calculations.
    """
    from aiida.engine.daemon.client import get_daemon_client
    from aiida.manage import get_config

    config = get_config()
    config.set_option("caching.default_enabled", cache)
    config.store()

    if is_daemon_running():
        return True

    _ensure_hq_env()

    try:
        client = get_daemon_client()
        response = client.start_daemon()

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
