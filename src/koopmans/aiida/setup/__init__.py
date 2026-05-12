"""Automatic AiiDA setup for koopmans.

This package handles automatic configuration of AiiDA profiles, computers,
codes, and the bundled HyperQueue scheduler so users can run koopmans
calculations without needing AiiDA knowledge. ``koopmans install`` is the
single user entry point; the implementation is split across submodules:

* :mod:`.profile` -- AiiDA profile setup (presto-style).
* :mod:`.computer` -- localhost Computer registration.
* :mod:`.codes` -- PATH scanning + Code registration for QE executables.
* :mod:`.pseudos` -- on-demand pseudopotential family installers.
* :mod:`.daemon` -- AiiDA daemon lifecycle helpers.
* :mod:`.hq` -- HyperQueue binary install + server/worker lifecycle.
* :mod:`.cores` -- physical core detection.
* :mod:`.orchestrate` -- ``koopmans install`` orchestration.

The names re-exported below are the symbols imported by other modules in
the codebase (CLI / progress / workflows / conversion). New code in those
modules should import from the relevant submodule directly; this stub
exists to keep the historical ``from koopmans.aiida.setup import ...``
call sites working.
"""

from __future__ import annotations

# ``codes.list_codes`` is used by the CLI; expose it under the same name.
from .codes import list_codes
from .daemon import (
    ensure_daemon_running,
    is_daemon_running,
    start_daemon,
    stop_daemon,
)
from .hq import ensure_hq_running, install_hq_binary
from .orchestrate import (
    print_status,
    setup_computers,
    uninstall_backend,
)
from .profile import load_koopmans_profile, setup_profile
from .pseudos import ensure_pseudo_family_installed

__all__ = [
    "ensure_daemon_running",
    "ensure_hq_running",
    "ensure_pseudo_family_installed",
    "install_hq_binary",
    "is_daemon_running",
    "list_codes",
    "load_koopmans_profile",
    "print_status",
    "setup_computers",
    "setup_profile",
    "start_daemon",
    "stop_daemon",
    "uninstall_backend",
]
