"""Physical core detection for resource sizing.

kcp.x and friends are tightly-coupled MPI workloads that lose performance
when oversubscribed onto hyperthreads, so we count physical cores — not
logical ones. Used both as the default ``num_mpiprocs_per_machine`` for
the localhost Computer and as the worker's advertised CPU pool when HQ
is enabled.
"""

from __future__ import annotations

import os
import platform


def detect_num_cores() -> int:
    """Return the number of physical CPU cores, falling back to logical."""
    physical = physical_core_count()
    if physical is not None:
        return physical
    return os.cpu_count() or 1


def physical_core_count() -> int | None:
    """Return the number of physical CPU cores, or None if undetectable.

    Linux: parse ``/proc/cpuinfo`` for unique ``(physical id, core id)``
    pairs. Other platforms: returns None and the caller falls back.
    """
    if platform.system() != "Linux":
        return None
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
    except OSError:
        return None
    physical: set[tuple[str, str]] = set()
    current_phys = current_core = None
    for line in text.splitlines():
        if ":" not in line:
            if current_phys is not None and current_core is not None:
                physical.add((current_phys, current_core))
            current_phys = current_core = None
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "physical id":
            current_phys = value
        elif key == "core id":
            current_core = value
    if current_phys is not None and current_core is not None:
        physical.add((current_phys, current_core))
    return len(physical) or None
