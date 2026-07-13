"""Localhost AiiDA Computer registration.

Always uses ``hyperqueue`` as the scheduler — HyperQueue is the
authoritative resource cap on the localhost backend (see :mod:`.hq`).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import click

from .cores import detect_num_cores, physical_core_count

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aiida.orm import Computer

COMPUTER_LABEL = "localhost"


def computer_exists() -> bool:
    """Check if the localhost computer already exists."""
    from aiida import orm

    try:
        orm.load_computer(COMPUTER_LABEL)
        return True
    except Exception:
        return False


def get_localhost_computer(nprocs: int | None = None) -> Computer:
    """Get or create the localhost computer with the ``hyperqueue`` scheduler.

    Args:
        nprocs: Number of MPI processes per machine. If None, auto-detects
            the box's physical core count.
    """
    from aiida import orm
    from aiida.manage.configuration import get_config

    if nprocs is None:
        nprocs = detect_num_cores()
        physical = physical_core_count()
        logical = os.cpu_count()
        if physical is not None and logical is not None and physical != logical:
            click.echo(
                f"  Detected {physical} physical / {logical} logical cores; "
                f"using {nprocs} MPI rank(s) per calculation to avoid hyperthread "
                f"oversubscription. Override with --procs-per-calc."
            )

    if computer_exists():
        computer = orm.load_computer(COMPUTER_LABEL)
        computer.set_default_mpiprocs_per_machine(nprocs)
        click.echo(
            f"Computer '{COMPUTER_LABEL}' already exists "
            f"(scheduler={computer.scheduler_type}, nprocs={nprocs})."
        )
        return computer

    click.echo(f"Creating computer '{COMPUTER_LABEL}'...")

    config = get_config()
    workdir = Path(config.dirpath) / "scratch" / "koopmans"
    workdir.mkdir(parents=True, exist_ok=True)

    computer = orm.Computer(
        label=COMPUTER_LABEL,
        description="Localhost computer configured by koopmans (HyperQueue scheduler)",
        hostname="localhost",
        transport_type="core.local",
        scheduler_type="hyperqueue",
        workdir=str(workdir),
    )
    computer.store()

    computer.configure()
    computer.set_minimum_job_poll_interval(0.0)
    computer.set_default_mpiprocs_per_machine(nprocs)

    click.echo(f"Created computer '{COMPUTER_LABEL}' (scheduler=hyperqueue, nprocs={nprocs}).")
    return computer
