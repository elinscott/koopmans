"""Tests for the localhost computer's default OpenMP/BLAS thread pin.

The GNU Quantum ESPRESSO builds link threaded OpenBLAS; without a pin each
MPI rank spawns its own BLAS thread pool and oversubscribes the allocation.
:func:`koopmans.aiida.setup.computer.get_localhost_computer` therefore stamps
a three-variable export block onto the (mutable) computer's prepend text, both
when creating it and when migrating one that predates the pin.
"""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.aiida.setup.computer import (
    THREAD_PIN_PREPEND,
    computer_has_thread_pin,
    get_localhost_computer,
)

pytestmark = pytest.mark.usefixtures("aiida_profile")

EXPECTED_EXPORTS = (
    "export OMP_NUM_THREADS=1",
    "export OPENBLAS_NUM_THREADS=1",
    "export MKL_NUM_THREADS=1",
)


def test_thread_pin_prepend_lists_all_three_exports() -> None:
    """The pin block exports all three thread-count variables."""
    for export in EXPECTED_EXPORTS:
        assert export in THREAD_PIN_PREPEND


def test_computer_has_thread_pin_detects_presence(aiida_computer_local: Any) -> None:
    """The detector reports the pin only once it is set on the prepend."""
    computer = aiida_computer_local(label="localhost")
    assert not computer_has_thread_pin(computer)
    computer.set_prepend_text(THREAD_PIN_PREPEND)
    assert computer_has_thread_pin(computer)


def test_existing_computer_is_migrated_in_place(aiida_computer_local: Any) -> None:
    """Rerunning install stamps the pin onto a computer that lacks it."""
    aiida_computer_local(label="localhost")
    computer = get_localhost_computer(nprocs=1)
    assert computer_has_thread_pin(computer)
    for export in EXPECTED_EXPORTS:
        assert export in computer.get_prepend_text()


def test_migration_preserves_a_prior_prepend(aiida_computer_local: Any) -> None:
    """An existing prepend is not clobbered when the pin is added."""
    existing = aiida_computer_local(label="localhost")
    existing.set_prepend_text("module load custom")
    computer = get_localhost_computer(nprocs=1)
    prepend = computer.get_prepend_text()
    assert "module load custom" in prepend
    assert computer_has_thread_pin(computer)


def test_migration_is_idempotent(aiida_computer_local: Any) -> None:
    """A second install does not duplicate the pin."""
    aiida_computer_local(label="localhost")
    get_localhost_computer(nprocs=1)
    computer = get_localhost_computer(nprocs=1)
    assert computer.get_prepend_text().count("export OMP_NUM_THREADS=1") == 1


def test_exports_land_in_the_generated_submit_script(aiida_computer_local: Any) -> None:
    """The pin reaches the scheduler submit script via the computer prepend."""
    aiida_computer_local(label="localhost")
    computer = get_localhost_computer(nprocs=1)

    from aiida.common.datastructures import CodeRunMode
    from aiida.schedulers.datastructures import JobTemplate

    scheduler = computer.get_scheduler()
    template = JobTemplate()
    template.job_resource = scheduler.create_job_resource(
        num_machines=1, num_mpiprocs_per_machine=1
    )
    template.codes_info = []
    template.codes_run_mode = CodeRunMode.SERIAL
    template.prepend_text = computer.get_prepend_text()
    script = scheduler.get_submit_script(template)
    for export in EXPECTED_EXPORTS:
        assert export in script
