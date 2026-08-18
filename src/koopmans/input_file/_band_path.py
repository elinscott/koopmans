"""Which tasks can interpolate along a ``kpoints.path``, and what to say when one cannot."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koopmans.input_file.workflow import (
    CalculateScreeningMethod,
    Task,
    VariationalOrbitalType,
)

if TYPE_CHECKING:
    from koopmans.input_file.workflow import WorkflowConfig

#: What to write instead of a k-path on the ph.x route.
NO_BAND_PATH_ON_DFT_EPS = (
    "`kpoints.path` cannot take effect in a `dft_eps` calculation: it runs one scf "
    "and then ph.x, which computes a dielectric constant and no band structure. "
    "Remove `kpoints.path`, or run `task: dft_bands` to get a band structure."
)

#: What to write instead of a k-path on a molecular ΔSCF. The interpolation
#: unfolds the converged Koopmans Hamiltonian in the Wannier basis, which the
#: Kohn-Sham-initialised molecular route does not build — and an isolated
#: molecule has no band structure to unfold onto in the first place.
NO_BAND_PATH_ON_MOLECULAR_DSCF = (
    "`kpoints.path` cannot take effect in a molecular calculation: a band structure "
    "is a property of a periodic system, and this cell is periodic along no direction. "
    "Remove `kpoints.path`; the ΔSCF eigenvalues are already the molecule's spectrum."
)

#: What to write instead of a k-path on the trajectory task.
NO_BAND_PATH_ON_TRAJECTORY = (
    "`kpoints.path` cannot take effect in a `trajectory` calculation: it screens each "
    "snapshot and reports screening parameters and eigenvalues, not band structures. "
    "Remove `kpoints.path`, or run `task: singlepoint` on the structure whose band "
    "structure you want."
)


def dscf_initialization_is_supported(init_orbitals: VariationalOrbitalType, periodic: bool) -> bool:
    """Report whether the kcp.x singlepoint runs this initialisation route.

    Two routes exist: molecular ``kohn-sham``, and periodic ``mlwfs`` /
    ``projwfs``. Every other pairing it refuses on entry.
    """
    if init_orbitals in (VariationalOrbitalType.MLWFS, VariationalOrbitalType.PROJWFS):
        return periodic
    return init_orbitals == VariationalOrbitalType.KOHN_SHAM and not periodic


def band_path_refusal(workflow: WorkflowConfig, periodic: bool) -> str | None:
    """Return what to tell an input whose task cannot interpolate along its path.

    ``None`` for a task that can, and for one whose path is not the first
    thing standing in its way: an input the task refuses outright must hear
    that refusal instead, or it is sent to fix a keyword and told no again.

    Args:
        workflow: The input's ``workflow`` block.
        periodic: Whether the structure is periodic along any cell vector.
    """
    if workflow.task == Task.DFT_EPS:
        return NO_BAND_PATH_ON_DFT_EPS

    if workflow.task == Task.SINGLEPOINT:
        if workflow.screening_method != CalculateScreeningMethod.DSCF:
            # kcw.x interpolates along the path.
            return None
        if not dscf_initialization_is_supported(workflow.init_orbitals, periodic):
            # No kcp.x route exists for this initialisation, and naming the
            # path earns the reader a second refusal.
            return None
        if workflow.init_orbitals in (
            VariationalOrbitalType.MLWFS,
            VariationalOrbitalType.PROJWFS,
        ):
            # The Wannier-initialised route unfolds its Koopmans Hamiltonian
            # and interpolates it along the path.
            return None
        return NO_BAND_PATH_ON_MOLECULAR_DSCF

    if workflow.task == Task.TRAJECTORY:
        if workflow.calculate_alpha and workflow.screening_method == CalculateScreeningMethod.DFPT:
            # The task runs kcp.x whatever the input asks for, and the reader
            # has to hear about the method they asked for first.
            return None
        return NO_BAND_PATH_ON_TRAJECTORY

    return None
