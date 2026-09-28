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

#: What to write instead of a k-path on the bse task.
NO_BAND_PATH_ON_BSE = (
    "`kpoints.path` cannot take effect in a `bse` calculation: the composed DFPT "
    "screening step runs no Koopmans band-structure interpolation, and the yambo BSE "
    "step reports an exciton spectrum, not a band structure. Remove `kpoints.path`, or "
    "run `task: singlepoint` with `screening_method: dfpt` to get the interpolated "
    "band structure."
)

#: What to write instead of a k-path under ``atoms.snapshots``. The fan-out
#: reports one set of screening parameters and eigenvalues per frame, and no
#: step of it interpolates a Koopmans Hamiltonian along a path.
NO_BAND_PATH_ON_SNAPSHOTS = (
    "`kpoints.path` cannot take effect together with `atoms.snapshots`: each frame is "
    "screened on a supercell and reports screening parameters and eigenvalues, not a "
    "band structure. Remove `kpoints.path`, or replace `atoms.snapshots` with the "
    "`atoms.atomic_positions` of the one structure whose band structure you want."
)


def dscf_initialization_is_supported(init_orbitals: VariationalOrbitalType, periodic: bool) -> bool:
    """Report whether the kcp.x singlepoint runs this initialisation route.

    Two routes exist: molecular ``kohn-sham``, and periodic ``mlwfs`` /
    ``projwfs``. Every other pairing it refuses on entry.
    """
    if init_orbitals in (VariationalOrbitalType.MLWFS, VariationalOrbitalType.PROJWFS):
        return periodic
    return init_orbitals == VariationalOrbitalType.KOHN_SHAM and not periodic


def band_path_refusal(workflow: WorkflowConfig, periodic: bool, has_snapshots: bool) -> str | None:
    """Return what to tell an input whose task cannot interpolate along its path.

    ``None`` for a task that can, and for one whose path is not the first
    thing standing in its way: an input the task refuses outright must hear
    that refusal instead, or it is sent to fix a keyword and told no again.

    ``atoms.snapshots`` fans a task out per frame; the fan-out itself only
    runs ``task: singlepoint`` with ``screening_method: dscf``, so a path
    under snapshots is refused only there — everywhere else, the fan-out's
    own task or screening-method refusal names the actual problem, and
    naming the path too would be a second refusal for one input.

    Args:
        workflow: The input's ``workflow`` block.
        periodic: Whether the structure is periodic along any cell vector.
        has_snapshots: Whether ``atoms.snapshots`` is set.
    """
    if has_snapshots:
        if workflow.task != Task.SINGLEPOINT:
            return None
        if workflow.calculate_alpha and workflow.screening_method == CalculateScreeningMethod.DFPT:
            return None
        return NO_BAND_PATH_ON_SNAPSHOTS

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

    if workflow.task == Task.BSE:
        return NO_BAND_PATH_ON_BSE

    return None
