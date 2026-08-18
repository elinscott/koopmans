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

#: Why no kcp.x route interpolates a band structure yet: kcp.x works in the
#: supercell the k-mesh folds to, where the band structure has collapsed onto
#: the zone centre. Recovering it means unfolding the converged Koopmans
#: Hamiltonian back onto the primitive cell and interpolating it — a stage
#: koopmans has not ported (koopmans#188).
_KCP_HAS_NO_BANDS = (
    "`kpoints.path` cannot take effect on the kcp.x route: its steps work in the "
    "supercell `kpoints.grid` folds to, and unfolding the converged Koopmans "
    "Hamiltonian back onto that path is not ported yet. Remove "
    "`kpoints.path`.{alternative}"
)

NO_BAND_PATH_ON_DSCF = _KCP_HAS_NO_BANDS.format(
    alternative=" Screening with `screening_method = 'dfpt'` gives you a Koopmans band "
    "structure along it."
)

#: The same rejection without the DFPT alternative, which the trajectory task
#: does not offer.
NO_BAND_PATH_ON_TRAJECTORY = _KCP_HAS_NO_BANDS.format(alternative="")


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
            # No kcp.x route exists for this initialisation, and sending the
            # reader to `screening_method = 'dfpt'` earns a second refusal.
            return None
        return NO_BAND_PATH_ON_DSCF

    if workflow.task == Task.TRAJECTORY:
        if workflow.calculate_alpha and workflow.screening_method == CalculateScreeningMethod.DFPT:
            # The task runs kcp.x whatever the input asks for, and the reader
            # has to hear about the method they asked for first.
            return None
        return NO_BAND_PATH_ON_TRAJECTORY

    return None
