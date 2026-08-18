"""Settings for a `Workflow` object."""

from enum import Enum, StrEnum
from typing import Annotated, Any, Self

from aiida_koopmans.functionals import Correction
from aiida_koopmans.variational_orbitals import VariationalOrbitalType
from aiida_quantumespresso.common.types import SpinType
from pydantic import Field, field_validator, model_validator

from koopmans.base import BaseModel

# ``Correction`` and ``VariationalOrbitalType`` are re-exported so that
# ``from koopmans.input_file.workflow import Correction`` keeps working — the
# canonical definitions live in ``aiida_koopmans.functionals`` and
# ``aiida_koopmans.variational_orbitals``.
__all__ = [
    "CalculateScreeningMethod",
    "Correction",
    "GroupOrbitalsBy",
    "Task",
    "VariationalOrbitalType",
    "WorkflowConfig",
]

FloatGE1 = Annotated[float, Field(ge=1.0)]


class Task(Enum):
    """Valid tasks that ``koopmans`` can perform."""

    SINGLEPOINT = "singlepoint"
    CONVERGENCE = "convergence"
    WANNIERIZE = "wannierize"
    DFT_BANDS = "dft_bands"
    DFT_EPS = "dft_eps"
    TRAJECTORY = "trajectory"


class CalculateScreeningMethod(Enum):
    """Valid methods for calculating screening parameters."""

    DSCF = "dscf"
    DFPT = "dfpt"


class GroupOrbitalsBy(StrEnum):
    """Criterion for grouping variational orbitals to share a screening parameter."""

    SELF_HARTREE = "self_hartree"
    SPREAD = "spread"
    NONE = "none"


class WorkflowConfig(BaseModel):
    """Model for the configuration of a `Workflow`."""

    task: Task = Field(default=Task.SINGLEPOINT, description="Task to perform")
    correction: Correction = Field(
        default=Correction.KI,
        description="orbital-density-dependent-functional/density-functional to use",
    )
    calculate_alpha: bool = Field(
        default=True, description="whether or not to calculate the screening parameters ab-initio"
    )
    pseudo_library: str = Field(
        description="the label of the pseudopotential family to use. Any family you have installed yourself is used as it stands, whatever its label; a label naming no installed family is downloaded, which koopmans can do for the norm-conserving PseudoDojo and SG15 families `koopmans pseudos` lists. A family that publishes no recommended cutoffs takes them from `calculator_parameters.ecutwfc` instead"
    )
    screening_method: CalculateScreeningMethod = Field(
        default=CalculateScreeningMethod.DSCF,
        description="the method to calculate the screening parameters: either with ΔSCF or DFPT",
    )
    init_orbitals: VariationalOrbitalType = Field(
        default=VariationalOrbitalType.PZ,
        description="which orbitals to use as an initial guess for the variational orbitals",
    )
    init_empty_orbitals: VariationalOrbitalType = Field(
        description="which orbitals to use as an initial guess for the empty variational orbitals"
    )
    frozen_orbitals: bool | None = Field(
        default=None,
        description="if True, freeze the variational orbitals for the duration of the calculation once they've been initialized",
    )
    calculate_bands: bool = Field(
        default=False,
        description="Calculate the band structure of the system along `kpoints.path`. A "
        "$\\Delta$SCF singlepoint computes on a supercell, so its band structure is "
        "recovered by unfolding the Koopmans Hamiltonian in the Wannier basis and "
        "interpolating it, which this switch asks for; it requires "
        "`init_orbitals = 'mlwfs'` or `'projwfs'`. The `dft_bands` and `wannierize` "
        "tasks compute bands whenever the input names a path, and need no switch",
    )
    spin: SpinType = Field(
        default=SpinType.NONE,
        description="how to treat the spin degrees of freedom: 'none' (spin-unpolarized), "
        "'collinear' (the system may break spin symmetry i.e. $n^{up}(r) != n^{down}(r)$, and "
        "``calculator_parameters.tot_magnetization`` states by how much), "
        "'non_collinear' (spinor wavefunctions), or 'spin_orbit' (spinor wavefunctions with "
        "spin-orbit coupling)",
    )
    initialize_with_smearing: bool = Field(
        default=False,
        description="if True, the first step of the workflow will use smearing. This can help convergence in some difficult cases.",
    )
    fix_spin_contamination: bool = Field(
        default=False,
        description="if True, steps will be taken to try and avoid spin contamination. This is only sensible when performing a non-spin-polarized calculation, and is turned on by default for such calculations",
    )
    gb_correction: bool | None = Field(
        default=None,
        description="if True, apply the Gygi-Baldereschi scheme to deal with the q->0 divergence of the Coulomb interation for periodic systems",
    )
    mp_correction: bool | None = Field(
        default=None,
        description="if True, apply the Makov-Payne correction for charged periodic systems",
    )
    mt_correction: bool | None = Field(
        default=None,
        description="if True, apply the Martyna-Tuckerman correction for charged aperiodic systems",
    )
    eps_inf: FloatGE1 | str | None = Field(
        default=None,
        description='dielectric constant of the system used by the Gygi-Baldereschi and Makov-Payne corrections; either provide an explicit value or set to "auto" to calculate it ab initio',
    )
    alpha_numsteps: int = Field(default=10, description="Number of steps for alpha calculation")
    alpha_conv_thr: float = Field(
        default=1e-3,
        description="convergence threshold for $|Delta E_i - epsilon_i|$; if below this threshold, the corresponding alpha value is not updated",
    )
    alpha_guess: float | list[float] = Field(
        default=0.6, description="starting guess for alpha (overridden if alpha_from_file is true)"
    )
    alpha_mixing: float = Field(default=1.0, description="mixing parameter for updating alpha")
    alpha_from_file: bool = Field(
        default=False,
        description="if True, uses the file_alpharef.txt from the base directory as a starting guess",
    )
    orbital_groups: list[list[int]] | None = Field(
        default=None,
        description="a list of integers the same length as the total number of bands, denoting which bands to assign the same screening parameter to",
    )
    group_orbitals_by: GroupOrbitalsBy | None = Field(
        default=None,
        description='criterion for grouping orbitals so they share a screening parameter: "self_hartree" (energies within group_orbitals_tol, in eV), "spread" (wannier90 spreads within group_orbitals_tol, in Angstrom^2), or "none". The criterion is independent of the screening method, though not every combination is wired up yet (currently self_hartree on DSCF and spread on DFPT). Left unset, resolves to "self_hartree" for Wannier-initialised DSCF runs (supercell images of one primitive orbital are physically equivalent) and "none" otherwise; the resolved value is recorded on the parsed input',
    )
    group_orbitals_tol: float | None = Field(
        default=None,
        description="tolerance for the group_orbitals_by criterion (units set by the criterion, e.g. eV for self_hartree, Angstrom^2 for spread). Left unset, takes the criterion's default (1e-4 for self_hartree, 0.05 for spread)",
    )
    dfpt_coarse_grid: tuple[int, int, int] | None = Field(
        default=None,
        description="The coarse k-point grid on which to perform the DFPT calculations",
    )
    block_wannierization_threshold: float | None = Field(
        default=None,
        description="blocks of bands separated by this threshold will be Wannierized separately",
    )
    auto_projections: bool = Field(
        default=False,
        description="if True, derive the Wannier projections automatically from the "
        "pseudopotentials' atomic orbitals (or, if `pw2wannier90.atom_proj_ext` is set, "
        "from external projector files) instead of requiring explicit projections in "
        "`calculator_parameters.w90.projections`. Setting `pw2wannier90.atom_proj_ext` "
        "requires this too: the projector files choose where the projector functions "
        "come from, not whether the projections are derived automatically",
    )
    max_time: int | float | None = Field(
        default=None,
        description="maximum time in seconds to wait for the workflow to complete; if None, no timeout is applied",
    )
    wait_time: int | float = Field(
        default=5,
        description="time in seconds to wait between checking the status of in-progress calculations",
    )

    @field_validator(
        "task",
        "correction",
        "screening_method",
        "init_orbitals",
        "init_empty_orbitals",
        mode="before",
    )
    @classmethod
    def make_lowercase(cls, v: Any) -> Any:
        """Convert string to lowercase."""
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("orbital_groups", mode="before")
    @classmethod
    def ensure_orbital_groups_is_list_of_lists(cls, v: Any) -> Any:
        """Convert a flat list to a list of lists for orbital_groups."""
        if v is not None:
            if len(v) == 0 or not isinstance(v[0], list):
                v = [v]
        return v

    @model_validator(mode="before")
    @classmethod
    def empty_variational_orbitals_default_to_same_as_filled(
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        """If init_empty_orbitals is not specified, set it to the same value as init_orbitals."""
        if values.get("init_empty_orbitals", None) is None:
            values["init_empty_orbitals"] = values.get("init_orbitals", VariationalOrbitalType.PZ)
        return values

    @model_validator(mode="after")
    def check_orbital_groups_length(self) -> Self:
        """Make the spin-dimension of ``orbital_groups`` is consistent with ``spin``."""
        if self.orbital_groups is not None:
            target_length = 2 if self.spin == SpinType.COLLINEAR else 1
            if len(self.orbital_groups) != target_length:
                raise ValueError(f"'orbital_groups' should be of length {target_length}")
        return self

    @model_validator(mode="after")
    def resolve_orbital_grouping(self) -> Self:
        """Resolve the orbital-grouping criterion and tolerance.

        Left unset, ``group_orbitals_by`` becomes ``self_hartree`` for
        Wannier-initialised DSCF runs — supercell images of one primitive
        orbital are physically equivalent and must share a screening
        parameter — and ``none`` otherwise (grouping is opt-in elsewhere).
        Tolerances default per criterion (``self_hartree``: 1e-4 eV;
        ``spread``: 0.05 Å²); a tolerance combined with ``none``, or without
        a criterion, is an error. Resolving here keeps the effective values
        visible on the parsed input. The criterion is in principle
        independent of the screening method — the defaults simply reflect
        the combinations wired up today, and the dispatcher rejects the
        rest explicitly.
        """
        if self.group_orbitals_by is None:
            wannier_init = self.init_orbitals in (
                VariationalOrbitalType.MLWFS,
                VariationalOrbitalType.PROJWFS,
            )
            dscf = self.screening_method == CalculateScreeningMethod.DSCF
            self.group_orbitals_by = (
                GroupOrbitalsBy.SELF_HARTREE if (wannier_init and dscf) else GroupOrbitalsBy.NONE
            )
        if self.group_orbitals_by == GroupOrbitalsBy.NONE:
            if self.group_orbitals_tol is not None:
                raise ValueError("group_orbitals_tol requires group_orbitals_by != 'none'")
        elif self.group_orbitals_tol is None:
            default_tol = {
                GroupOrbitalsBy.SELF_HARTREE: 1.0e-4,
                GroupOrbitalsBy.SPREAD: 0.05,
            }.get(self.group_orbitals_by)
            # Assigning ``None`` back would re-trigger this validator forever
            # (validate_assignment), so criteria without a default keep None.
            if default_tol is not None:
                self.group_orbitals_tol = default_tol
        return self
