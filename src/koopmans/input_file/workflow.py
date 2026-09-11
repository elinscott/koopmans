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

#: Keywords retired from ``workflow``, each with what to write instead.
_REMOVED_WORKFLOW_KEYWORDS: dict[str, str] = {
    "calculate_bands": (
        "A band structure is computed whenever `kpoints.path` names a path and the "
        "task can interpolate along it; set `kpoints.path`."
    ),
}

#: Default tolerance per grouping criterion, applied when
#: ``group_orbitals_by`` is active (explicit or resolved) but
#: ``group_orbitals_tol`` is unset.
_ORBITAL_GROUPING_DEFAULT_TOLERANCE: dict[str, float] = {
    "self_hartree": 1.0e-4,
    "spread": 0.05,
}


class Task(Enum):
    """Valid tasks that ``koopmans`` can perform."""

    SINGLEPOINT = "singlepoint"
    CONVERGENCE = "convergence"
    WANNIERIZE = "wannierize"
    DFT_BANDS = "dft_bands"
    DFT_EPS = "dft_eps"
    TRAJECTORY = "trajectory"
    BSE = "bse"


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
    alpha_numsteps: int = Field(
        default=1, description="maximum number of self-consistency steps for calculating alpha"
    )
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
    group_orbitals_by: GroupOrbitalsBy = Field(
        description='criterion for grouping orbitals so they share a screening parameter: "self_hartree" (energies within group_orbitals_tol, in eV), "spread" (wannier90 spreads within group_orbitals_tol, in Angstrom^2), or "none". The criterion is independent of the screening method, though not every combination is wired up yet (currently self_hartree on DSCF and spread on DFPT). Resolved at parse time from init_orbitals/screening_method when unset: "self_hartree" for Wannier-initialized DSCF runs (supercell images of one primitive orbital are physically equivalent), "none" otherwise; the parsed input always carries the resolved value',
    )
    group_orbitals_tol: float | None = Field(
        default=None,
        description="tolerance for the group_orbitals_by criterion (units set by the criterion, e.g. eV for self_hartree, Angstrom^2 for spread). Left unset, resolved at parse time to the criterion's default (1e-4 for self_hartree, 0.05 for spread) whenever group_orbitals_by is active; stays unset when group_orbitals_by resolves to 'none'",
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
    def reject_removed_workflow_keywords(cls, data: Any) -> Any:
        """Point a retired ``workflow`` keyword at what replaced it.

        Runs before field validation, so it reports the removed spelling
        instead of the generic "extra_forbidden" error.

        Raises:
            ValueError: If a removed key is present.
        """
        if not isinstance(data, dict):
            return data
        for key, replacement in _REMOVED_WORKFLOW_KEYWORDS.items():
            if key in data:
                raise ValueError(f"`workflow.{key}` no longer exists. {replacement}")
        return data

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

    @model_validator(mode="before")
    @classmethod
    def resolve_group_orbitals_by(cls, data: Any) -> Any:
        """Resolve ``group_orbitals_by``/``group_orbitals_tol`` on the raw input.

        An explicit ``group_orbitals_by: 'none'`` next to
        ``group_orbitals_tol`` directly contradicts it and is rejected here,
        on the raw dict, before either field is validated. Left unset,
        ``group_orbitals_by`` resolves to ``self_hartree`` for
        Wannier-initialized DSCF runs — supercell images of one primitive
        orbital are physically equivalent and must share a screening
        parameter — and ``none`` otherwise (grouping is opt-in elsewhere);
        an explicit criterion passes through unchanged. Once a criterion is
        active (explicit or resolved), an unset tolerance takes the
        criterion's default; ``none`` never defaults a tolerance, so a
        tolerance next to a criterion that *resolved* to (rather than was
        typed as) ``none`` survives for
        :func:`koopmans.aiida.workflows.advisories_for` to flag instead,
        since only the dispatcher knows which routes group orbitals at all.

        Args:
            data: The raw value pydantic is validating — a dict for a
                normal parse, but possibly something else on re-validation;
                returned untouched when it is not a dict.

        Raises:
            ValueError: If an explicit group_orbitals_by == 'none' accompanies group_orbitals_tol.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)

        def _value(v: Any, default: str) -> str:
            if v is None:
                return default
            return v.value if isinstance(v, Enum) else str(v).lower()

        raw_tol = data.get("group_orbitals_tol")
        raw_criterion = data.get("group_orbitals_by")
        if raw_criterion is not None:
            criterion = _value(raw_criterion, GroupOrbitalsBy.NONE.value)
            if criterion == GroupOrbitalsBy.NONE.value and raw_tol is not None:
                raise ValueError("group_orbitals_tol requires group_orbitals_by != 'none'")
        else:
            wannier_init = _value(data.get("init_orbitals"), VariationalOrbitalType.PZ.value) in (
                VariationalOrbitalType.MLWFS.value,
                VariationalOrbitalType.PROJWFS.value,
            )
            dscf = (
                _value(data.get("screening_method"), CalculateScreeningMethod.DSCF.value)
                == CalculateScreeningMethod.DSCF.value
            )
            criterion = (
                GroupOrbitalsBy.SELF_HARTREE.value
                if (wannier_init and dscf)
                else GroupOrbitalsBy.NONE.value
            )
            data["group_orbitals_by"] = criterion

        if raw_tol is None and criterion in _ORBITAL_GROUPING_DEFAULT_TOLERANCE:
            data["group_orbitals_tol"] = _ORBITAL_GROUPING_DEFAULT_TOLERANCE[criterion]
        return data
