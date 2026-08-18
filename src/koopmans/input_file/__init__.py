"""Input file schema for `koopmans`."""

from __future__ import annotations

from collections.abc import Callable
from json import load
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, Field, ValidationError, field_validator, model_validator
from pydantic_core import ErrorDetails
from wannier90_input.models.parameters import Projection
from yaml import safe_load

from koopmans.base import BaseModel
from koopmans.input_file.atomic_positions import AtomicPositionsInput
from koopmans.input_file.cell_parameters import (
    CellParametersViaAlat,
    CellParametersViaIbrav,
    CellParametersViaVectors,
)
from koopmans.input_file.kcp import KCPInputParameters
from koopmans.input_file.ml import MLConfig
from koopmans.input_file.parallelization import ParallelizationInput
from koopmans.input_file.ph import PHInputParameters
from koopmans.input_file.pw import PWInputParameters
from koopmans.input_file.pw2wannier90 import PW2Wannier90InputParameters
from koopmans.input_file.unfold_and_interpolate import UnfoldAndInterpolateConfig
from koopmans.input_file.wannier90 import RestrictedWannier90InputParameters
from koopmans.input_file.workflow import WorkflowConfig

# The public schema surface. The documentation renders this list, so a name
# absent from it is undocumented however it reaches the module namespace;
# `Projection` and the per-calculator models are re-exports and belong here.
__all__ = [
    "INPUT_FILE_FORMAT_VERSION",
    "AtomicPositionsInput",
    "AtomsInput",
    "CalculatorParametersInput",
    "CellParametersViaAlat",
    "CellParametersViaIbrav",
    "CellParametersViaVectors",
    "GammaOnlyKpointsInput",
    "GridKpointsInput",
    "KCPInputParameters",
    "KoopmansInput",
    "KpointOffset",
    "KpointsOverridesInput",
    "MLConfig",
    "NoOffset",
    "PHInputParameters",
    "PW2Wannier90InputParameters",
    "PWInputParameters",
    "ParallelizationInput",
    "Projection",
    "RestrictedWannier90InputParameters",
    "SpinSpecificWannierInput",
    "StepKpointsOverridesInput",
    "UnfoldAndInterpolateConfig",
    "Wannier90InputParametersWithUpDown",
    "WannierKpointsOverridesInput",
    "WorkflowConfig",
    "migrate_input_dict",
    "read_input_file",
]

INPUT_FILE_FORMAT_VERSION = 1
"""Current version of the input file format.

Bump this (and register a migration in ``_MIGRATIONS``) only when the format
changes incompatibly. Adding new optional fields does not require a bump.
"""

_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}
"""Migrations between input file format versions.

``_MIGRATIONS[n]`` takes a raw input dict in format version ``n`` and returns
the equivalent dict in format version ``n + 1``. Migrations are applied in
sequence by :func:`migrate_input_dict` before validation, so the Pydantic
models only ever describe the current format.
"""


def migrate_input_dict(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a raw input dict to the current input file format version.

    A missing ``version`` key is treated as version 1 (the format predates
    the key).

    Args:
        input_dict: The raw input file contents.

    Returns:
        The input dict, upgraded to ``INPUT_FILE_FORMAT_VERSION``.

    Raises:
        ValueError: If the version is invalid or newer than this version of
            ``koopmans`` supports.
    """
    version = input_dict.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"`version` must be a positive integer, not `{version!r}`")
    if version > INPUT_FILE_FORMAT_VERSION:
        raise ValueError(
            f"This input file uses format version {version}, but this version of `koopmans` "
            f"only supports up to version {INPUT_FILE_FORMAT_VERSION}. Please upgrade `koopmans`."
        )
    for v in range(version, INPUT_FILE_FORMAT_VERSION):
        input_dict = _MIGRATIONS[v](input_dict)
    return {**input_dict, "version": INPUT_FILE_FORMAT_VERSION}


class AtomsInput(BaseModel):
    """Input model for specifying the cell and atomic positions."""

    cell_parameters: CellParametersViaIbrav | CellParametersViaVectors | CellParametersViaAlat
    atomic_positions: AtomicPositionsInput | None = None
    snapshots: str | None = None
    """Path to a multi-frame ``xyz`` file (one structure per frame)."""

    @model_validator(mode="after")
    def _exactly_one_positions_source(self) -> AtomsInput:
        """Require exactly one of ``atomic_positions`` and ``snapshots``."""
        if (self.atomic_positions is None) == (self.snapshots is None):
            raise ValueError(
                "the `atoms` block must contain exactly one of `atomic_positions` "
                "(explicit positions) and `snapshots` (a multi-frame xyz path)"
            )
        if self.snapshots is not None and not self.snapshots.strip():
            raise ValueError("`snapshots` must be a non-empty path to a multi-frame xyz file")
        return self


def _expressible_shift(value: float) -> float:
    """Check one axis carries a shift a k-point mesh can express.

    The offset is a fraction of a grid step, so only 0 and 0.5 mean
    anything: Quantum ESPRESSO's ``K_POINTS automatic`` card has no way to
    write any other shift. A whole step of 1 is called out separately — it
    is arithmetically the same as no shift, so anyone writing it means the
    opposite of what they would get.
    """
    if value == 1:
        raise ValueError(
            "`offset` is a fraction of a grid step, not a flag: 1 is a whole "
            "grid step, which lands back on the unshifted mesh and so means "
            "exactly the same as 0. Write 0.5 for a half-shifted mesh."
        )
    if value not in (0, 0.5):
        raise ValueError(
            "each component of `offset` must be 0 (unshifted) or 0.5 (shifted "
            f"by half a grid step); got {value}"
        )
    return value


#: One axis of a k-point mesh offset: 0 leaves it Gamma-centred, 0.5
#: half-shifts it, and nothing else is expressible.
KpointOffset = Annotated[float, AfterValidator(_expressible_shift)]


def _no_shift(value: float) -> float:
    """Reject any shift: a shifted mesh no longer samples Gamma."""
    if value != 0:
        raise ValueError(
            "a gamma_only calculation samples Gamma itself, so `offset` must be "
            "(0, 0, 0); use a grid to sample anywhere else"
        )
    return value


#: One axis of a gamma-only offset. ``Literal[0.0]`` cannot express this:
#: PEP 586 admits int, str, bytes, bool, Enum and None, but not float.
NoOffset = Annotated[float, AfterValidator(_no_shift)]


class StepKpointsOverridesInput(BaseModel):
    """K-point sampling for one step, in place of the top-level values.

    Every attribute is absolute and every one left unset is taken from the
    top-level ``kpoints``.
    """

    grid: tuple[int, int, int] | None = None
    """Monkhorst-Pack dimensions of the mesh this step samples."""

    offset: tuple[KpointOffset, KpointOffset, KpointOffset] | None = None
    """Per-axis fraction of a grid step to shift this step's mesh by.

    Available on the ``scf`` entry alone.
    """

    grid_spacing: float | None = Field(default=None, gt=0.0)
    """Largest spacing between neighbouring k-points, in inverse angstrom.

    The cell fixes the mesh dimensions, so a converged value carries from
    one structure to the next. Excludes ``grid`` and ``offset``.
    """

    @model_validator(mode="after")
    def _one_statement_of_the_mesh(self) -> StepKpointsOverridesInput:
        """Require ``grid_spacing`` to be the entry's only statement of the mesh."""
        if self.grid_spacing is None:
            return self
        for name in ("grid", "offset"):
            if getattr(self, name) is not None:
                raise ValueError(
                    f"`grid_spacing` and `{name}` both describe the same mesh: "
                    "`grid_spacing` lets the cell fix the mesh, so it cannot be "
                    f"combined with `{name}`. Give one of the two."
                )
        return self


class WannierKpointsOverridesInput(BaseModel):
    """The k-point path wannier90 interpolates its band structure along.

    Carries a density alone: the path itself, and its special points, are
    the same ones the route's own k-path uses, so the interpolated bands
    stay lined up against the diagonalized ones the plot overlays them with.
    """

    path_density: float = 50.0
    """Number of k-points per inverse angstrom (2π convention) wannier90
    interpolates its band structure at.

    Independent of the top-level ``path_density``: wannier90's interpolation
    is nearly free once the Wannier functions are built, so it defaults far
    denser than a diagonalized pw.x bands run would.
    """


class KpointsOverridesInput(BaseModel):
    """K-point sampling for individual steps, in place of the top-level values.

    Steps left out sample the top-level ``grid`` and ``offset``.
    """

    scf: StepKpointsOverridesInput | None = None
    """The mesh the ground-state calculation converges the density on."""

    nscf: StepKpointsOverridesInput | None = None
    """The Gamma-centred mesh the Wannier functions are built from."""

    wannier90: WannierKpointsOverridesInput | None = None
    """The density wannier90 interpolates its band structure at.

    Unset takes :attr:`WannierKpointsOverridesInput.path_density`'s default.
    """

    @model_validator(mode="after")
    def _nscf_states_a_mesh_koopmans_builds(self) -> KpointsOverridesInput:
        """Restrict the nscf entry to the meshes the Wannierization runs on.

        The nscf mesh is wannier90's ``mp_grid``, which a spacing does not
        state, and koopmans builds it Gamma-centred.
        """
        if self.nscf is None:
            return self
        if self.nscf.grid_spacing is not None:
            raise ValueError(
                "`nscf.grid_spacing` cannot be used: the nscf mesh dimensions are "
                "wannier90's `mp_grid`, which a spacing does not state. Give "
                "`nscf.grid` instead."
            )
        if self.nscf.offset is not None:
            raise ValueError(
                "`nscf.offset` is not supported: the nscf mesh koopmans builds is "
                "Gamma-centred. An offset belongs on the `scf` entry, and the "
                "top-level `kpoints.offset` already applies to the scf."
            )
        return self


class GammaOnlyKpointsInput(BaseModel):
    """K-points configuration for gamma-only calculations."""

    gamma_only: Literal[True] = True
    grid: tuple[Literal[1], Literal[1], Literal[1]] = (1, 1, 1)
    offset: tuple[NoOffset, NoOffset, NoOffset] = (0.0, 0.0, 0.0)
    """A gamma-only calculation samples Gamma itself, so it cannot be shifted."""

    path: Literal["G"] = "G"
    path_density: float = 10.0
    """Number of k-points per inverse angstrom (2π convention) along ``path``.

    Measured in the same Cartesian reciprocal basis as ``grid_spacing``, so a
    converged value carries from one structure to the next.
    """

    overrides: KpointsOverridesInput = Field(default_factory=KpointsOverridesInput)
    """Per-step k-point sampling, which a gamma-only calculation cannot have."""

    @field_validator("overrides")
    @classmethod
    def check_no_step_is_given_a_mesh(
        cls, overrides: KpointsOverridesInput
    ) -> KpointsOverridesInput:
        """Reject a per-step mesh: every step of a gamma-only run samples Gamma."""
        for step in ("scf", "nscf"):
            if getattr(overrides, step) is not None:
                raise ValueError(
                    f"`overrides.{step}` cannot be used together with `gamma_only`, whose "
                    "every step samples Gamma alone. Give a `grid` instead of `gamma_only`."
                )
        if overrides.wannier90 is not None:
            raise ValueError(
                "`overrides.wannier90` cannot be used together with `gamma_only`: every "
                "step samples Gamma alone, so there is no band structure to interpolate. "
                "Give a `grid` instead of `gamma_only`."
            )
        return overrides


class GridKpointsInput(BaseModel):
    """K-points configuration for calculations with explicit grid."""

    gamma_only: Literal[False] = False
    grid: tuple[int, int, int]
    offset: tuple[KpointOffset, KpointOffset, KpointOffset] = (0.0, 0.0, 0.0)
    """Per-axis fraction of a grid step to shift the mesh by."""

    path: str | None = None
    path_density: float = 10.0
    """Number of k-points per inverse angstrom (2π convention) along ``path``.

    Measured in the same Cartesian reciprocal basis as ``grid_spacing``, so a
    converged value carries from one structure to the next.
    """

    overrides: KpointsOverridesInput = Field(default_factory=KpointsOverridesInput)
    """Per-step k-point sampling, in place of ``grid`` and ``offset``."""


KpointsInput = GammaOnlyKpointsInput | GridKpointsInput


class SpinSpecificWannierInput(BaseModel):
    """Spin-specific Wannier90 input parameters."""

    dis_froz_max: float | None = None
    dis_froz_min: float | None = None
    dis_win_max: float | None = None
    dis_win_min: float | None = None
    projections: list[list[Projection]] = Field(default_factory=list)


class Wannier90InputParametersWithUpDown(RestrictedWannier90InputParameters):
    """Wannier90 input parameters with optional spin-up/spin-down configuration."""

    up: SpinSpecificWannierInput | None = None
    down: SpinSpecificWannierInput | None = None

    @model_validator(mode="after")
    def check_up_down_exclusivity(self) -> Wannier90InputParametersWithUpDown:
        """Validate that up and down are both specified or both omitted."""
        if (self.up is None) != (self.down is None):
            raise ValueError("Both 'up' and 'down' must be specified together.")
        return self


class CalculatorParametersInput(BaseModel):
    """Calculator-specific input parameters."""

    ecutwfc: float | None = Field(default=None, gt=0.0)
    nbnd: int | None = None
    tot_magnetization: float | None = None
    ph: PHInputParameters = Field(default_factory=lambda: PHInputParameters())
    pw: PWInputParameters = Field(default_factory=lambda: PWInputParameters())
    pw2wannier90: PW2Wannier90InputParameters = Field(
        default_factory=lambda: PW2Wannier90InputParameters()
    )
    wannier90: Wannier90InputParametersWithUpDown = Field(
        default_factory=lambda: Wannier90InputParametersWithUpDown()  # type: ignore[call-arg]
    )
    unfold_and_interpolate: UnfoldAndInterpolateConfig = Field(
        default_factory=lambda: UnfoldAndInterpolateConfig()
    )
    kcp: KCPInputParameters = Field(default_factory=lambda: KCPInputParameters())

    @model_validator(mode="before")
    @classmethod
    def reject_removed_per_calculator_cutoffs(cls, data: Any) -> Any:
        """Point a per-calculator cutoff at the single ``ecutwfc`` field.

        ``pw.system``/``kcp.system`` no longer carry their own
        ``ecutwfc``/``ecutrho``: pw.x and kcp.x always share one grid, derived
        from ``calculator_parameters.ecutwfc``. Runs before field validation,
        so it reports the removed spelling instead of the generic
        "extra_forbidden" error the nested model would otherwise raise.

        Raises:
            ValueError: If any of the four removed keys is present.
        """
        if not isinstance(data, dict):
            return data
        for calc in ("pw", "kcp"):
            system = data.get(calc)
            if not isinstance(system, dict) or not isinstance(system.get("system"), dict):
                continue
            for key in ("ecutwfc", "ecutrho"):
                if key in system["system"]:
                    raise ValueError(
                        f"`calculator_parameters.{calc}.system.{key}` no longer exists. "
                        "Set `calculator_parameters.ecutwfc`; `ecutrho` follows at four "
                        "times it."
                    )
        return data


class KoopmansInput(BaseModel):
    """Input schema for ``koopmans`` input files."""

    version: int = Field(
        default=INPUT_FILE_FORMAT_VERSION,
        description="Version of the input file format (older files are upgraded "
        "automatically when loaded from disk)",
    )
    workflow: WorkflowConfig = Field(
        description="Configuration specifying the workflow to be executed"
    )
    atoms: AtomsInput = Field(description="Atomic structure information")
    kpoints: KpointsInput = Field(
        default_factory=GammaOnlyKpointsInput,
        description="k-point sampling information",
    )
    calculator_parameters: CalculatorParametersInput = Field(
        description="Parameters for the individual electronic structure calculators (``pw.x``, etc...)"
    )
    ml: MLConfig = Field(
        default_factory=lambda: MLConfig(),
        description="Machine-learning configuration for predicting screening parameters",
    )
    parallelization: ParallelizationInput = Field(
        default_factory=ParallelizationInput,
        description="Per-code parallelization settings (MPI ranks and k-point pools)",
    )

    @field_validator("kpoints", mode="before")
    @classmethod
    def check_density_was_renamed(cls, kpoints: Any) -> Any:
        """Reject the former ``density`` spelling of ``path_density``.

        ``KpointsInput`` is an untagged union, so a check inside either
        member is reported against both. This one names a keyword the two
        share, so it belongs where the field does.
        """
        if isinstance(kpoints, dict) and "density" in kpoints:
            raise ValueError(
                "`density` has been renamed `path_density`: it counts k-points per "
                "unit length along `path`, and sitting beside `grid` it reads like "
                "the density of a mesh. Rename it."
            )
        return kpoints

    @field_validator("version")
    @classmethod
    def check_version_is_current(cls, version: int) -> int:
        """Validate that the version matches the current format version.

        The model always describes the current format; older files are
        upgraded by :func:`migrate_input_dict` before they reach validation.
        """
        if version != INPUT_FILE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported input file format version {version} (this version of `koopmans` "
                f"uses version {INPUT_FILE_FORMAT_VERSION}; files loaded via `read_input_file` "
                "are upgraded automatically)"
            )
        return version

    @classmethod
    def from_file(cls, filename: str | Path) -> KoopmansInput:
        """Load an input file and return a KoopmansInput object."""
        filename = Path(filename)
        if filename.suffix in {".yaml", ".yml"}:
            with open(filename) as f:
                input_dict = safe_load(f)
        elif filename.suffix == ".json":
            with open(filename) as f:
                input_dict = load(f)
        else:
            raise ValueError(f"Unrecognized file type for `{filename}`")

        koopmans_input = cls.model_validate(migrate_input_dict(input_dict))
        koopmans_input.resolve_paths(filename.parent)
        return koopmans_input

    def resolve_paths(self, base_dir: str | Path) -> None:
        """Resolve file-path input fields against the input file's directory.

        ``ml.model_file`` and ``atoms.snapshots`` both name files on disk. A
        relative path in the input file is interpreted relative to the file's
        own location, not the process working directory. Absolute paths are
        left untouched.
        """
        base = Path(base_dir)

        def _resolve(path: str) -> str:
            candidate = Path(path)
            return str(candidate if candidate.is_absolute() else base / candidate)

        if self.ml.model_file is not None:
            self.ml.model_file = _resolve(self.ml.model_file)

        if self.atoms.snapshots is not None:
            self.atoms.snapshots = _resolve(self.atoms.snapshots)


CUSTOM_MESSAGES = {
    "type": 'is the wrong type (should be "{expected_type}", not "{given_type}")',
    "extra_forbidden": "is not a valid keyword.",
    "missing": "was not provided.",
    "greater_than": "must be greater than {gt}.",
}


def convert_errors(e: ValidationError) -> list[ErrorDetails]:
    """Make the validation errors more user-friendly."""
    new_errors: list[ErrorDetails] = []
    for error in e.errors():
        custom_message = CUSTOM_MESSAGES.get(error["type"], None)
        if custom_message:
            ctx = error.get("ctx")
            error["msg"] = custom_message.format(**ctx) if ctx else custom_message
        new_errors.append(error)
    return new_errors


def prettify_errors(e: ValidationError) -> str:
    """Return a prettified string of validation errors."""
    errors = convert_errors(e)
    error_lines = []
    for error in errors:
        loc = ".".join(str(part) for part in error["loc"])
        msg = error["msg"]
        error_lines.append(f" `{loc}` {msg}")
    return "\n".join(error_lines)


def read_input_file(filename: str | Path) -> KoopmansInput:
    """Read and parse a ``koopmans`` input file.

    Args:
        filename: Path to the input file (JSON or YAML format).

    Returns:
        Parsed ``KoopmansInput`` object.

    Raises:
        ValueError: If the input file contains validation errors.
    """
    try:
        koopmans_input = KoopmansInput.from_file(filename)
    except ValidationError as e:
        raise ValueError(
            f"Errors found in the input file: \n\n{prettify_errors(e)}\n\n"
            "For more information, see URL_HERE."
        ) from e
    return koopmans_input
