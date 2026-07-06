"""Input file schema for `koopmans`."""

from __future__ import annotations

from collections.abc import Callable
from json import load
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_core import ErrorDetails
from wannier90_input.models.parameters import Projection
from yaml import safe_load

from koopmans.base import BaseModel
from koopmans.input_file.atomic_positions import AtomicPositionsInput, SnapshotsInput
from koopmans.input_file.cell_parameters import (
    CellParametersViaAlat,
    CellParametersViaIbrav,
    CellParametersViaVectors,
)
from koopmans.input_file.kcp import KCPInputParameters
from koopmans.input_file.ml import MLConfig
from koopmans.input_file.pw import PWInputParameters
from koopmans.input_file.pw2wannier90 import PW2Wannier90InputParameters
from koopmans.input_file.unfold_and_interpolate import UnfoldAndInterpolateConfig
from koopmans.input_file.wannier90 import RestrictedWannier90InputParameters
from koopmans.input_file.workflow import Task, WorkflowConfig

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
    """Input model for specifying the cell and atomic positions.

    ``atomic_positions`` either lists the positions explicitly
    (:class:`AtomicPositionsInput`) or points at a multi-frame xyz
    trajectory via a ``snapshots`` key (:class:`SnapshotsInput`; only
    valid for the ``trajectory`` task).
    """

    cell_parameters: CellParametersViaIbrav | CellParametersViaVectors | CellParametersViaAlat
    atomic_positions: AtomicPositionsInput | SnapshotsInput


class GammaOnlyKpointsInput(BaseModel):
    """K-points configuration for gamma-only calculations."""

    gamma_only: Literal[True] = True
    grid: tuple[Literal[1], Literal[1], Literal[1]] = (1, 1, 1)
    offset: tuple[Literal[0], Literal[0], Literal[0]] = (0, 0, 0)
    path: Literal["G"] = "G"
    density: float = 10.0


class GridKpointsInput(BaseModel):
    """K-points configuration for calculations with explicit grid."""

    gamma_only: Literal[False] = False
    grid: tuple[int, int, int]
    offset: tuple[int, int, int] = (0, 0, 0)
    path: str | None = None
    density: float = 10.0


KpointsInput = GammaOnlyKpointsInput | GridKpointsInput


class SpinSpecificWannierInput(BaseModel):
    """Spin-specific Wannier90 input parameters."""

    dis_froz_max: float | None = None
    dis_froz_min: float | None = None
    dis_win_max: float | None = None
    dis_win_min: float | None = None
    auto_projections: bool = False
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

    ecutwfc: float | None = None
    nbnd: int | None = None
    tot_magnetization: float | None = None
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

    @model_validator(mode="after")
    def snapshots_require_trajectory_task(self) -> KoopmansInput:
        """Validate that a ``snapshots`` trajectory file is only used with ``task: trajectory``.

        Other tasks consume a single structure; restricting the snapshots
        input keeps their dispatch paths free of multi-frame handling.
        (The legacy code silently used only the first frame elsewhere.)
        """
        if (
            isinstance(self.atoms.atomic_positions, SnapshotsInput)
            and self.workflow.task != Task.TRAJECTORY
        ):
            raise ValueError(
                "`atoms.atomic_positions.snapshots` is only valid for `workflow.task = "
                f"'trajectory'`, not `{self.workflow.task.value}`. Provide explicit "
                "`atomic_positions` instead."
            )
        return self

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

        return cls.model_validate(migrate_input_dict(input_dict))


CUSTOM_MESSAGES = {
    "type": 'is the wrong type (should be "{expected_type}", not "{given_type}")',
    "extra_forbidden": "is not a valid keyword.",
    "missing": "was not provided.",
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
