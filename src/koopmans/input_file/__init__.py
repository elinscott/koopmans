"""Input file schema for `koopmans`."""

from __future__ import annotations

from json import load
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator
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
from koopmans.input_file.pw import PWInputParameters
from koopmans.input_file.pw2wannier90 import PW2Wannier90InputParameters
from koopmans.input_file.ui import UnfoldAndInterpolateConfig
from koopmans.input_file.wannier90 import RestrictedWannier90InputParameters
from koopmans.input_file.workflow import WorkflowConfig


class AtomsInput(BaseModel):
    """Input model for specifying the cell and atomic positions."""

    cell_parameters: CellParametersViaIbrav | CellParametersViaVectors | CellParametersViaAlat
    atomic_positions: AtomicPositionsInput


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


class Wannier90InputParametersWithUpDown(RestrictedWannier90InputParameters):  # type: ignore[misc]
    """Wannier90 input parameters with optional spin-up/spin-down configuration."""

    up: SpinSpecificWannierInput | None = None
    down: SpinSpecificWannierInput | None = None
    # In the input file, projections are specified as a list of lists to separate each block
    projections: list[list[Projection]] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_up_down_exclusivity(self) -> Wannier90InputParametersWithUpDown:
        """Validate that up and down are both specified or both omitted."""
        if (self.up is None) != (self.down is None):
            raise ValueError("Both 'up' and 'down' must be specified together.")
        return self


class CalculatorParametersInput(BaseModel):
    """Calculator-specific input parameters."""

    ecutwfc: float | None = None
    nbnd: float | None = None
    tot_magnetization: float | None = None
    pw: PWInputParameters = Field(default_factory=lambda: PWInputParameters())
    pw2wannier90: PW2Wannier90InputParameters = Field(
        default_factory=lambda: PW2Wannier90InputParameters()
    )
    wannier90: Wannier90InputParametersWithUpDown = Field(
        default_factory=lambda: Wannier90InputParametersWithUpDown()
    )
    ui: UnfoldAndInterpolateConfig = Field(default_factory=lambda: UnfoldAndInterpolateConfig())
    kcp: KCPInputParameters = Field(default_factory=lambda: KCPInputParameters())


class KoopmansInput(BaseModel):
    """Input schema for ``koopmans`` input files."""

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

        return cls.model_validate(input_dict)


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
