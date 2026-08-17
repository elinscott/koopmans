"""Input schema for cell parameters."""

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BeforeValidator, Field

from koopmans.base import BaseModel
from koopmans.input_file._utils import tidy_units

__all__ = [
    "CellParametersBase",
    "CellParametersViaAlat",
    "CellParametersViaIbrav",
    "CellParametersViaVectors",
    "Celldms",
    "Periodic",
]


def _require_celldm1(celldms: dict[int, float]) -> dict[int, float]:
    """Require celldm(1), which sets the length scale of the cell."""
    if 1 not in celldms:
        raise ValueError("'celldms' must include celldm(1) (the lattice parameter in Bohr)")
    return celldms


Celldms = Annotated[dict[int, float], AfterValidator(_require_celldm1)]


def _one_per_cell_vector(periodic: Any) -> Any:
    """Expand a single ``periodic`` bool to one entry per cell vector."""
    if isinstance(periodic, bool):
        return (periodic, periodic, periodic)
    return periodic


Periodic = Annotated[
    tuple[bool, bool, bool],
    BeforeValidator(_one_per_cell_vector, json_schema_input_type=bool | tuple[bool, bool, bool]),
]
"""Periodicity per cell vector, which a single bool may state for all three."""


class CellParametersBase(BaseModel):
    """Shared base for the cell parameter specification variants."""

    periodic: Periodic = Field(
        default=(True, True, True),
        description="whether the cell repeats along each of its three vectors; write a "
        "single `true` or `false` to say the same of all three",
    )


class CellParametersViaIbrav(CellParametersBase):
    """Cell parameters specified via ``ibrav`` and ``celldms``."""

    ibrav: int
    celldms: Celldms


class CellParametersViaAlat(CellParametersBase):
    """Cell parameters specified via ``celldms`` and explicit vectors in ``alat`` units."""

    celldms: Celldms
    vectors: list[tuple[float, float, float]]
    units: Annotated[Literal["alat"], BeforeValidator(tidy_units)] = "alat"


class CellParametersViaVectors(CellParametersBase):
    """Cell parameters specified via explicit vectors in ``bohr`` or ``ang`` units."""

    vectors: list[tuple[float, float, float]]
    units: Annotated[Literal["bohr", "ang"], BeforeValidator(tidy_units)] = "ang"
