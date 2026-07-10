"""Input schema for cell parameters."""

from typing import Annotated, Literal

from pydantic import BeforeValidator

from koopmans.base import BaseModel
from koopmans.input_file._utils import tidy_units


class CellParametersBase(BaseModel):
    """Shared base for the cell parameter specification variants."""

    periodic: bool | tuple[bool, bool, bool] = True


class CellParametersViaIbrav(CellParametersBase):
    """Cell parameters specified via ``ibrav`` and ``celldms``."""

    ibrav: int
    celldms: dict[int, float]


class CellParametersViaAlat(CellParametersBase):
    """Cell parameters specified via ``celldms`` and explicit vectors in ``alat`` units."""

    celldms: dict[int, float]
    vectors: list[tuple[float, float, float]]
    units: Annotated[Literal["alat"], BeforeValidator(tidy_units)] = "alat"


class CellParametersViaVectors(CellParametersBase):
    """Cell parameters specified via explicit vectors in ``bohr`` or ``ang`` units."""

    vectors: list[tuple[float, float, float]]
    units: Annotated[Literal["bohr", "ang"], BeforeValidator(tidy_units)] = "ang"
