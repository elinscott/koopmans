"""Input schema for cell parameters."""

from typing import Annotated, Literal

from pydantic import AfterValidator, BeforeValidator

from koopmans.base import BaseModel
from koopmans.input_file._utils import tidy_units


def _require_celldm1(celldms: dict[int, float]) -> dict[int, float]:
    """Require celldm(1), which sets the length scale of the cell."""
    if 1 not in celldms:
        raise ValueError("'celldms' must include celldm(1) (the lattice parameter in Bohr)")
    return celldms


Celldms = Annotated[dict[int, float], AfterValidator(_require_celldm1)]


class CellParametersBase(BaseModel):
    """Shared base for the cell parameter specification variants."""

    periodic: bool | tuple[bool, bool, bool] = True


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
