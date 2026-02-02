"""Input schema for cell parameters."""

from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, model_validator

from koopmans.base import BaseModel


def tidy_units(value: str) -> str:
    """Normalize unit strings to a canonical form."""
    value = value.lower()
    value = value.replace("angstrom", "ang")
    return value


class CellParametersABC(BaseModel):
    """Abstract base class for cell parameter specifications."""

    periodic: bool | tuple[bool, bool, bool] = True


class CellParametersViaIbrav(CellParametersABC):
    """Cell parameters specified via ``ibrav`` and ``celldms``."""

    ibrav: int
    celldms: dict[int, float]


class CellParametersViaAlat(CellParametersABC):
    """Cell parameters specified via ``celldms`` and explicit vectors in ``alat`` units."""

    celldms: dict[int, float]
    vectors: list[tuple[float, float, float]]
    units: Annotated[Literal["alat"], BeforeValidator(tidy_units)] = "alat"


class CellParametersViaVectors(CellParametersABC):
    """Cell parameters specified via explicit vectors in ``bohr`` or ``ang`` units."""

    vectors: list[tuple[float, float, float]]
    units: Annotated[Literal["bohr", "ang"], BeforeValidator(tidy_units)] = "ang"


class CellParametersInput(BaseModel):
    """Input schema for cell parameters with flexible specification methods."""

    periodic: bool = True
    ibrav: int | None = None
    celldms: dict[int, float] | None = None
    vectors: list[tuple[float, float, float]] | None = None
    units: Annotated[Literal["bohr", "ang", "alat"], BeforeValidator(tidy_units)] = "ang"

    @model_validator(mode="after")
    def check_ibrav_and_celldms_xor_vectors(self) -> Self:
        """Validate that cell is specified either via ``ibrav``/``celldms`` or ``vectors``."""
        if (self.ibrav is None) != (self.celldms is None):
            raise ValueError("'ibrav' and 'celldms' must be provided together.")
        if (self.ibrav is None) == (self.vectors is None):
            raise ValueError("Specify either both 'ibrav' and 'celldms', or 'vectors'.")
        return self
