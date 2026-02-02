from typing import Annotated, Literal

from pydantic import BeforeValidator, model_validator

from koopmans.base import BaseModel


def tidy_units(value: str) -> str:
    value = value.lower()
    value = value.replace("angstrom", "ang")
    return value


class CellParametersABC(BaseModel):
    periodic: bool | tuple[bool, bool, bool] = True


class CellParametersViaIbrav(CellParametersABC):
    ibrav: int
    celldms: dict[int, float]

class CellParametersViaAlat(CellParametersABC):
    celldms: dict[int, float]
    vectors: list[tuple[float, float, float]]
    units: Annotated[Literal['alat'], BeforeValidator(tidy_units)] = 'alat'


class CellParametersViaVectors(CellParametersABC):

    vectors: list[tuple[float, float, float]]
    units: Annotated[Literal['bohr', 'ang'], BeforeValidator(tidy_units)] = 'ang'


class CellParametersInput(BaseModel):
    periodic: bool = True
    ibrav: int | None = None
    celldms: dict[int, float] | None = None
    vectors: list[tuple[float, float, float]] | None = None
    units: Annotated[Literal['bohr', 'ang', 'alat'], BeforeValidator(tidy_units)] = 'ang'

    @model_validator(mode="after")
    def check_ibrav_and_celldms_xor_vectors(self):
        if (self.ibrav is None) != (self.celldms is None):
            raise ValueError("'ibrav' and 'celldms' must be provided together.")
        if (self.ibrav is None) == (self.vectors is None):
            raise ValueError("Specify either both 'ibrav' and 'celldms', or 'vectors'.")
        return self
