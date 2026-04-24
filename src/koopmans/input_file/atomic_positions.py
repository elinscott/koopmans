"""Input schema for atomic positions."""

from typing import Annotated, Literal

from pydantic import BeforeValidator

from koopmans.base import BaseModel
from koopmans.input_file._utils import tidy_units


class AtomicPositionsInput(BaseModel):
    """Input schema for specifying atomic positions in a structure."""

    positions: list[tuple[str, float, float, float]]
    units: Annotated[Literal["crystal", "ang", "bohr", "alat"], BeforeValidator(tidy_units)] = (
        "alat"
    )
