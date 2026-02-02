"""Input schema for atomic positions."""

from typing import Literal

from koopmans.base import BaseModel


class AtomicPositionsInput(BaseModel):
    """Input schema for specifying atomic positions in a structure."""

    positions: list[tuple[str, float, float, float]]
    units: Literal["crystal", "ang", "bohr", "alat"] = "alat"
