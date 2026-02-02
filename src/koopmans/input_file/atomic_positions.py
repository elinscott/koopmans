from typing import Literal

from koopmans.base import BaseModel


class AtomicPositionsInput(BaseModel):
    positions: list[tuple[str, float, float, float]]
    units: Literal['crystal', 'ang', 'bohr', 'alat'] = 'alat'
