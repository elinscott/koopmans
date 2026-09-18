"""Input parameters for ``kcw.x`` calculations."""

from pydantic import Field

from koopmans.base import BaseModel

# Generated (``just generate-input-models``): the keywords the workflow
# determines are absent from these, and nothing here is added to them.
from koopmans.input_file._generated.kcw import (
    ControlNamelist,
    HamNamelist,
    ScreenNamelist,
    WannierNamelist,
)

__all__ = [
    "ControlNamelist",
    "HamNamelist",
    "KCWInputParameters",
    "ScreenNamelist",
    "WannierNamelist",
]


class KCWInputParameters(BaseModel):
    """Input parameters for ``kcw.x`` calculations, split by namelist.

    Threaded per step: wann2kcw reads ``control`` + ``wannier``; screen
    reads ``control`` + ``wannier`` + ``screen``; ham reads ``control`` +
    ``wannier`` + ``ham``. ``control`` and ``wannier`` are therefore the
    same across every step by construction.
    """

    control: ControlNamelist = Field(default_factory=lambda: ControlNamelist())
    wannier: WannierNamelist = Field(default_factory=lambda: WannierNamelist())
    screen: ScreenNamelist = Field(default_factory=lambda: ScreenNamelist())
    ham: HamNamelist = Field(default_factory=lambda: HamNamelist())
