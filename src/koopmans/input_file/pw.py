"""Input parameters for ``pw.x`` calculations."""

from pydantic import Field
from pydantic_espresso.models.pw.develop import ElectronsNamelist

from koopmans.base import BaseModel
from koopmans.input_file._generated.pw import _ControlNamelist, _SystemNamelist

__all__ = ["ControlNamelist", "PWInputParameters", "SystemNamelist"]


class ControlNamelist(_ControlNamelist):
    """``CONTROL`` namelist for ``pw.x`` calculations."""


class SystemNamelist(_SystemNamelist):
    """``SYSTEM`` namelist for ``pw.x`` calculations."""


class PWInputParameters(BaseModel):
    """Input parameters for ``pw.x`` calculations."""

    control: ControlNamelist = Field(default_factory=lambda: ControlNamelist())
    system: SystemNamelist = Field(default_factory=lambda: SystemNamelist())
    electrons: ElectronsNamelist = Field(default_factory=lambda: ElectronsNamelist())
