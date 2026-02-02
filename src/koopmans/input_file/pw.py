"""Input parameters for ``pw.x`` calculations."""

from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_espresso.models.qe_7_4.pw import ControlNamelist as _ControlNamelist
from pydantic_espresso.models.qe_7_4.pw import ElectronsNamelist, SystemNamelist

from koopmans.base import BaseModel


class ControlNamelist(_ControlNamelist):
    """``CONTROL`` namelist for ``pw.x`` calculations."""

    @field_validator("verbosity", mode="before")
    @classmethod
    def enforce_high_verbosity(cls, v: Any) -> Literal["high"]:
        """High verbosity is required to guarantee that all bands will be printed."""
        return "high"


class PWInputParameters(BaseModel):
    """Input parameters for ``pw.x`` calculations."""

    control: ControlNamelist = Field(default_factory=lambda: ControlNamelist())
    system: SystemNamelist = Field(default_factory=lambda: SystemNamelist())
    electrons: ElectronsNamelist = Field(default_factory=lambda: ElectronsNamelist())
