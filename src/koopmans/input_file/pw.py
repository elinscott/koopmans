"""Input parameters for ``pw.x`` calculations."""

from pydantic import Field
from pydantic_espresso.models.pw.develop import ElectronsNamelist

from koopmans.base import BaseModel

# Generated (``just generate-input-models``): the keywords the workflow
# determines are absent from these, and nothing here is added to them.
from koopmans.input_file._generated.pw import ControlNamelist, SystemNamelist

__all__ = ["ControlNamelist", "PWInputParameters", "SystemNamelist"]


class PWInputParameters(BaseModel):
    """Input parameters for ``pw.x`` calculations."""

    control: ControlNamelist = Field(default_factory=lambda: ControlNamelist())
    system: SystemNamelist = Field(default_factory=lambda: SystemNamelist())
    electrons: ElectronsNamelist = Field(default_factory=lambda: ElectronsNamelist())
