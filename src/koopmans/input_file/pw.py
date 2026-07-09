"""Input parameters for ``pw.x`` calculations."""

from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator
from pydantic_espresso.models.pw.develop import ControlNamelist as _ControlNamelist
from pydantic_espresso.models.pw.develop import ElectronsNamelist
from pydantic_espresso.models.pw.develop import SystemNamelist as _SystemNamelist

from koopmans.base import BaseModel


class ControlNamelist(_ControlNamelist):
    """``CONTROL`` namelist for ``pw.x`` calculations."""

    pseudo_dir: ClassVar[str | None] = None  # Exclude this field
    outdir: ClassVar[str | None] = None  # Exclude this field
    prefix: ClassVar[str | None] = None  # Exclude this field

    @field_validator("verbosity", mode="before")
    @classmethod
    def enforce_high_verbosity(cls, v: Any) -> Literal["high"]:
        """High verbosity is required to guarantee that all bands will be printed."""
        return "high"


class SystemNamelist(_SystemNamelist):
    """``SYSTEM`` namelist for ``pw.x`` calculations.

    ``ibrav``, ``nat`` and ``ntyp`` are derived from the input structure, not provided
    by the user. ``ecutwfc`` is optional at parse time; the dispatcher raises if it is
    still unset when the workgraph is built.
    """

    ibrav: ClassVar[int | None] = None  # Exclude this field
    nat: ClassVar[int | None] = None  # Exclude this field
    ntyp: ClassVar[int | None] = None  # Exclude this field
    ecutwfc: float | None = None


class PWInputParameters(BaseModel):
    """Input parameters for ``pw.x`` calculations."""

    control: ControlNamelist = Field(default_factory=lambda: ControlNamelist())
    system: SystemNamelist = Field(default_factory=lambda: SystemNamelist())
    electrons: ElectronsNamelist = Field(default_factory=lambda: ElectronsNamelist())
