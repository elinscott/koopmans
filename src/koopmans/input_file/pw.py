"""Input parameters for ``pw.x`` calculations."""

from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator
from pydantic_espresso.models.pw.develop import ControlNamelist as _ControlNamelist
from pydantic_espresso.models.pw.develop import ElectronsNamelist
from pydantic_espresso.models.pw.develop import SystemNamelist as _SystemNamelist

from koopmans.base import BaseModel


class ControlNamelist(_ControlNamelist):
    """``CONTROL`` namelist for ``pw.x`` calculations."""

    # Excluded fields: koopmans manages these itself, so they are demoted to
    # class variables to drop them from the pydantic schema. mypy --strict (even
    # with the pydantic plugin) cannot express a ClassVar overriding a base
    # model field, hence the targeted ignores.
    pseudo_dir: ClassVar[str | None] = None  # type: ignore[misc, assignment]
    outdir: ClassVar[str | None] = None  # type: ignore[misc, assignment]
    prefix: ClassVar[str | None] = None  # type: ignore[misc, assignment]

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

    # Excluded fields (see ``ControlNamelist`` above for the ClassVar rationale).
    ibrav: ClassVar[int | None] = None  # type: ignore[misc, assignment]
    nat: ClassVar[int | None] = None  # type: ignore[misc, assignment]
    ntyp: ClassVar[int | None] = None  # type: ignore[misc, assignment]
    # Optional at parse time; the dispatcher raises if still unset at build time.
    ecutwfc: float | None = None  # type: ignore[assignment]


class PWInputParameters(BaseModel):
    """Input parameters for ``pw.x`` calculations."""

    control: ControlNamelist = Field(default_factory=lambda: ControlNamelist())
    system: SystemNamelist = Field(default_factory=lambda: SystemNamelist())  # type: ignore[call-arg]
    electrons: ElectronsNamelist = Field(default_factory=lambda: ElectronsNamelist())
