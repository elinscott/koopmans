"""Input parameters for ``ph.x`` calculations."""

from pathlib import Path
from typing import ClassVar

from pydantic import Field, model_validator
from pydantic_espresso.models.ph.develop import InputphNamelist as _InputphNamelist

from koopmans.input_file._utils import reject_route_owned_fields

__all__ = ["PHInputParameters"]

# Fields the dft_eps route (``koopmans.aiida.workflows.eps``) derives itself,
# keyed to the input-file setting that determines the value. Explicitly
# stating one of these here would silently disagree with what the route
# actually runs, so they are rejected rather than merged.
_INPUTPH_OWNERS = {
    "epsil": "the dft_eps route (a dielectric-constant response is always epsil = .true.)",
    "trans": "the dft_eps route (it never solves for phonons; trans is always .false.)",
    "verbosity": "aiida-quantumespresso's ph.x plugin (forced high for every run)",
}


class PHInputParameters(_InputphNamelist):
    """``INPUTPH`` namelist for ``ph.x`` calculations.

    Legacy koopmans exposes ph.x settings as a single flat block, since
    ph.x reads only one namelist; this mirrors that shape rather than
    nesting a sub-block the way the multi-namelist ``pw``/``kcw`` blocks do.
    """

    # Excluded fields: aiida-quantumespresso's ``PhCalculation`` forces
    # these itself for every run (its own ``_blocked_keywords``) and raises
    # if the parameters ``Dict`` states them, so koopmans never lets them
    # reach the input file in the first place.
    outdir: ClassVar[Path | None] = None  # type: ignore[misc, assignment, unused-ignore]
    prefix: ClassVar[str | None] = None  # type: ignore[misc, assignment, unused-ignore]
    fildyn: ClassVar[str | None] = None  # type: ignore[misc, assignment, unused-ignore]
    ldisp: ClassVar[bool | None] = None  # type: ignore[misc, assignment, unused-ignore]
    nq1: ClassVar[int | None] = None  # type: ignore[misc, assignment, unused-ignore]
    nq2: ClassVar[int | None] = None  # type: ignore[misc, assignment, unused-ignore]
    nq3: ClassVar[int | None] = None  # type: ignore[misc, assignment, unused-ignore]
    qplot: ClassVar[bool | None] = None  # type: ignore[misc, assignment, unused-ignore]

    # The generated model states this required unconditionally; QE only
    # requires it when ``electron_phonon`` selects the ahc self-energy
    # method, so it is optional here too.
    ahc_nbnd: int | None = Field(default=None)  # type: ignore[assignment]

    @model_validator(mode="after")
    def reject_route_owned_keys(self) -> "PHInputParameters":
        """Reject keys the dft_eps route derives itself."""
        reject_route_owned_fields(self, _INPUTPH_OWNERS, "calculator_parameters.ph")
        return self
