"""Input parameters for ``ph.x`` calculations."""

from pydantic import Field

from koopmans.input_file._generated.ph import _InputphNamelist

__all__ = ["PHInputParameters"]


class PHInputParameters(_InputphNamelist):
    """``INPUTPH`` namelist for ``ph.x`` calculations.

    Legacy koopmans exposes ph.x settings as a single flat block, since
    ph.x reads only one namelist; this mirrors that shape rather than
    nesting a sub-block the way the multi-namelist ``pw``/``kcw`` blocks do.
    """

    # The generated model states this required unconditionally; QE only
    # requires it when ``electron_phonon`` selects the ahc self-energy
    # method, so it is optional here too.
    ahc_nbnd: int | None = Field(default=None)  # type: ignore[assignment]
