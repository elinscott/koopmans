"""Settings module for plotting-related parameters (DOS smearing and window)."""

from pydantic import Field

from koopmans.base import BaseModel


class PlottingConfig(BaseModel):
    """Model for the plotting settings (legacy ``plotting`` input block)."""

    degauss: float = Field(
        default=0.05, description="gaussian broadening (in eV) for the DOS interpolation, as in QE"
    )
    nstep: int = Field(
        default=1000, description="number of steps for the plot of the interpolated DOS"
    )
    Emin: float | None = Field(
        default=None, description="minimum energy for the plot of the interpolated DOS"
    )
    Emax: float | None = Field(
        default=None, description="maximum energy for the plot of the interpolated DOS"
    )
