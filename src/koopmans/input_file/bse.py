"""Input parameters for a yambo Bethe-Salpeter (BSE) spectrum on Koopmans eigenvalues.

Read by the ``bse`` task alone: it composes a DFPT singlepoint (kcw.x KI
eigenvalues) with a yambo BSE run seeded by those eigenvalues in place of a
GW quasiparticle correction (see
``aiida_koopmans.workgraphs.bethe_salpeter``). Every field here maps
straight onto one yambo BSE runcard variable; keywords the route computes
or fixes for itself (``KfnQPdb``, the MPI role split, the light-polarisation
direction) have no field and cannot be set.
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from koopmans.base import BaseModel

__all__ = ["BSEInput"]


class BSEInput(BaseModel):
    """The yambo BSE runcard parameters a ``bse`` task needs."""

    screening_bands: int = Field(
        gt=0,
        description="number of bands (counted from band 1) included in the screening "
        "function W (yambo's `BndsRnXs`). A convergence parameter: tune it against the "
        "BSE spectrum, the way you would converge a plane-wave cutoff",
    )
    g_cutoff: float = Field(
        gt=0.0,
        description="G-vector cutoff, in Ry, shared by the screening (`NGsBlkXs`) and "
        "the BSE kernel (`BSENGBlk`). A convergence parameter",
    )
    bands: tuple[int, int] = Field(
        description="[first, last] band range (1-indexed, inclusive) the BSE kernel is "
        "built over (`BSEBands`). Koopmans quasiparticle corrections exist only for the "
        "bands the DFPT chain Wannierizes, so this range must lie inside it",
    )
    energy_range: tuple[float, float] = Field(
        description="[min, max] absorption energy window, in eV, the spectrum is "
        "computed over (`BEnRange`)",
    )
    energy_steps: int = Field(
        default=1000,
        gt=0,
        description="number of energy points sampling `energy_range` (`BEnSteps`). A "
        "numerical resolution, not a physical property of the system",
    )
    broadening: float | tuple[float, float] = Field(
        default=0.1,
        description="artificial linewidth, in eV, applied to each exciton peak "
        "(`BDmRange`): a single value broadens uniformly, or give [value at "
        "`energy_range[0]`, value at `energy_range[1]`] to vary it linearly across the "
        "window. A numerical smoothing parameter, not a physical property of the system",
    )
    eigenvalues: Literal["ki", "pki"] = Field(
        default="ki",
        description="which Koopmans eigenvalue flavour seeds the quasiparticle "
        "database: the self-consistent KI eigenvalues, or the perturbative `pki` "
        "correction computed alongside them in the same kcw.x `ham` run. Only `ki` is "
        "wired through so far",
    )

    @model_validator(mode="after")
    def check_bands_ordered(self) -> Self:
        """Require ``bands`` to name an ascending, positive range."""
        first, last = self.bands
        if first < 1 or last < first:
            raise ValueError(
                f"`bse.bands` = [{first}, {last}] must be a positive, ascending "
                "[first, last] band range."
            )
        return self

    @model_validator(mode="after")
    def check_energy_range_ordered(self) -> Self:
        """Require ``energy_range`` to name an ascending window."""
        low, high = self.energy_range
        if not low < high:
            raise ValueError(
                f"`bse.energy_range` = [{low}, {high}] must be ascending: the minimum "
                "energy must be strictly less than the maximum."
            )
        return self

    @model_validator(mode="after")
    def check_broadening_positive(self) -> Self:
        """Require every stated broadening value to be positive."""
        values = self.broadening if isinstance(self.broadening, tuple) else (self.broadening,)
        if any(value <= 0 for value in values):
            raise ValueError(f"`bse.broadening` = {self.broadening!r} must be positive.")
        return self
