"""Input parameters for ``wannier90.x`` calculations."""

from typing import Any, ClassVar, Self

from pydantic import Field, model_validator
from wannier90_input.models.latest import Wannier90Input
from wannier90_input.models.parameters import (
    AtomCart,
    AtomFrac,
    Coordinate,
    FractionalCoordinate,
    Projection,
)


class RestrictedWannier90InputParameters(Wannier90Input):
    """Wannier90 input parameters, excluding those that ``koopmans`` manages itself.

    The structure and k-points are stored centrally in the input file, and the
    band/projection bookkeeping is derived by the workflow, so those keywords
    are demoted to class variables to drop them from the pydantic schema (see
    ``pw.py`` for the ClassVar rationale and the mypy ignores).
    """

    num_wann: ClassVar[int | None] = None  # type: ignore[misc, assignment, unused-ignore]
    num_bands: ClassVar[int | None] = None  # type: ignore[misc, assignment, unused-ignore]
    exclude_bands: ClassVar[list[int] | None] = None  # type: ignore[misc, assignment, unused-ignore]
    unit_cell_cart: ClassVar[list[Coordinate] | None] = None  # type: ignore[misc, assignment, unused-ignore]
    atoms_cart: ClassVar[list[AtomCart] | None] = None  # type: ignore[misc, assignment, unused-ignore]
    atoms_frac: ClassVar[list[AtomFrac] | None] = None  # type: ignore[misc, assignment, unused-ignore]
    mp_grid: ClassVar[tuple[int, int, int] | None] = None  # type: ignore[misc, assignment, unused-ignore]
    kpoints: ClassVar[list[FractionalCoordinate] | None] = None  # type: ignore[misc, assignment, unused-ignore]

    # Redefined (not excluded): in the input file, projections are specified as a
    # list of lists to separate each block
    projections: list[list[Projection]] = Field(default_factory=list)  # type: ignore[assignment]

    @model_validator(mode="before")
    @classmethod
    def set_default_num_bands(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Disable the base-class default: ``num_bands`` is not a field of this model."""
        return values

    @model_validator(mode="after")
    def atoms_frac_xor_cart(self) -> Self:
        """Disable the base-class check: the structure comes from the ``atoms`` block."""
        return self
