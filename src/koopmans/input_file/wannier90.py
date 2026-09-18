"""Input parameters for ``wannier90.x`` calculations."""

from typing import Any, Self

from pydantic import Field, model_validator
from wannier90_input.models.parameters import Projection

from koopmans.input_file._generated.wannier90 import _Wannier90Input

__all__ = ["RestrictedWannier90InputParameters"]


class RestrictedWannier90InputParameters(_Wannier90Input):
    """Wannier90 input parameters, excluding those that ``koopmans`` manages itself."""

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
