"""Machine learning configuration for screening parameter prediction."""

from enum import Enum
from typing import Self
from warnings import warn

from aiida_koopmans.ml import MLDescriptor, MLMode
from pydantic import Field, field_validator, model_validator

from koopmans.base import BaseModel

__all__ = ["MLConfig"]

# The plugin's placeholder for "no descriptor named", matched on the member's
# name so it stays rejected however its value is spelled. It is not a
# descriptor: it exists to give the workflows' enum sockets a value to carry.
_PLACEHOLDER_DESCRIPTOR = "unset"


def _selectable_descriptors() -> str:
    """List the descriptors an input file may name, in input-file spelling."""
    return " or ".join(
        repr(str(member.value))
        for member in MLDescriptor
        if member.name.casefold() != _PLACEHOLDER_DESCRIPTOR
    )


class MLConfig(BaseModel):
    """Configuration for machine learning models used to predict screening parameters."""

    mode: MLMode = Field(
        default=MLMode.NONE,
        description="'train' fits a screening model on the computed alphas, 'test' scores "
        "an existing model against them, 'predict' applies an existing model in place of "
        "the alpha calculation",
    )
    model: int | str | None = Field(
        default=None,
        description="reuse a model trained in this database: the identifier a "
        "mode='train' run prints on completion ('Trained model stored as node "
        "<pk> ...'). If you have a model.json file instead, use `model_file`",
    )
    model_file: str | None = Field(
        default=None,
        description="path to a trained model's JSON file (a mode='train' run writes "
        "model.json next to its other outputs)",
    )
    n_max: int = Field(
        default=4,
        gt=0,
        description="the maximum expansion coefficient n for radial basis functions",
    )
    l_max: int = Field(default=4, gt=0, description="The maximum angular expansion coefficient")
    r_min: float = Field(
        default=0.5, gt=0.0, description="The width of the narrowest radial basis function"
    )
    r_max: float = Field(
        default=4.0, gt=0.0, description="The width of the broadest radial basis function"
    )
    occ_and_emp_together: bool = Field(
        default=True,
        description="If True, use one ML model for both occupied and empty states",
    )
    estimator: str = Field(
        default="ridge_regression", description="What to use as the estimator for the ML model"
    )
    descriptor: MLDescriptor = Field(
        default=MLDescriptor.POWER_SPECTRUM,
        description="What the ML model reads for each orbital. 'power_spectrum' expands the "
        "orbital density in a radial and spherical basis. 'self_hartree' is a single number "
        "per orbital: simplistic and unexpressive, and unlikely to carry enough to predict a "
        "screening parameter",
    )

    @field_validator("descriptor", mode="before")
    @classmethod
    def reject_placeholder_descriptor(cls, value: object) -> object:
        """Reject the plugin's placeholder for "no descriptor named".

        Runs before coercion, so an input file naming it is refused whether
        or not the plugin's enum carries the member yet.
        """
        if isinstance(value, Enum):
            spellings = {value.name.casefold(), str(value.value).casefold()}
        elif isinstance(value, str):
            spellings = {value.casefold()}
        else:
            return value
        if _PLACEHOLDER_DESCRIPTOR in spellings:
            raise ValueError(
                f"`ml:descriptor` takes {_selectable_descriptors()}; "
                f"'{_PLACEHOLDER_DESCRIPTOR}' is an internal placeholder the workflow "
                "puts on its own inputs, not a descriptor you can choose."
            )
        return value

    @field_validator("model", mode="before")
    @classmethod
    def check_model_identifier(cls, value: object) -> object:
        """Reject identifier forms that would silently coerce to a wrong PK.

        Pydantic would turn ``42.0`` into PK 42 and ``true`` into PK 1;
        only an integer PK or a string UUID names a node deliberately.
        """
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"`ml:model` takes the stored model node's integer PK or string UUID; "
                f"got {value!r}."
            )
        return value

    @model_validator(mode="after")
    def model_sources_are_exclusive(self) -> Self:
        """Validate that ``model`` and ``model_file`` are not both given."""
        if self.model is not None and self.model_file is not None:
            raise ValueError(
                "`ml:model` (a stored model node) and `ml:model_file` (a JSON copy) "
                "are two sources for the same model; supply exactly one."
            )
        return self

    @field_validator("r_min", mode="after")
    @classmethod
    def check_small_rmin(cls, v: float) -> float:
        """Warn if ``r_min`` is very small."""
        if v < 0.5:
            warn(
                "Small values of `r_min` (<0.5) can lead to problems in the construction of "
                f"the radial basis. The provided value is `r_min = {v}`.",
                stacklevel=2,
            )
        return v

    @model_validator(mode="after")
    def check_rmin_less_than_rmax(self) -> Self:
        """Validate that ``r_min`` is less than ``r_max``."""
        if not self.r_min < self.r_max:
            raise ValueError(f"`r_min` is larger or equal to `r_max = {self.r_max}`.")
        return self
