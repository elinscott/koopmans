"""Machine learning configuration for screening parameter prediction."""

from typing import Self
from warnings import warn

from aiida_koopmans.ml import MLDescriptor, MLMode
from pydantic import Field, field_validator, model_validator

from koopmans.base import BaseModel

__all__ = ["MLConfig"]


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
        description="PK or UUID of the stored trained-model node (the `model` output "
        "of a mode='train' run in this profile)",
    )
    model_file: str | None = Field(
        default=None,
        description="JSON file containing the ML model information",
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
        description="What to use as the descriptor for the ML model",
    )

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
