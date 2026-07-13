"""Machine learning configuration for screening parameter prediction."""

from typing import Self
from warnings import warn

from pydantic import Field, field_validator, model_validator

from koopmans.base import BaseModel


class MLConfig(BaseModel):
    """Configuration for machine learning models used to predict screening parameters."""

    train: bool = Field(
        default=False,
        description="train a machine learning model to predict the screening parameters",
    )
    test: bool = Field(default=False, description="test the machine learning model")
    predict: bool = Field(
        default=False,
        description="use a machine learning model to predict the screening parameters",
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
    alphas_from_file: bool = Field(
        default=False,
        description="If True, read the screening coefficients from file",
    )
    train_on_the_fly: bool = Field(
        default=False,
        description="If True, train the ML model after each orbital calculation",
    )
    occ_and_emp_together: bool = Field(
        default=True,
        description="If True, use one ML model for both occupied and empty states",
    )
    estimator: str = Field(
        default="ridge_regression", description="What to use as the estimator for the ML model"
    )
    descriptor: str = Field(
        default="orbital_density", description="What to use as the descriptor for the ML model"
    )

    @model_validator(mode="after")
    def mutually_exclusive_modes(self) -> Self:
        """Validate that ``train``, ``test``, and ``predict`` are mutually exclusive."""
        if [self.predict, self.train, self.test].count(True) > 1:
            raise ValueError(
                "Training, testing, and using the ML model are mutually exclusive; change `ml:predict` "
                "/`ml:train`/`ml:test` so that at most one is `True`"
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
