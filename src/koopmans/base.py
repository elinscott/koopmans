"""Pydantic base model to use throughout `koopmans`."""

from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict

default_config = ConfigDict(extra="forbid",
                            arbitrary_types_allowed=True,
                            strict=False,
                            validate_assignment=True,
                            revalidate_instances='never',
                            )


class BaseModel(_BaseModel):
    """Base model with a modified default configuration."""

    model_config = default_config
