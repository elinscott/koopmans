"""Input parameters for ``pw2wannier90.x`` calculations."""

from pathlib import Path
from typing import ClassVar, Self

from pydantic import Field, model_validator
from pydantic_espresso.models.pw2wannier90.develop import InputppNamelist


class PW2Wannier90InputParameters(InputppNamelist):
    """Input parameters for ``pw2wannier90.x`` calculations (the ``INPUTPP`` namelist)."""

    # Excluded fields: koopmans manages these itself, so they are demoted to
    # class variables to drop them from the pydantic schema. mypy --strict (even
    # with the pydantic plugin) cannot express a ClassVar overriding a base
    # model field, hence the ignores; unused-ignore is included because the
    # generated base models' field optionality varies between checkouts.
    prefix: ClassVar[str | None] = None  # type: ignore[misc, assignment, unused-ignore]
    outdir: ClassVar[Path | None] = None  # type: ignore[misc, assignment, unused-ignore]
    seedname: ClassVar[str | None] = None  # type: ignore[misc, assignment, unused-ignore]

    # Not in the generated base model (a newer pw2wannier90 keyword): the
    # 1-based indices of external projectors to hold fixed during the Lowdin
    # orthonormalization. Absent means none are frozen, i.e. every projector
    # is orthonormalized.
    atom_proj_frozen: list[int] | None = Field(
        default=None,
        description=(
            "1-based indices of external projectors to freeze (hold fixed during the "
            "Lowdin orthonormalization). Requires `atom_proj_ext`; when absent, no "
            "projector is frozen."
        ),
    )

    @model_validator(mode="after")
    def frozen_requires_external(self) -> Self:
        """Reject a frozen-projector list without external projectors.

        ``atom_proj_frozen`` indexes the external projector files, so without
        ``atom_proj_ext`` it cannot take effect.
        """
        if self.atom_proj_frozen is not None and not self.atom_proj_ext:
            raise ValueError(
                "`pw2wannier90.atom_proj_frozen` requires `pw2wannier90.atom_proj_ext`: "
                "the frozen indices refer to the external projector files."
            )
        return self
