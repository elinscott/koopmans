"""Input parameters for ``pw2wannier90.x`` calculations."""

from pathlib import Path
from typing import ClassVar

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
