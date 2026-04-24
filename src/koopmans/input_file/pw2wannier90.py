"""Input parameters for ``pw2wannier90.x`` calculations."""

from koopmans.base import BaseModel


class PW2Wannier90InputParameters(BaseModel):
    """Input parameters for ``pw2wannier90.x`` calculations."""

    atom_proj_ext: bool = False
    atom_proj_dir: str | None = None
