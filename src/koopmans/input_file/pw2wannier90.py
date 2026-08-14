"""Input parameters for ``pw2wannier90.x`` calculations."""

from koopmans.input_file._generated.pw2wannier90 import _InputppNamelist

__all__ = ["PW2Wannier90InputParameters"]


class PW2Wannier90InputParameters(_InputppNamelist):
    """Input parameters for ``pw2wannier90.x`` calculations (the ``INPUTPP`` namelist)."""
