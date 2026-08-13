"""Input parameters for ``kcw.x`` calculations."""

from pydantic import Field, model_validator
from pydantic_espresso.models.kcw.develop import ControlNamelist as _ControlNamelist
from pydantic_espresso.models.kcw.develop import HamNamelist as _HamNamelist
from pydantic_espresso.models.kcw.develop import ScreenNamelist as _ScreenNamelist
from pydantic_espresso.models.kcw.develop import WannierNamelist as _WannierNamelist

from koopmans.base import BaseModel
from koopmans.input_file._utils import reject_route_owned_fields

__all__ = [
    "ControlNamelist",
    "HamNamelist",
    "KCWInputParameters",
    "ScreenNamelist",
    "WannierNamelist",
]

# Fields the DFPT route (``koopmans.aiida.workflows.dfpt``) derives itself,
# keyed to the input-file setting that determines the value. Explicitly
# stating one of these here would silently disagree with what the route
# actually runs, so they are rejected rather than merged.
_CONTROL_OWNERS = {
    "calculation": "the kcw.x step being run (wann2kcw / screen / ham)",
    "prefix": "the preceding pw.x run's prefix, managed by koopmans",
    "outdir": "the preceding pw.x run's outdir, managed by koopmans",
    "mp1": "kpoints (the nscf mesh the Wannier functions were built on)",
    "mp2": "kpoints (the nscf mesh the Wannier functions were built on)",
    "mp3": "kpoints (the nscf mesh the Wannier functions were built on)",
    "l_vcut": "workflow.gb_correction",
    "spin_component": "workflow.spin (one kcw.x chain runs per spin channel)",
    "kcw_at_ks": "workflow.init_orbitals (only the mlwfs/projwfs Wannier-function routes are wired)",
    "read_unitary_matrix": (
        "workflow.init_orbitals (only the mlwfs/projwfs Wannier-function routes are wired)"
    ),
}

_WANNIER_OWNERS = {
    "seedname": "the wannier90 seedname koopmans writes its products under",
    "num_wann_occ": "calculator_parameters.wannier90.projections (the occupied manifold's blocks)",
    "num_wann_emp": "calculator_parameters.wannier90.projections (the empty manifold's blocks)",
    "have_empty": "calculator_parameters.wannier90.projections (whether an empty manifold is defined)",
    "has_disentangle": "the empty manifold's disentanglement window (num_bands vs num_wann)",
}

_SCREEN_OWNERS = {
    "i_orb": "workflow.group_orbitals_by / workflow.group_orbitals_tol",
    "check_spread": "workflow.group_orbitals_by / workflow.group_orbitals_tol",
    "eps_inf": "workflow.eps_inf",
}

_HAM_OWNERS = {
    "do_bands": "kpoints.path (an interpolated band structure runs whenever a k-path is given)",
}


class ControlNamelist(_ControlNamelist):
    """``CONTROL`` namelist for ``kcw.x`` calculations.

    Threaded into every kcw.x step (wann2kcw, screen, ham); see
    ``KCWInputParameters``.
    """

    @model_validator(mode="after")
    def reject_route_owned_keys(self) -> "ControlNamelist":
        """Reject keys the DFPT route derives itself."""
        reject_route_owned_fields(self, _CONTROL_OWNERS, "calculator_parameters.kcw.control")
        return self


class WannierNamelist(_WannierNamelist):
    """``WANNIER`` namelist for ``kcw.x`` calculations.

    Threaded into every kcw.x step; see ``KCWInputParameters``.
    """

    @model_validator(mode="after")
    def reject_route_owned_keys(self) -> "WannierNamelist":
        """Reject keys the DFPT route derives itself."""
        reject_route_owned_fields(self, _WANNIER_OWNERS, "calculator_parameters.kcw.wannier")
        return self


class ScreenNamelist(_ScreenNamelist):
    """``SCREEN`` namelist for ``kcw.x`` calculations.

    Threaded into the screen step only, alongside ``control`` and
    ``wannier``.
    """

    @model_validator(mode="after")
    def reject_route_owned_keys(self) -> "ScreenNamelist":
        """Reject keys the DFPT route derives itself."""
        reject_route_owned_fields(self, _SCREEN_OWNERS, "calculator_parameters.kcw.screen")
        return self


class HamNamelist(_HamNamelist):
    """``HAM`` namelist for ``kcw.x`` calculations.

    Threaded into the ham step only, alongside ``control`` and ``wannier``.
    ``on_site_only`` trades accuracy for speed and is expert-only: leave it
    at its default unless you know you want the on-site approximation.
    """

    @model_validator(mode="after")
    def reject_route_owned_keys(self) -> "HamNamelist":
        """Reject keys the DFPT route derives itself."""
        reject_route_owned_fields(self, _HAM_OWNERS, "calculator_parameters.kcw.ham")
        return self


class KCWInputParameters(BaseModel):
    """Input parameters for ``kcw.x`` calculations, split by namelist.

    Threaded per step: wann2kcw reads ``control`` + ``wannier``; screen
    reads ``control`` + ``wannier`` + ``screen``; ham reads ``control`` +
    ``wannier`` + ``ham``. ``control`` and ``wannier`` are therefore the
    same across every step by construction.
    """

    control: ControlNamelist = Field(default_factory=lambda: ControlNamelist())
    wannier: WannierNamelist = Field(default_factory=lambda: WannierNamelist())
    screen: ScreenNamelist = Field(default_factory=lambda: ScreenNamelist())
    ham: HamNamelist = Field(default_factory=lambda: HamNamelist())
