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

# Fields the DFPT route (``koopmans.aiida.workflows.dfpt``) computes, derives
# or hardcodes itself, keyed to the input-file setting that determines the
# value (or, absent one, a description of what the workflow derives it
# from). Every field of every namelist below is accounted for as either
# owned here or a genuine pass-through default (documented on the owning
# class): stating an owned field here would silently disagree with, or be
# silently overwritten by, what the route actually runs, so it is rejected
# rather than merged. See CLAUDE.md's "explicit failure over silent ignore".
_CONTROL_OWNERS = {
    "calculation": "the kcw.x step being run (wann2kcw / screen / ham)",
    "prefix": "the preceding pw.x run, managed by koopmans",
    "outdir": "the preceding pw.x run, managed by koopmans",
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
    "seedname": "the block Wannierization's product files (koopmans always writes them under one fixed seedname)",
    "num_wann_occ": "calculator_parameters.wannier90.projections (the occupied manifold's blocks)",
    "num_wann_emp": "calculator_parameters.wannier90.projections (the empty manifold's blocks)",
    "have_empty": "calculator_parameters.wannier90.projections (whether an empty manifold is defined)",
    "has_disentangle": "the empty manifold's disentanglement window (whether num_bands exceeds num_wann)",
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
    ``KCWInputParameters``. ``kcw_iverbosity``, ``lrpa``, ``assume_isolated``,
    ``homo_only``, ``spread_thr``, ``io_sp``, ``io_real_space``, ``irr_bz``
    and ``use_wct`` are the pass-through fields: the DFPT route either never
    touches them or only seeds a literal default a user value freely
    replaces, so whatever is set here reaches kcw.x unchanged.
    """

    @model_validator(mode="after")
    def reject_route_owned_keys(self) -> "ControlNamelist":
        """Reject keys the DFPT route derives itself."""
        reject_route_owned_fields(self, _CONTROL_OWNERS, "calculator_parameters.kcw.control")
        return self


class WannierNamelist(_WannierNamelist):
    """``WANNIER`` namelist for ``kcw.x`` calculations.

    Threaded into every kcw.x step; see ``KCWInputParameters``. ``check_ks``
    and ``alpha_mix`` are the pass-through fields: the route either only
    seeds a literal default a user value freely replaces, or never touches
    the field at all.
    """

    @model_validator(mode="after")
    def reject_route_owned_keys(self) -> "WannierNamelist":
        """Reject keys the DFPT route derives itself."""
        reject_route_owned_fields(self, _WANNIER_OWNERS, "calculator_parameters.kcw.wannier")
        return self


class ScreenNamelist(_ScreenNamelist):
    """``SCREEN`` namelist for ``kcw.x`` calculations.

    Threaded into the screen step only, alongside ``control`` and
    ``wannier``. ``niter``, ``nmix`` and ``tr2`` are the pass-through
    fields: the route only seeds a literal convergence default, which a
    user value freely replaces.
    """

    @model_validator(mode="after")
    def reject_route_owned_keys(self) -> "ScreenNamelist":
        """Reject keys the DFPT route derives itself."""
        reject_route_owned_fields(self, _SCREEN_OWNERS, "calculator_parameters.kcw.screen")
        return self


class HamNamelist(_HamNamelist):
    """``HAM`` namelist for ``kcw.x`` calculations.

    Threaded into the ham step only, alongside ``control`` and ``wannier``.
    ``use_ws_distance``, ``write_hr`` and ``on_site_only`` are the
    pass-through fields: the route only seeds a literal default, which a
    user value freely replaces. ``on_site_only`` trades accuracy for speed
    and is expert-only: leave it at its default unless you know you want
    the on-site approximation.
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
