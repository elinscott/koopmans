"""Input parameters for unfold-and-interpolate post-processing."""

from typing import Any

from pydantic import Field, field_validator

from koopmans.base import BaseModel

__all__ = ["UnfoldAndInterpolateConfig"]


class UnfoldAndInterpolateConfig(BaseModel):
    """Input parameters for unfold-and-interpolate post-processing."""

    use_ws_distance: bool = Field(
        default=True,
        description=(
            "if True, the real Wigner-Seitz distance between the Wannier functions centers is considered as in "
            "the Wannier90 code. In particular, this accounts for the periodic boundary conditions and it is "
            "crucial for a good interpolation when using coarse MP meshes or, equivalently, small supercells"
        ),
    )
    smooth_int_factor: tuple[int, int, int] = Field(
        default=(1, 1, 1),
        description=(
            "if this is > 1 (or is a 3-element list with at least one entry > 1), the smooth interpolation "
            "method is used. This consists of removing the DFT part of the Hamiltonian from the full Koopmans "
            "Hamiltonian and adding the DFT Hamiltonian from a calculation with a denser k-points mesh, where "
            "this keyword defines how many times denser to make the mesh. (If this is set to a scalar a, the "
            "new k-grid will be [a*kx_old, a*ky_old, a*kz_old]. If it is a list [a, b, c], the dense k-grid "
            "will be [a*kx_old, b*ky_old, c*kz_old].) This works only for a non self-consistent Koopmans "
            "calculation using Wannier since, to be consistent, all the Hamiltonians must be in the same "
            "gauge, i.e. the Wannier gauge"
        ),
    )
    do_dos: bool = Field(
        default=True,
        description=(
            "if True, the density-of-states is interpolated along the k-point path specified in the "
            '`kpoints` block. The DOS is written to a file called "dos_interpolated.dat"'
        ),
    )

    @field_validator("smooth_int_factor", mode="before")
    @classmethod
    def ensure_smooth_int_factor_is_tuple(cls, v: Any) -> Any:
        """Convert smooth_int_factor to a tuple if it is an int or list."""
        if isinstance(v, int):
            v = (v, v, v)
        elif isinstance(v, list):
            v = tuple(v)
        return v

    @property
    def do_smooth_interpolation(self) -> bool:
        """Return True if the smooth interpolation is used."""
        return any(f > 1 for f in self.smooth_int_factor)
