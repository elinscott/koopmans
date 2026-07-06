"""Input schema for atomic positions."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field

from koopmans.base import BaseModel
from koopmans.input_file._utils import tidy_units


class AtomicPositionsInput(BaseModel):
    """Input schema for specifying atomic positions in a structure."""

    positions: list[tuple[str, float, float, float]]
    units: Annotated[Literal["crystal", "ang", "bohr", "alat"], BeforeValidator(tidy_units)] = (
        "alat"
    )


class SnapshotsInput(BaseModel):
    """Input schema pointing at a multi-frame trajectory file of atomic positions.

    The legacy convention (``read_atoms_dict`` in
    ``koopmans/workflows/_workflow.py``): ``atomic_positions`` carries a
    single ``snapshots`` key whose value is the path to an (extended)
    xyz file; every frame becomes one snapshot of a ``trajectory`` task.
    The cell and periodicity always come from the ``cell_parameters``
    block (which the schema requires), overriding whatever lattice the
    xyz file declares — the legacy behaviour when ``cell_parameters``
    is present.
    """

    snapshots: str = Field(
        description="Path to a multi-frame (extended) xyz file; each frame is one snapshot. "
        "Relative paths are resolved against the current working directory."
    )

    def read_frames(self) -> list[AtomicPositionsInput]:
        """Read every frame of the snapshots file as an ``AtomicPositionsInput``.

        Positions are returned in angstrom (ASE's native unit for xyz files).

        Raises:
            ValueError: If the file does not exist or contains no frames.
        """
        from ase.io import read as ase_read

        path = Path(self.snapshots).expanduser()
        if not path.is_file():
            raise ValueError(
                f"`atomic_positions.snapshots` points to `{self.snapshots}`, which does not exist."
            )
        frames = ase_read(path, index=":")
        if not frames:
            raise ValueError(f"`{self.snapshots}` contains no snapshots.")
        return [
            AtomicPositionsInput(
                positions=[
                    (symbol, position[0], position[1], position[2])
                    for symbol, position in zip(
                        frame.get_chemical_symbols(), frame.get_positions(), strict=True
                    )
                ],
                units="ang",
            )
            for frame in frames
        ]
