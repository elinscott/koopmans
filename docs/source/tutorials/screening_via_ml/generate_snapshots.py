"""Generate the water configurations this tutorial trains and tests on.

Displaces the atoms of a single water molecule by Gaussian noise, twenty
times over, and writes the first ``N_TRAIN`` configurations to
``training_snapshots.xyz`` and the rest to ``testing_snapshots.xyz``. The
seed is fixed, so the files are the ones the tutorial ships.
"""

import numpy as np

# How many of the twenty configurations to train on. The rest are for
# testing; one of the tutorial's exercises raises this to 10.
N_TRAIN = 5

# The cubic box the molecule sits in, in angstrom, matching the input files.
CELL = 6.8929

# An undistorted water molecule, near the middle of that box.
SYMBOLS = ["O", "H", "H"]
POSITIONS = np.array(
    [
        [3.0000, 3.11915, 3.35845],
        [3.2774, 4.01205, 3.61285],
        [3.6068, 2.88085, 2.64155],
    ]
)

# The width of the displacement applied to every coordinate, in angstrom.
NOISE = 0.1


def write_xyz(filename: str, frames: list[np.ndarray]) -> None:
    """Write ``frames`` to an extended-xyz file, one configuration per frame."""
    lattice = f"{CELL} 0.0 0.0 0.0 {CELL} 0.0 0.0 0.0 {CELL}"
    comment = f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"'
    with open(filename, "w") as handle:
        for positions in frames:
            handle.write(f"{len(SYMBOLS)}\n{comment}\n")
            for symbol, (x, y, z) in zip(SYMBOLS, positions, strict=True):
                handle.write(f"{symbol:<2}{x:17.8f}{y:17.8f}{z:17.8f}\n")


def main() -> None:
    """Write the training and testing configurations."""
    rng = np.random.RandomState(0)
    frames = [POSITIONS + NOISE * rng.normal(0, 1, POSITIONS.shape) for _ in range(20)]
    write_xyz("training_snapshots.xyz", frames[:N_TRAIN])
    write_xyz("testing_snapshots.xyz", frames[N_TRAIN:])


if __name__ == "__main__":
    main()
