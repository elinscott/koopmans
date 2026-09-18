"""Plot how closely the predicted screening parameters match the computed ones.

Run this from the directory holding the ``test/`` output directory that
``koopmans run test.yaml`` wrote. Each snapshot there ran two final KI
calculations, one at the computed screening parameters and one at the
model's predictions, so the difference between them is the model's doing
and nothing else. The script writes ``screening_accuracy.svg``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN = Path("test")

# The two final KI calculations of each snapshot: the reference one, at the
# screening parameters the Delta-SCF calculations produced, and the one at the
# parameters the model predicted from the same trial calculation.
COMPUTED = "04-RunFinalKI"
PREDICTED = "05-run_final_ki_predicted"

# kcp.x reads its screening parameters from these two files, filled orbitals in
# one and empty in the other, and prints its orbital energies under these two
# headers. Both list the spin-up channel before the spin-down one; the molecule
# is closed-shell, so the two channels are copies and only the first is read.
ALPHA_FILES = {"filled": "file_alpharef.txt", "empty": "file_alpharef_empty.txt"}
EIGENVALUE_HEADERS = [
    "Eigenvalues (eV), kp =   1 , spin =  1",
    "Empty States Eigenvalues (eV), kp =   1 , spin =  1",
]


def screening_parameters(calculation: Path) -> dict[str, list[float]]:
    """Return the spin-up screening parameters a final KI calculation used.

    Keyed by manifold, filled and empty being fitted by models of their own.
    Each file opens with a count and then lists one orbital per line, as
    ``index alpha weight``, spin-up channel first.
    """
    alphas: dict[str, list[float]] = {}
    for manifold, name in ALPHA_FILES.items():
        alpha_file = calculation / "inputs" / name
        if not alpha_file.is_file():
            alphas[manifold] = []
            continue
        fields = alpha_file.read_text().split()
        norbitals = int(fields[0])
        alphas[manifold] = [float(fields[3 * i + 2]) for i in range(norbitals // 2)]
    return alphas


def orbital_energies(calculation: Path) -> list[float]:
    """Return the spin-up orbital energies of a final KI calculation, in eV.

    ``kcp.x`` prints a block under each header once per self-consistency
    cycle, so the converged values are the last block under each.
    """
    output_file = calculation / "outputs" / "aiida.cpo"
    if not output_file.is_file():
        raise SystemExit(f"No such file: {output_file}. Run `koopmans run test.yaml` first.")
    lines = output_file.read_text().splitlines()

    energies: list[float] = []
    for header in EIGENVALUE_HEADERS:
        positions = [i for i, line in enumerate(lines) if line.strip() == header]
        if not positions:
            raise SystemExit(f"{output_file} has no `{header}` line; did the calculation finish?")
        block: list[float] = []
        for line in lines[positions[-1] + 1 :]:
            if line.strip():
                block += [float(value) for value in line.split()]
            elif block:
                break
        energies += block
    return energies


def main() -> None:
    """Read every snapshot of the test run and write screening_accuracy.svg."""
    snapshots = sorted(RUN.glob("*-dscf_snapshot_*"))
    if not snapshots:
        raise SystemExit(f"No snapshots found in {RUN}/. Run `koopmans run test.yaml` first.")

    computed_alphas: dict[str, list[float]] = {manifold: [] for manifold in ALPHA_FILES}
    predicted_alphas: dict[str, list[float]] = {manifold: [] for manifold in ALPHA_FILES}
    energy_errors: list[float] = []
    for snapshot in snapshots:
        for manifold, alphas in screening_parameters(snapshot / COMPUTED).items():
            computed_alphas[manifold] += alphas
        for manifold, alphas in screening_parameters(snapshot / PREDICTED).items():
            predicted_alphas[manifold] += alphas
        energy_errors += [
            predicted - computed
            for computed, predicted in zip(
                orbital_energies(snapshot / COMPUTED),
                orbital_energies(snapshot / PREDICTED),
                strict=True,
            )
        ]

    errors = np.array(energy_errors) * 1000

    fig, (left, right) = plt.subplots(ncols=2, figsize=(8, 4))

    lims = (0.3, 0.65)
    for manifold in ALPHA_FILES:
        left.scatter(
            computed_alphas[manifold], predicted_alphas[manifold], alpha=0.6, label=manifold
        )
    left.plot(lims, lims, "k--", linewidth=0.8)
    left.set_xlim(lims)
    left.set_ylim(lims)
    left.set_aspect("equal", adjustable="box")
    left.set_xlabel(r"computed $\alpha_i$")
    left.set_ylabel(r"predicted $\alpha_i$")
    left.legend(loc="lower right")

    right.hist(errors, bins=20)
    right.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
    right.set_xlabel(
        r"$\varepsilon_i^\mathsf{predicted} - "
        r"\varepsilon_i^\mathsf{computed}$ (meV)"
    )
    right.set_ylabel("orbitals")
    right.annotate(
        f"mean = {errors.mean():+.0f} meV\n$\\sigma$ = {errors.std():.0f} meV",
        xy=(0.03, 0.95),
        xycoords="axes fraction",
        va="top",
    )

    fig.tight_layout()
    fig.savefig("screening_accuracy.svg")


if __name__ == "__main__":
    main()
