"""Plot the KI and PBE binding energies of ozone against photoemission.

Run this from the directory holding the ``ozone/`` output directory that
``koopmans run ozone.yaml`` wrote. It writes ``ozone_spectrum.svg``.
"""

from pathlib import Path

import matplotlib.pyplot as plt

# Experimental binding energies (eV) of the three outermost occupied orbitals
# of ozone, ordered from the most tightly bound to the HOMO, from gas-phase
# photoemission: Wiesner et al., Chem. Phys. Lett. 375, 76 (2003),
# doi:10.1016/S0009-2614(03)00818-2. Deeper orbitals are excluded from the
# comparison: their experimental assignments are not as clean.
EXPERIMENT = [13.54, 13.00, 12.73]

# The output of the final KI calculation, and of the last of the three
# initialization steps -- the spin-resolved PBE calculation whose eigenvalues
# are the PBE ones to compare against.
OUTPUTS = {
    "KI": Path("ozone/06-RunFinalKI/outputs/aiida.cpo"),
    "PBE": Path("ozone/04-dft_init_nspin2/outputs/aiida.cpo"),
}

HEADER = "Eigenvalues (eV), kp =   1 , spin =  1"


def filled_eigenvalues(output_file: Path) -> list[float]:
    """Return the filled orbital energies of the first spin channel, in eV.

    ``kcp.x`` prints them under an ``Eigenvalues`` header, and the empty
    states separately under one of their own, so the block that follows the
    last such header is the filled manifold.
    """
    if not output_file.is_file():
        raise SystemExit(f"No such file: {output_file}. Run `koopmans run ozone.yaml` first.")

    lines = output_file.read_text().splitlines()
    headers = [i for i, line in enumerate(lines) if line.strip() == HEADER]
    if not headers:
        raise SystemExit(f"{output_file} has no `{HEADER}` line; did the calculation finish?")

    energies: list[float] = []
    for line in lines[headers[-1] + 1 :]:
        if line.strip():
            energies.extend(float(value) for value in line.split())
        elif energies:
            break
    return energies


def main() -> None:
    """Read the two output files and write ozone_spectrum.svg."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for label, output_file in OUTPUTS.items():
        # The last three filled eigenvalues, in the same most-bound-to-HOMO
        # order as the experimental values; binding energy = -eigenvalue.
        energies = filled_eigenvalues(output_file)
        ax.scatter([-e for e in energies[-3:]], EXPERIMENT, label=label)

    lims = (7, 15)
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Calculated binding energy (eV)")
    ax.set_ylabel("Experimental binding energy (eV)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("ozone_spectrum.svg")


if __name__ == "__main__":
    main()
