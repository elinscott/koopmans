"""Plot the KI and PBE binding energies of ozone against photoemission.

Run this script in the directory that contains the ``ozone/`` output directory produced
by ``koopmans run ozone.json``. It writes ``ozone_spectrum.svg``.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PBE_OUTPUT = Path("ozone/06-dft_init_nspin2/01-dft_init-KcpCalculation/outputs/aiida.cpo")
KI_OUTPUT = Path("ozone/08-RunFinalKI/01-ki_final-KcpCalculation/outputs/aiida.cpo")

# Experimental binding energies (eV) of the three outermost occupied orbitals
# of ozone, ordered from the most tightly bound to the HOMO, from gas-phase
# photoemission: Mocellin et al., Chem. Phys. Lett. 375, 76 (2003),
# doi:10.1016/S0009-2614(03)00818-2. Deeper orbitals are excluded from the
# comparison: their experimental assignments are not as clean.
EXPERIMENT = [13.54, 13.00, 12.73]


def read_filled_eigenvalues(output_file: Path) -> list[float]:
    """Return the filled-orbital energies in eV from a kcp.x output file."""
    text = output_file.read_text()
    return _section(text, r"^ *Eigenvalues \(eV\), kp = +1 , spin = +1 *$")


def _section(text: str, header_pattern: str) -> list[float]:
    """Return the floats listed under the last occurrence of a section header."""
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if re.match(header_pattern, line)]
    if not headers:
        raise ValueError(f"No line matching {header_pattern!r} found")
    values: list[float] = []
    for line in lines[headers[-1] + 1 :]:
        if not line.strip():
            if values:
                break
            continue
        values.extend(float(v) for v in line.split())
    return values


def main() -> None:
    """Read the two output files and write ozone_spectrum.svg."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    for output_file, label in [(KI_OUTPUT, "KI"), (PBE_OUTPUT, "PBE")]:
        # The last three filled eigenvalues, in the same most-bound-to-HOMO
        # order as the experimental values; binding energy = -eigenvalue.
        binding_energies = [-e for e in read_filled_eigenvalues(output_file)[-3:]]
        ax.scatter(binding_energies, EXPERIMENT, label=label)

    lims = [7, 15]
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
