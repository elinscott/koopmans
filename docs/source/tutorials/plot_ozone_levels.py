"""Plot the PBE and KI orbital energies of ozone from a completed tutorial run.

Run this script in the directory that contains the ``ozone/`` output directory produced
by ``koopmans run ozone.json``. It writes ``ozone_levels.svg``.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PBE_OUTPUT = Path(
    "ozone/06-dft_init_nspin2-WorkGraph<dft_init_nspin2>/01-dft_init-KcpCalculation/outputs/aiida.cpo"
)
KI_OUTPUT = Path(
    "ozone/08-RunFinalKI-WorkGraph<RunFinalKI>/01-ki_final-KcpCalculation/outputs/aiida.cpo"
)

# Experimental references: the ionization potential from photoionization
# (NIST WebBook) and the electron affinity from photoelectron spectroscopy
# of the ozone anion, plotted as -IP and -EA.
EXPERIMENT_HOMO = -12.53
EXPERIMENT_LUMO = -2.10


def read_eigenvalues(output_file: Path) -> tuple[list[float], list[float]]:
    """Return the (filled, empty) orbital energies in eV from a kcp.x output file."""
    text = output_file.read_text()
    filled = _section(text, r"^ *Eigenvalues \(eV\), kp = +1 , spin = +1 *$")
    empty = _section(text, r"^ *Empty States Eigenvalues \(eV\), kp = +1 , spin = +1 *$")
    return filled, empty


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


def plot_levels(ax, x: float, filled: list[float], empty: list[float], color: str) -> None:
    """Draw one column of energy levels: filled orbitals solid, empty dashed."""
    for e in filled:
        ax.hlines(e, x - 0.3, x + 0.3, color=color)
    for e in empty:
        ax.hlines(e, x - 0.3, x + 0.3, color=color, linestyles="dashed")


def main() -> None:
    """Read the two output files and write ozone_levels.svg."""
    fig, ax = plt.subplots(figsize=(4.5, 5.0))

    for x, output_file in enumerate([PBE_OUTPUT, KI_OUTPUT]):
        filled, empty = read_eigenvalues(output_file)
        plot_levels(ax, x, filled, empty, "tab:blue")
    plot_levels(ax, 2, [EXPERIMENT_HOMO], [EXPERIMENT_LUMO], "tab:red")

    ax.set_xticks([0, 1, 2], ["PBE", "KI", "experiment"])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("Orbital energy (eV)")
    fig.tight_layout()
    fig.savefig("ozone_levels.svg")


if __name__ == "__main__":
    main()
