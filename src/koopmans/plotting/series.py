"""The records a band-structure figure is drawn from.

A record holds everything needed to draw one curve and nothing about where
it came from, so the same figure can later be drawn from a portable dump
rather than from the AiiDA database.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

__all__ = [
    "BandSeries",
    "EnergyZero",
    "NoEnergyZeroError",
    "apply_energy_zero",
    "apply_labels",
    "describe_energy_zero",
    "write_series_json",
]


class EnergyZero(StrEnum):
    """Which energy the axes put at zero."""

    VBM = "vbm"
    FERMI = "fermi"
    NONE = "none"


class NoEnergyZeroError(Exception):
    """No series on the axes reports the requested reference energy."""


@dataclass
class BandSeries:
    """One band structure on the axes.

    ``energies`` are ``len(kpoints)`` rows of band energies in ``units``, as
    computed; ``zero`` is the shift the figure subtracts from them. ``kpoints``
    are crystal coordinates of the reciprocal basis ``cell`` defines; without a
    cell the path can only be measured in those coordinates, which distorts the
    relative lengths of its segments. ``path_labels`` pairs a k-point index
    with the name of the high-symmetry point sitting there.
    """

    label: str
    kpoints: list[list[float]]
    energies: list[list[float]]
    cell: list[list[float]] | None = None
    path_labels: list[tuple[int, str]] = field(default_factory=list)
    units: str = "eV"
    vbm: float | None = None
    fermi: float | None = None
    zero: float = 0.0

    def reference(self, kind: EnergyZero) -> float | None:
        """Return the energy this series would put at zero, or ``None``."""
        if kind == EnergyZero.VBM:
            return self.vbm
        if kind == EnergyZero.FERMI:
            return self.fermi
        return 0.0


def apply_labels(series: Sequence[BandSeries], labels: Sequence[str]) -> None:
    """Rename the leading series, in order.

    :raises ValueError: if more labels are given than there are series.
    """
    if len(labels) > len(series):
        raise ValueError(
            f"{len(labels)} --label values were given but only {len(series)} band "
            f"structure(s) were found; labels are applied in order."
        )
    for item, label in zip(series, labels, strict=False):
        item.label = label


#: What each choice of zero is called in prose.
_ENERGY_NAMES = {
    EnergyZero.VBM: "valence band edge",
    EnergyZero.FERMI: "Fermi level",
    EnergyZero.NONE: "zero",
}


def apply_energy_zero(
    series: Sequence[BandSeries], kind: EnergyZero
) -> tuple[float, BandSeries | None]:
    """Set every series' ``zero`` from the first series that reports one.

    One shift governs the whole figure. Referencing each series to its own
    valence band edge would subtract away the band-edge shift between them,
    which is the physical result an overlay exists to show.

    :return: the shift, and the series it came from (``None`` for no zero).
    :raises NoEnergyZeroError: if no series reports the requested energy.
    """
    if kind == EnergyZero.NONE:
        for item in series:
            item.zero = 0.0
        return 0.0, None

    for candidate in series:
        value = candidate.reference(kind)
        if value is not None:
            for item in series:
                item.zero = value
            return value, candidate

    alternatives = " or ".join(f"--zero {other.value}" for other in EnergyZero if other != kind)
    raise NoEnergyZeroError(
        f"None of the {len(series)} band structure(s) found reports a "
        f"{_ENERGY_NAMES[kind]}, so `--zero {kind.value}` has nothing to subtract. "
        f"Use {alternatives}."
    )


def describe_energy_zero(
    kind: EnergyZero, value: float, reference: BandSeries | None, units: str = "eV"
) -> str:
    """Return a sentence fragment stating what the figure's zero is."""
    if reference is None:
        return "energies as computed, no zero applied"
    return (
        f"energies relative to the {_ENERGY_NAMES[kind]} of "
        f"'{reference.label}' at {value:.4f} {units}"
    )


def write_series_json(series: Sequence[BandSeries], path: Path) -> None:
    """Write the records the figure was drawn from as JSON.

    Energies are as computed; ``zero`` records the shift the figure applied,
    so the file is enough to redraw the figure or to restyle it elsewhere.
    """
    payload = {"series": [asdict(item) for item in series]}
    path.write_text(json.dumps(payload, indent=2) + "\n")
