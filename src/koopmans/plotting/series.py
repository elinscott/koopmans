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
    "PathMismatchError",
    "apply_energy_zero",
    "check_paths_agree",
    "describe_energy_zero",
    "energy_axis_label",
    "write_series_json",
]


class EnergyZero(StrEnum):
    """Which energy the axes put at zero."""

    VBM = "vbm"
    FERMI = "fermi"
    NONE = "none"


class NoEnergyZeroError(Exception):
    """No series on the axes reports the requested reference energy."""


class PathMismatchError(Exception):
    """The series were computed along different k-point paths."""


@dataclass
class BandSeries:
    """One band structure on the axes.

    ``energies`` are ``len(kpoints)`` rows of band energies in ``units``, as
    computed; ``zero`` is the shift the figure subtracts from them. ``kpoints``
    are crystal coordinates of the reciprocal basis ``cell`` defines; without a
    cell the path can only be measured in those coordinates, which distorts the
    relative lengths of its segments. ``path_labels`` pairs a k-point index
    with the name of the high-symmetry point sitting there. ``style`` is the
    matplotlib format string the curve is drawn in, ``None`` leaving its
    appearance to the figure.
    """

    label: str
    kpoints: list[list[float]]
    energies: list[list[float]]
    cell: list[list[float]] | None = None
    path_labels: list[tuple[int, str]] = field(default_factory=list)
    units: str = "eV"
    style: str | None = None
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


#: How far apart two crystal coordinates may be and still name the same point.
PATH_TOLERANCE = 1e-4


def _special_points(item: BandSeries) -> list[tuple[str, tuple[float, ...]]]:
    """Return the series' high-symmetry points and their coordinates, in order."""
    return [
        (name, tuple(item.kpoints[index]))
        for index, name in sorted(item.path_labels)
        if 0 <= index < len(item.kpoints)
    ]


def _describe_path(item: BandSeries) -> str:
    """Return the series' path as its named corners, for an error message."""
    points = _special_points(item)
    if not points:
        return "no high-symmetry points"
    return " -> ".join(
        f"{name} ({', '.join(format(x, '.4g') for x in coordinates)})"
        for name, coordinates in points
    )


def check_paths_agree(series: Sequence[BandSeries], tolerance: float = PATH_TOLERANCE) -> None:
    """Reject series computed along different k-point paths.

    Only the high-symmetry points are compared, by coordinate rather than by
    name, so two runs may sample one path at different densities and may spell
    its corners differently. Series that do not share a path share one x axis
    all the same, and the figure looks right while it is not.

    :raises PathMismatchError: if any series' corners differ from the first's.
    """
    if len(series) < 2:
        return
    reference, *rest = series
    expected = _special_points(reference)
    for item in rest:
        found = _special_points(item)
        if len(found) == len(expected) and all(
            abs(x - y) <= tolerance
            for (_, first), (_, second) in zip(expected, found, strict=True)
            for x, y in zip(first, second, strict=True)
        ):
            continue
        raise PathMismatchError(
            f"'{reference.label}' and '{item.label}' were computed along different "
            "k-point paths, so they cannot share one axis:\n"
            f"  {reference.label}: {_describe_path(reference)}\n"
            f"  {item.label}: {_describe_path(item)}\n"
            "Plot them separately, or give both runs the same `kpoints: {path: ...}`."
        )


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


#: How the y axis names each choice of zero.
_ENERGY_SYMBOLS = {
    EnergyZero.VBM: "$E - E_\\mathrm{VBM}$",
    EnergyZero.FERMI: "$E - E_\\mathrm{F}$",
    EnergyZero.NONE: "Energy",
}


def energy_axis_label(kind: EnergyZero, units: str = "eV") -> str:
    """Return the y-axis label, naming the energy the figure subtracted."""
    return f"{_ENERGY_SYMBOLS[kind]} ({units})"


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
