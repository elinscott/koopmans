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
from typing import Any

import numpy as np

__all__ = [
    "BandGap",
    "BandSeries",
    "EnergyZero",
    "NoEnergyZeroError",
    "PathMismatchError",
    "SpectrumSeries",
    "apply_energy_zero",
    "band_gap",
    "check_paths_agree",
    "describe_energy_zero",
    "energy_axis_label",
    "path_distances",
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
    appearance to the figure. ``show_gap`` asks the figure to annotate this
    series' band gap; it is silently left undrawn when the series reports no
    valence band edge.
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
    show_gap: bool = False

    def reference(self, kind: EnergyZero) -> float | None:
        """Return the energy this series would put at zero, or ``None``."""
        if kind == EnergyZero.VBM:
            return self.vbm
        if kind == EnergyZero.FERMI:
            return self.fermi
        return 0.0


def _labelled(series: BandSeries) -> np.ndarray:
    """Return a boolean mask of the k-points carrying a high-symmetry label."""
    mask = np.zeros(len(series.kpoints), dtype=bool)
    for index, _ in series.path_labels:
        if 0 <= index < mask.size:
            mask[index] = True
    return mask


def _jumps(series: BandSeries) -> np.ndarray:
    """Return a mask over steps that are jumps rather than steps along the path.

    Two consecutive k-points that both carry a high-symmetry label sit at a
    discontinuity: the path stops at one special point and restarts at another.
    A branch sampled at its two endpoints alone is indistinguishable from one,
    and is read as a jump; aiida-core's own band plotting reads it the same way.
    """
    labelled = _labelled(series)
    if labelled.size < 2:
        return np.zeros(max(labelled.size - 1, 0), dtype=bool)
    return np.asarray(labelled[:-1] & labelled[1:], dtype=bool)


def path_distances(series: BandSeries, cell: list[list[float]] | None = None) -> np.ndarray:
    """Return the cumulative distance along the path of each k-point.

    Distance is measured in the reciprocal basis ``cell`` defines, defaulting
    to the series' own; with no cell at all the crystal coordinates stand in,
    which distorts the relative lengths of the path's segments. A jump
    contributes no distance.
    """
    kpoints = np.asarray(series.kpoints, dtype=np.float64)
    if len(kpoints) == 0:
        return np.zeros(0)
    frame = series.cell if cell is None else cell
    if frame is None:
        cartesian = kpoints
    else:
        reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(frame, dtype=np.float64)).T
        cartesian = kpoints @ reciprocal
    steps = np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
    steps[_jumps(series)] = 0.0
    return np.concatenate(([0.0], np.cumsum(steps)))


@dataclass
class BandGap:
    """A series' valence-to-conduction gap.

    ``value``, ``vbm`` and ``cbm`` are in the series' own units, as computed —
    before the figure's zero. ``vbm_kpoint_index``/``cbm_kpoint_index`` are the
    k-point each edge sits at; ``vbm_distance``/``cbm_distance`` are that same
    point's position along the series' own path (:func:`path_distances`).
    ``direct`` is whether the two indices agree.
    """

    value: float
    vbm: float
    cbm: float
    vbm_kpoint_index: int
    cbm_kpoint_index: int
    vbm_distance: float
    cbm_distance: float
    direct: bool


#: How far above the reported valence band maximum a state must sit to count
#: as the conduction band minimum rather than the same band as the edge.
_GAP_TOLERANCE = 1e-6

#: The smallest valence-to-conduction separation read as an insulating gap.
#: Below it, the "conduction" state is a metal's own partially filled band
#: sampled at another k-point, not a real gap — QE's own occupation smearing
#: routinely leaves states this close together at the Fermi level.
_MIN_GAP = 1e-3


def _nearest_pair(
    vbm_kpoints: np.ndarray, cbm_kpoints: np.ndarray, distances: np.ndarray
) -> tuple[int, int]:
    """Return the VBM/CBM k-point pair closest together along the path.

    A high-symmetry point sampled at both ends of the path (Γ opening and
    closing a loop, say) attains the same energy at more than one k-point;
    picking the wrong one draws the gap arrow across the whole figure
    instead of at the band edge it belongs to. Ties keep the pair the
    ascending scan meets first.
    """
    best_pair = (int(vbm_kpoints[0]), int(cbm_kpoints[0]))
    best_distance = np.inf
    for vbm_kpoint in vbm_kpoints:
        for cbm_kpoint in cbm_kpoints:
            separation = abs(float(distances[cbm_kpoint]) - float(distances[vbm_kpoint]))
            if separation < best_distance:
                best_distance = separation
                best_pair = (int(vbm_kpoint), int(cbm_kpoint))
    return best_pair


def band_gap(item: BandSeries) -> BandGap:
    """Return the series' valence-to-conduction gap.

    The valence band maximum is ``item.vbm``; the conduction band minimum is
    the lowest energy more than ``_GAP_TOLERANCE`` above it, provided that
    exceeds ``_MIN_GAP`` — otherwise the two are read as the same partially
    filled band sampled at different k-points (a metal), not an insulating
    gap. A high-symmetry point the path visits more than once can attain
    either energy at several k-points alike, within ``_GAP_TOLERANCE``; the
    pair reported is whichever of those sits closest together along the
    path, so the gap is drawn at the band edge rather than stretched between
    unrelated repeats of the same point.

    :raises ValueError: if the series reports no valence band edge, no state
        above it, or a gap no wider than a metal's own dispersion.
    """
    if item.vbm is None:
        raise ValueError(f"'{item.label}' reports no valence band edge to measure a gap from.")

    energies = np.asarray(item.energies, dtype=np.float64)
    vbm_kpoints = np.flatnonzero(np.any(np.abs(energies - item.vbm) <= _GAP_TOLERANCE, axis=1))
    if vbm_kpoints.size == 0:
        # Numerical drift between the reported edge and the band table
        # itself: fall back to the single closest k-point rather than
        # matching nothing.
        vbm_kpoints = np.array([int(np.argmin(np.abs(energies - item.vbm).min(axis=1)))])

    above = np.where(energies > item.vbm + _GAP_TOLERANCE, energies, np.inf)
    cbm = float(np.min(above))
    if not np.isfinite(cbm):
        raise ValueError(
            f"'{item.label}' reports no state above its valence band maximum to measure a gap to."
        )
    if cbm - item.vbm <= _MIN_GAP:
        raise ValueError(
            f"'{item.label}' reports no band gap: the state above its valence "
            "band maximum sits within a metal's own partially filled band."
        )
    cbm_kpoints = np.flatnonzero(np.any(np.abs(energies - cbm) <= _GAP_TOLERANCE, axis=1))

    distances = path_distances(item)
    vbm_kpoint, cbm_kpoint = _nearest_pair(vbm_kpoints, cbm_kpoints, distances)

    return BandGap(
        value=cbm - item.vbm,
        vbm=item.vbm,
        cbm=cbm,
        vbm_kpoint_index=vbm_kpoint,
        cbm_kpoint_index=cbm_kpoint,
        vbm_distance=float(distances[vbm_kpoint]),
        cbm_distance=float(distances[cbm_kpoint]),
        direct=vbm_kpoint == cbm_kpoint,
    )


@dataclass
class SpectrumSeries:
    """One optical absorption spectrum on the axes.

    ``energies`` are eV; ``im_eps``/``re_eps`` are the macroscopic dielectric
    function a yambo BSE run computes with local-field and excitonic effects
    included. ``im_eps_o``/``re_eps_o`` are the independent-particle spectrum
    the same run reports, ``None`` when it reported none. ``style`` is the
    matplotlib format string the curve is drawn in, ``None`` leaving its
    appearance to the figure.
    """

    label: str
    energies: list[float]
    im_eps: list[float]
    re_eps: list[float]
    im_eps_o: list[float] | None = None
    re_eps_o: list[float] | None = None
    style: str | None = None


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


def _series_record(item: BandSeries | SpectrumSeries) -> dict[str, Any]:
    """Return one series' JSON record, with its gap if it is a band structure reporting an edge.

    ``gap`` is written whether or not the figure was asked to draw one, so a
    script can read the gap off the file without asking for the annotation;
    ``show_gap`` itself, which only says whether the figure drew it, is left
    out to keep this key's shape the same either way.
    """
    record = asdict(item)
    if isinstance(item, BandSeries):
        record.pop("show_gap", None)
        try:
            record["gap"] = asdict(band_gap(item))
        except ValueError:
            record["gap"] = None
    return record


def write_series_json(series: Sequence[BandSeries] | Sequence[SpectrumSeries], path: Path) -> None:
    """Write the records a figure was drawn from as JSON.

    Works on either a band-structure or a spectrum figure's records alike, both
    being plain dataclasses. Energies are as computed; a ``BandSeries``' zero
    records the shift the figure applied, so the file is enough to redraw the
    figure or to restyle it elsewhere.
    """
    payload = {"series": [_series_record(item) for item in series]}
    path.write_text(json.dumps(payload, indent=2) + "\n")
