"""Draw band-structure series onto one set of axes.

Sees only :class:`~koopmans.plotting.series.BandSeries` records, never AiiDA
nodes.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from koopmans.plotting.series import BandSeries

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "DIVIDER_LABEL",
    "draw_band_structures",
    "path_distances",
    "render_band_structures",
]

#: Label of the vertical rules drawn at interior special points. The leading
#: underscore keeps them out of the legend, and names them for anyone reading
#: the axes back.
DIVIDER_LABEL = "_path divider"

#: Special-point names spelled out by seekpath or ASE, and their symbols.
_SYMBOLS = {
    "G": "Γ",
    "GAMMA": "Γ",
    "DELTA": "Δ",
    "LAMBDA": "Λ",
    "SIGMA": "Σ",
}


def _format_label(name: str) -> str:
    """Return a high-symmetry point's name as it is drawn on the axis."""
    base, _, subscript = name.partition("_")
    text = _SYMBOLS.get(base.upper(), base)
    return f"{text}$_{{{subscript}}}$" if subscript else text


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


def _shared_cell(series: Sequence[BandSeries]) -> list[list[float]] | None:
    """Return the cell every series' path axis is measured in.

    Series on one figure run along one path, so one reciprocal basis measures
    them all. Taking the first cell available puts a series that carries none
    on the same x scale as the rest instead of on crystal coordinates.
    """
    for item in series:
        if item.cell is not None:
            return item.cell
    return None


def _segments(series: BandSeries) -> list[slice]:
    """Return the index ranges of the path's continuous stretches."""
    breaks = [int(index) + 1 for index in np.flatnonzero(_jumps(series))]
    bounds = [0, *breaks, len(series.kpoints)]
    return [slice(start, stop) for start, stop in pairwise(bounds)]


def _tick_source(series: Sequence[BandSeries]) -> BandSeries:
    """Return the series the axis takes its ticks from.

    The first one that names any high-symmetry points, so that a series
    carrying none does not cost the figure its axis.
    """
    return next((item for item in series if item.path_labels), series[0])


def _ticks(
    series: BandSeries, cell: list[list[float]] | None = None
) -> tuple[list[float], list[str]]:
    """Return the axis tick positions and names of one series' path.

    Two special points at the same position — the two sides of a jump — share
    one tick, named ``"X|Y"``.
    """
    distances = path_distances(series, cell)
    positions: list[float] = []
    names: list[str] = []
    for index, name in sorted(series.path_labels):
        if not 0 <= index < distances.size:
            continue
        position = float(distances[index])
        if positions and np.isclose(position, positions[-1]):
            names[-1] = f"{names[-1]}|{_format_label(name)}"
        else:
            positions.append(position)
            names.append(_format_label(name))
    return positions, names


def _path_extent(distances: Sequence[np.ndarray]) -> tuple[float, float] | None:
    """Return the first and last x the curves reach, or ``None`` if they reach none.

    ``None`` also stands for a path of zero length, which no limits can frame.
    """
    reached = [item for item in distances if item.size]
    if not reached:
        return None
    first = min(float(item[0]) for item in reached)
    last = max(float(item[-1]) for item in reached)
    return None if last <= first else (first, last)


def draw_band_structures(
    axes: Axes,
    series: Sequence[BandSeries],
    caption: str | None = None,
) -> None:
    """Draw every series onto one set of axes, shifted by its own ``zero``.

    Every series is measured in one reciprocal basis, and the ticks come from
    the first series that names any high-symmetry points; a jump breaks the
    curves rather than joining them across the gap. A rule marks each interior
    special point, and the x limits are the ends of the path itself.

    :param axes: where to draw.
    :param series: the curves, each already carrying the figure's ``zero``.
    :param caption: a sentence stating what the figure's zero is.
    """
    cell = _shared_cell(series)
    drawn_distances: list[np.ndarray] = []
    for index, item in enumerate(series):
        distances = path_distances(item, cell)
        drawn_distances.append(distances)
        energies = np.asarray(item.energies, dtype=np.float64) - item.zero
        color = f"C{index % 10}"
        drawn = False
        for span in _segments(item):
            for band in range(energies.shape[1]):
                axes.plot(
                    distances[span],
                    energies[span, band],
                    color=color,
                    linewidth=1.2,
                    label=None if drawn else item.label,
                )
                drawn = True

    positions, names = _ticks(_tick_source(series), cell)
    if positions:
        axes.set_xticks(positions)
        axes.set_xticklabels(names)
        # The first and last special points sit on the spines, which already
        # draw them.
        for position in positions[1:-1]:
            axes.axvline(position, color="0.8", linewidth=0.6, zorder=0, label=DIVIDER_LABEL)

    limits = _path_extent(drawn_distances)
    if limits is not None:
        axes.set_xlim(*limits)

    axes.set_ylabel(f"Energy ({series[0].units})")
    if caption is not None:
        # Only the first character changes: ``str.capitalize`` would lowercase
        # the rest, turning "KI" into "ki" and "eV" into "ev".
        axes.set_title(caption[:1].upper() + caption[1:], fontsize="small")
    axes.legend(frameon=False, fontsize="small")


def render_band_structures(
    series: Sequence[BandSeries],
    output_path: Path | None = None,
    show: bool = False,
    caption: str | None = None,
) -> None:
    """Draw the series and write or show the figure.

    :param series: the curves to draw, each already carrying its ``zero``.
    :param output_path: where to write the figure; the extension sets the
        format. ``None`` writes nothing.
    :param show: open an interactive window.
    :param caption: a sentence stating what the figure's zero is.
    """
    import matplotlib

    if not show:
        # Chosen before pyplot is imported: a run that only writes a file must
        # not depend on a display, so that it works over ssh and in CI.
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6.0, 4.5))
    draw_band_structures(axes, series, caption=caption)
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(figure)
