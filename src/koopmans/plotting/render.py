"""Draw band-structure series onto one set of axes.

Sees only :class:`~koopmans.plotting.series.BandSeries` records, never AiiDA
nodes.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from koopmans.plotting.series import (
    BandGap,
    BandSeries,
    EnergyZero,
    _jumps,
    band_gap,
    energy_axis_label,
    path_distances,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "DIVIDER_LABEL",
    "StyleError",
    "check_style",
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


#: How close two gap labels' midpoints may sit — as a fraction of the path's
#: drawn length in x, in eV in y — before the second is pushed to the arrow's
#: outer (left) side instead of the default right side, so the two do not
#: overlap.
_GAP_LABEL_TOL_FRACTION = 0.08
_GAP_LABEL_TOL_ENERGY = 0.5


def _gap_midpoint(item: BandSeries, edge: BandGap, distances: np.ndarray) -> tuple[float, float]:
    """Return one series' gap arrow midpoint, in the axes' data coordinates."""
    mid_x = (distances[edge.vbm_kpoint_index] + distances[edge.cbm_kpoint_index]) / 2
    mid_y = (edge.vbm + edge.cbm) / 2 - item.zero
    return float(mid_x), mid_y


def _draw_gap(
    axes: Axes,
    item: BandSeries,
    edge: BandGap,
    distances: np.ndarray,
    color: Any,
    outward: bool = False,
) -> None:
    """Draw one series' band gap: a double-headed arrow labelled with its value.

    The arrow runs from the valence band maximum to the conduction band
    minimum, at their own k-points and shifted energies; slanted for an
    indirect gap, vertical for a direct one. The label sits at the arrow's
    midpoint, offset to the right unless ``outward``, and never joins the
    legend.

    :param outward: offset the label to the left of the midpoint instead of
        the right, to clear another series' label whose arrow sits nearby.
    """
    from_x, to_x = float(distances[edge.vbm_kpoint_index]), float(distances[edge.cbm_kpoint_index])
    from_y, to_y = edge.vbm - item.zero, edge.cbm - item.zero
    mid_x, mid_y = _gap_midpoint(item, edge, distances)

    axes.annotate(
        "",
        xy=(to_x, to_y),
        xytext=(from_x, from_y),
        arrowprops={
            "arrowstyle": "<->",
            "color": color,
            "linewidth": 1.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )
    offset, alignment = ((-8, 0), "right") if outward else ((8, 0), "left")
    axes.annotate(
        f"{edge.value:.2f} {item.units}",
        xy=(mid_x, mid_y),
        xytext=offset,
        textcoords="offset points",
        va="center",
        ha=alignment,
        fontsize="small",
        color=color,
        annotation_clip=False,
        # A white backing keeps the label readable where it lands on a band —
        # the gap it measures is exactly where the curves are densest.
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1},
    )


def _draw_series_gap(
    axes: Axes,
    item: BandSeries,
    distances: np.ndarray,
    color: Any,
    path_length: float,
    previous_midpoints: list[tuple[float, float]],
) -> None:
    """Draw a series' gap annotation, skipping one that reports no edge.

    Nudges the label to the arrow's outer side when its midpoint would
    otherwise land within ``_GAP_LABEL_TOL_FRACTION``/``_GAP_LABEL_TOL_ENERGY``
    of an already-drawn gap's label.
    """
    try:
        edge = band_gap(item)
    except ValueError:
        return
    mid_x, mid_y = _gap_midpoint(item, edge, distances)
    tol_x = _GAP_LABEL_TOL_FRACTION * path_length
    outward = any(
        abs(mid_x - other_x) < tol_x and abs(mid_y - other_y) < _GAP_LABEL_TOL_ENERGY
        for other_x, other_y in previous_midpoints
    )
    _draw_gap(axes, item, edge, distances, color, outward)
    previous_midpoints.append((mid_x, mid_y))


def _draw_series_curves(
    axes: Axes,
    item: BandSeries,
    distances: np.ndarray,
    style: list[str],
    color: str | None,
) -> Any:
    """Plot one series' bands and return the color they were drawn in.

    ``None`` if the series drew no curve at all (an empty path).
    """
    energies = np.asarray(item.energies, dtype=np.float64) - item.zero
    drawn = False
    drawn_color: Any = None
    for span in _segments(item):
        for band in range(energies.shape[1]):
            (line,) = axes.plot(
                distances[span],
                energies[span, band],
                *style,
                linewidth=1.2,
                label=None if drawn else item.label,
            )
            if color is not None:
                line.set_color(color)
            drawn = True
            drawn_color = line.get_color()
    return drawn_color


class StyleError(ValueError):
    """A style is not one of matplotlib's format strings."""


def _format_parser() -> Any:
    """Return matplotlib's own parser for format strings.

    matplotlib publishes none, so this reads the private one rather than
    keeping a second copy of the grammar beside it. A release that renames it
    turns every ``--style`` into this message instead of a traceback.

    :raises StyleError: if this matplotlib no longer carries it.
    """
    try:
        from matplotlib.axes._base import (  # type: ignore[attr-defined]
            _process_plot_format,
        )
    except ImportError as exc:
        raise StyleError(
            "this version of matplotlib cannot say whether a format string is "
            "valid, so --style cannot be checked. Leave --style out, or install "
            "matplotlib 3.10 or similar."
        ) from exc
    return _process_plot_format


def _style_color(style: str) -> object | None:
    """Return the color a matplotlib format string names, or ``None`` for none.

    :raises StyleError: if matplotlib cannot read ``style`` as a format string.
    """
    parser = _format_parser()
    try:
        parsed = parser(style)
    except ValueError as exc:
        raise StyleError(str(exc)) from exc
    color: object | None = parsed[2]
    return color


def check_style(style: str) -> None:
    """Reject a string matplotlib cannot read as a format string.

    :raises StyleError: if matplotlib cannot read it.
    """
    _style_color(style)


def draw_band_structures(
    axes: Axes,
    series: Sequence[BandSeries],
    zero: EnergyZero = EnergyZero.NONE,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
    gap: bool = False,
) -> None:
    """Draw every series onto one set of axes, shifted by its own ``zero``.

    Every series is measured in one reciprocal basis, and the ticks come from
    the first series that names any high-symmetry points; a jump breaks the
    curves rather than joining them across the gap. A rule marks each interior
    special point, and the x limits are the ends of the path itself. Only an
    overlay carries a legend.

    A series carrying a ``style`` is drawn in that format string, color
    included; where the string names no color the series keeps the one these
    axes give it, so its bands are drawn in one color rather than in as many
    as it has bands.

    :param axes: where to draw.
    :param series: the curves, each already carrying the figure's ``zero``.
    :param zero: which energy was subtracted, which the y axis names.
    :param ylim: the energy range to show, in the shifted energies the axis
        is drawn in. ``None`` shows every band in full.
    :param legend: draw the key, or leave it out. ``None`` draws it for an
        overlay and leaves it out for a single curve.
    :param gap: annotate each series' band gap, skipping a series that
        reports no valence band edge.
    """
    cell = _shared_cell(series)
    drawn_distances: list[np.ndarray] = []
    gap_midpoints: list[tuple[float, float]] = []
    for index, item in enumerate(series):
        distances = path_distances(item, cell)
        drawn_distances.append(distances)
        style = [item.style] if item.style else []
        # One band is one plot call, and matplotlib advances its color cycle
        # once per call, so a series whose style names no color still has to be
        # given one — otherwise its bands come out in as many colors.
        names_color = bool(style) and _style_color(style[0]) is not None
        color = None if names_color else f"C{index % 10}"
        drawn_color = _draw_series_curves(axes, item, distances, style, color)

        if gap and drawn_color is not None:
            path_length = float(distances[-1] - distances[0]) if distances.size else 0.0
            _draw_series_gap(axes, item, distances, drawn_color, path_length, gap_midpoints)

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
    if ylim is not None:
        axes.set_ylim(*ylim)

    axes.set_ylabel(energy_axis_label(zero, series[0].units))
    # One curve needs no key to tell it from the others.
    wanted = len(series) > 1 if legend is None else legend
    if wanted:
        axes.legend(frameon=False, fontsize="small")


def render_band_structures(
    series: Sequence[BandSeries],
    output_path: Path | None = None,
    show: bool = False,
    zero: EnergyZero = EnergyZero.NONE,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
    gap: bool = False,
) -> None:
    """Draw the series and write or show the figure.

    :param series: the curves to draw, each already carrying its ``zero``.
    :param output_path: where to write the figure; the extension sets the
        format. ``None`` writes nothing.
    :param show: open an interactive window.
    :param zero: which energy was subtracted, which the y axis names.
    :param ylim: the energy range to show, in the shifted energies the axis
        is drawn in. ``None`` shows every band in full.
    :param legend: draw the key, or leave it out. ``None`` draws it for an
        overlay and leaves it out for a single curve.
    :param gap: annotate each series' band gap, skipping a series that
        reports no valence band edge.
    """
    import matplotlib

    if not show:
        # Chosen before pyplot is imported: a run that only writes a file must
        # not depend on a display, so that it works over ssh and in CI.
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6.0, 4.5))
    draw_band_structures(axes, series, zero=zero, ylim=ylim, legend=legend, gap=gap)
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(figure)
