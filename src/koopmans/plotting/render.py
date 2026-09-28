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
    ParityQuantity,
    ParitySeries,
    SpectrumSeries,
    _jumps,
    band_gap,
    energy_axis_label,
    parity_axis_label,
    parity_metrics,
    parity_residuals,
    path_distances,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "DIVIDER_LABEL",
    "EMPTY_LABEL",
    "OCCUPIED_LABEL",
    "StyleError",
    "check_style",
    "draw_band_structures",
    "draw_parity",
    "draw_spectra",
    "path_distances",
    "render_band_structures",
    "render_parity",
    "render_spectra",
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


def _draw_gap(
    axes: Axes,
    item: BandSeries,
    edge: BandGap,
    distances: np.ndarray,
    color: Any,
) -> None:
    """Draw one series' band gap in the conventional textbook form.

    A vertical double-headed arrow, at the conduction band minimum's own
    k-point, runs from the valence band maximum's energy to the conduction
    band minimum's; for an indirect gap a dashed rule at the valence level
    marks where it sits, reaching from its own k-point to the arrow. A
    direct gap needs no such rule, since the arrow's own foot already sits
    at the valence band maximum's k-point. The label reads the gap's value
    to the right of the arrow, and never joins the legend. Every piece is
    clipped to the axes, so a y-range that excludes an edge cuts the arrow
    or the label off at the frame rather than drawing it past it.

    Two series whose conduction band minima coincide draw their arrows on
    top of each other rather than displaced — a displaced arrow would claim
    the conduction band minimum sits somewhere it does not.
    """
    vbm_x = float(distances[edge.vbm_kpoint_index])
    arrow_x = float(distances[edge.cbm_kpoint_index])
    vbm_y, cbm_y = edge.vbm - item.zero, edge.cbm - item.zero

    if not edge.direct:
        axes.plot(
            [vbm_x, arrow_x],
            [vbm_y, vbm_y],
            linestyle="--",
            linewidth=1.0,
            color=color,
            alpha=0.5,
        )

    arrow = axes.annotate(
        "",
        xy=(arrow_x, cbm_y),
        xytext=(arrow_x, vbm_y),
        arrowprops={
            "arrowstyle": "<->",
            "color": color,
            "linewidth": 1.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=True,
        clip_on=True,
    )
    # clip_on above clips the (empty) text; the arrow is a separate artist
    # that needs its own clip path to stop at the axes too.
    if arrow.arrow_patch is not None:
        arrow.arrow_patch.set_clip_path(axes.patch)
    axes.annotate(
        f"{edge.value:.2f} {item.units}",
        xy=(arrow_x, (vbm_y + cbm_y) / 2),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize="small",
        color=color,
        annotation_clip=True,
        clip_on=True,
        # A white backing keeps the label readable where it lands on a band —
        # the gap it measures is exactly where the curves are densest.
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1},
    )


def _draw_gaps(
    axes: Axes,
    candidates: Sequence[tuple[BandSeries, BandGap, np.ndarray, Any]],
) -> None:
    """Draw every series' gap annotation, each at its own conduction band minimum."""
    for item, edge, distances, color in candidates:
        _draw_gap(axes, item, edge, distances, color)


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


def _parse_style(style: str) -> tuple[Any, Any, Any]:
    """Return the line style, marker and color a matplotlib format string names.

    :raises StyleError: if matplotlib cannot read ``style`` as a format string.
    """
    parser = _format_parser()
    try:
        parsed: tuple[Any, Any, Any] = parser(style)
    except ValueError as exc:
        raise StyleError(str(exc)) from exc
    return parsed


def _style_color(style: str) -> object | None:
    """Return the color a matplotlib format string names, or ``None`` for none.

    :raises StyleError: if matplotlib cannot read ``style`` as a format string.
    """
    color: object | None = _parse_style(style)[2]
    return color


def check_style(style: str) -> None:
    """Reject a string matplotlib cannot read as a format string.

    :raises StyleError: if matplotlib cannot read it.
    """
    _style_color(style)


def _cycle_color(style: Sequence[str], index: int) -> str | None:
    """Return the color to force a plot call to, or ``None`` to keep matplotlib's own.

    matplotlib advances its color cycle once per plot call, so a curve whose
    style names no color still needs one assigned, or its bands or spectra
    come out in as many colors as they have plot calls. A style that already
    names a color is left alone.
    """
    if style and _style_color(style[0]) is not None:
        return None
    return f"C{index % 10}"


def draw_band_structures(
    axes: Axes,
    series: Sequence[BandSeries],
    zero: EnergyZero = EnergyZero.NONE,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
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
    as it has bands. A series with ``show_gap`` set draws its band gap —
    skipped silently if it reports no valence band edge — as an arrow at its
    own conduction-band-minimum k-point; two series whose minima coincide
    draw their arrows on top of each other.

    :param axes: where to draw.
    :param series: the curves, each already carrying the figure's ``zero``.
    :param zero: which energy was subtracted, which the y axis names.
    :param ylim: the energy range to show, in the shifted energies the axis
        is drawn in. ``None`` shows every band in full.
    :param legend: draw the key, or leave it out. ``None`` draws it for an
        overlay and leaves it out for a single curve.
    """
    cell = _shared_cell(series)
    drawn_distances: list[np.ndarray] = []
    gap_candidates: list[tuple[BandSeries, BandGap, np.ndarray, Any]] = []
    for index, item in enumerate(series):
        distances = path_distances(item, cell)
        drawn_distances.append(distances)
        style = [item.style] if item.style else []
        color = _cycle_color(style, index)
        drawn_color = _draw_series_curves(axes, item, distances, style, color)

        if item.show_gap and drawn_color is not None:
            try:
                edge = band_gap(item)
            except ValueError:
                pass
            else:
                gap_candidates.append((item, edge, distances, drawn_color))

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
    _draw_gaps(axes, gap_candidates)
    if ylim is not None:
        axes.set_ylim(*ylim)

    axes.set_ylabel(energy_axis_label(zero, series[0].units))
    # One curve needs no key to tell it from the others.
    wanted = len(series) > 1 if legend is None else legend
    if wanted:
        axes.legend(frameon=False, fontsize="small")


#: Label of the independent-particle overlay, shared across every series so
#: it appears once in the legend no matter how many spectra are drawn.
_IP_LABEL = "independent particle"


def draw_spectra(
    axes: Axes,
    series: Sequence[SpectrumSeries],
    real: bool = False,
    ip: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
) -> None:
    """Draw every optical spectrum onto one set of axes.

    Draws Im ε against energy, one curve per series, or Re ε with ``real``.
    ``ip`` overlays each series' independent-particle spectrum, where it
    reported one, as a lighter dashed curve in the same color. The x axis is
    tight to the energies drawn, with no margin, unless ``xlim`` overrides it.
    Im ε's y axis starts at 0; Re ε goes negative, so its automatic limits are
    left alone unless ``ylim`` overrides them.

    :param axes: where to draw.
    :param series: the spectra to draw.
    :param real: draw Re ε instead of Im ε.
    :param ip: overlay the independent-particle spectrum.
    :param xlim: the energy range to show, in eV. ``None`` is tight to the
        energies drawn.
    :param ylim: the range to show. ``None`` starts Im ε at 0 and leaves
        Re ε automatic.
    :param legend: draw the key, or leave it out. ``None`` always draws it.
    """
    quantity = "re_eps" if real else "im_eps"
    quantity_o = "re_eps_o" if real else "im_eps_o"
    symbol = "Re" if real else "Im"

    drawn_energies: list[np.ndarray] = []
    for index, item in enumerate(series):
        energies = np.asarray(item.energies, dtype=np.float64)
        drawn_energies.append(energies)
        values = np.asarray(getattr(item, quantity), dtype=np.float64)
        style = [item.style] if item.style else []
        color = _cycle_color(style, index)
        (line,) = axes.plot(energies, values, *style, linewidth=1.2, label=item.label)
        if color is not None:
            line.set_color(color)
        drawn_color = line.get_color()

        if ip:
            independent = getattr(item, quantity_o)
            if independent is not None:
                axes.plot(
                    energies,
                    np.asarray(independent, dtype=np.float64),
                    linestyle="--",
                    linewidth=1.0,
                    color=drawn_color,
                    alpha=0.6,
                    label=_IP_LABEL if index == 0 else None,
                )

    axes.set_xlabel("Energy (eV)")
    axes.set_ylabel(rf"{symbol} $\varepsilon$")

    if xlim is not None:
        axes.set_xlim(*xlim)
    elif drawn_energies:
        all_energies = np.concatenate(drawn_energies)
        axes.set_xlim(float(all_energies.min()), float(all_energies.max()))

    if ylim is not None:
        axes.set_ylim(*ylim)
    elif not real:
        axes.set_ylim(bottom=0)

    wanted = True if legend is None else legend
    if wanted:
        axes.legend(frameon=False, fontsize="small")


#: Legend entries for the two markers, shared across every series so each
#: appears once however many runs are drawn.
OCCUPIED_LABEL = "occupied"
EMPTY_LABEL = "empty"

#: How many points a panel draws before its markers are thinned down and made
#: translucent, so that a long trajectory reads as a cloud rather than a blot.
_CROWDED = 400

#: The margin a parity panel's frame keeps beyond its furthest point, as a
#: fraction of the range it spans, so that point sits inside the frame
#: rather than on it.
_FRAME_MARGIN = 0.15


def _marker_size(points: int) -> float:
    """Return the marker size a panel of this many points is drawn at."""
    return 4.0 if points > _CROWDED else 6.0


def _split_by_occupancy(
    item: ParitySeries, x: np.ndarray, y: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """Return the point groups one series is drawn as, and whether each is filled.

    One group when the run reports no occupancy, otherwise the occupied
    and the empty orbitals apart, so that a reader can see which of the
    two the model is getting wrong.
    """
    if item.filled is None or len(item.filled) != x.size:
        return [(x, y, True)]
    mask = np.asarray(item.filled, dtype=bool)
    return [
        (x[selected], y[selected], fill)
        for selected, fill in ((mask, True), (~mask, False))
        if selected.any()
    ]


def _occupancy_legend_handles() -> list[Any]:
    """Return the key's two entries: the filled marker and the open one."""
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color="0.3",
            markerfacecolor="0.3" if fill else "none",
            label=name,
        )
        for name, fill in ((OCCUPIED_LABEL, True), (EMPTY_LABEL, False))
    ]


#: The fraction of each axis a corner label's own footprint is assumed to
#: cover, for deciding whether a point would sit under it. Wider than the
#: label actually draws, since a point right at the true corner would
#: otherwise count as belonging to whichever wider quadrant is emptiest
#: while still sitting under the label itself.
_CORNER_FRACTION = 0.25


def _corner_boxes(
    xlim: tuple[float, float], ylim: tuple[float, float]
) -> dict[str, tuple[float, float, float, float]]:
    """Return each corner's own ``(xmin, xmax, ymin, ymax)`` footprint."""
    xspan = (xlim[1] - xlim[0]) * _CORNER_FRACTION
    yspan = (ylim[1] - ylim[0]) * _CORNER_FRACTION
    return {
        "upper right": (xlim[1] - xspan, xlim[1], ylim[1] - yspan, ylim[1]),
        "upper left": (xlim[0], xlim[0] + xspan, ylim[1] - yspan, ylim[1]),
        "lower left": (xlim[0], xlim[0] + xspan, ylim[0], ylim[0] + yspan),
        "lower right": (xlim[1] - xspan, xlim[1], ylim[0], ylim[0] + yspan),
    }


def _corner_loc(
    x: np.ndarray, y: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]
) -> str:
    """Return the axes corner whose own footprint holds the fewest points.

    Counted inside each corner's own small footprint rather than the
    quadrant it sits in, so a lone point right at an otherwise sparse
    quadrant's corner still rules that corner out.
    """
    counts = {
        loc: int(np.sum((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)))
        for loc, (xmin, xmax, ymin, ymax) in _corner_boxes(xlim, ylim).items()
    }
    return min(counts, key=lambda loc: counts[loc])


#: The scale and unit the MAE/RMSE annotation reports each quantity in.
#: Eigenvalue errors are a small fraction of an eV, so meV reads more
#: naturally than a string of leading zeros.
_ANNOTATION_UNITS = {
    ParityQuantity.ALPHAS: (1.0, ""),
    ParityQuantity.EIGENVALUES: (1000.0, "meV"),
}


def _metrics_text(item: ParitySeries) -> str:
    """Return the run's mean absolute and root-mean-square error, as one line."""
    metrics = parity_metrics(item)
    scale, unit = _ANNOTATION_UNITS[item.quantity]
    suffix = f" {unit}" if unit else ""
    return f"MAE {metrics.mae * scale:.2g}{suffix}, RMSE {metrics.rmse * scale:.2g}{suffix}"


def _annotate_metrics(axes: Axes, item: ParitySeries, loc: str) -> None:
    """Write the run's mean absolute and root-mean-square error at ``loc``."""
    from matplotlib.offsetbox import AnchoredText

    anchored = AnchoredText(
        _metrics_text(item), loc=loc, frameon=False, prop={"fontsize": "x-small"}
    )
    axes.add_artist(anchored)


def _draw_marginal(
    marginal: Axes, residuals: np.ndarray, color: Any, bins: Sequence[float]
) -> None:
    """Draw the run's residual histogram beside the panel it belongs to, filled."""
    marginal.hist(
        residuals,
        bins=bins,
        orientation="horizontal",
        histtype="stepfilled",
        color=color,
        alpha=0.4,
    )
    marginal.axhline(0.0, color="0.4", linewidth=0.8, zorder=0)
    marginal.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
    for side in ("top", "right", "bottom"):
        marginal.spines[side].set_visible(False)


def draw_parity(
    axes: Axes,
    item: ParitySeries,
    residuals: bool = False,
    marginal: Axes | None = None,
) -> bool:
    """Draw one panel of a run's predicted values against what it computed.

    By default the predicted value is drawn against the computed one, with
    the identity line a perfect model would sit on; with ``residuals`` the
    difference is drawn against the computed value instead, with a rule at
    zero and the y axis symmetric about it. Occupied and empty orbitals are
    told apart by a filled and an open marker; a run that reports no
    occupancy is drawn filled throughout. Drawing the occupancy key itself
    is left to the caller, since one key serves every panel of a figure;
    the mean absolute and root-mean-square error are annotated in whichever
    corner has no point in its own footprint, or the least crowded one if
    every corner has one.

    :param axes: where to draw.
    :param item: the run to draw, holding one quantity.
    :param residuals: draw predicted minus computed instead of predicted.
    :param marginal: where to draw the residual histogram; ``None`` draws
        none. Expected to share this panel's y axis.
    :return: whether the run's orbitals split by occupancy, so the caller
        knows whether an occupancy key is worth drawing.
    """
    quantity = item.quantity
    computed = np.asarray(item.computed, dtype=np.float64)
    errors = parity_residuals(item)
    vertical = errors if residuals else np.asarray(item.predicted, dtype=np.float64)
    color = "C0"

    groups = _split_by_occupancy(item, computed, vertical)
    split = len(groups) > 1
    for x, y, fill in groups:
        axes.plot(
            x,
            y,
            linestyle="none",
            marker="o",
            markersize=_marker_size(computed.size),
            markeredgewidth=1.0,
            color=color,
            markerfacecolor=color if fill else "none",
            alpha=0.75 if computed.size > _CROWDED else 1.0,
        )

    if residuals:
        axes.axhline(0.0, color="0.4", linewidth=0.8, zorder=0)
        # Symmetric about zero: a residual's sign carries no more weight
        # than its magnitude, so the rule at zero sits in the middle.
        limit = float(np.max(np.abs(errors))) if errors.size else 1.0
        span = limit + (_FRAME_MARGIN * limit or 1.0)
        axes.set_ylim(-span, span)
    elif computed.size:
        # One range on both axes, in a square frame: only then does the
        # identity line run at 45 degrees, which is what a reader judges
        # the points against.
        reach = np.concatenate([computed, vertical])
        low, high = float(reach.min()), float(reach.max())
        margin = _FRAME_MARGIN * (high - low) or 1.0
        frame = (low - margin, high + margin)
        axes.plot(frame, frame, color="0.4", linewidth=0.8, zorder=0)
        axes.set_xlim(*frame)
        axes.set_ylim(*frame)
        axes.set_box_aspect(1)

    axes.set_xlabel(parity_axis_label(quantity, "true"))
    axes.set_ylabel(parity_axis_label(quantity, "residual" if residuals else "pred"))

    loc = _corner_loc(computed, vertical, axes.get_xlim(), axes.get_ylim())
    _annotate_metrics(axes, item, loc)

    if marginal is not None:
        edges = np.histogram_bin_edges(errors, bins="auto") if errors.size else [0.0, 1.0]
        _draw_marginal(marginal, errors, color, list(edges))

    return split


def _parity_panels(figure: Any, panels: int, residuals: bool) -> list[tuple[Any, Any]]:
    """Return one drawing axes per panel, each with its marginal axes or ``None``.

    A residual panel keeps a narrow strip on its right, sharing its y axis,
    for the histogram of the residuals it drew. The strip exists only in
    that mode, so a plain parity panel is not left with an empty gap where
    a histogram would otherwise go.
    """
    grid = figure.add_gridspec(1, panels, wspace=0.35)
    made: list[tuple[Any, Any]] = []
    for column in range(panels):
        if not residuals:
            made.append((figure.add_subplot(grid[0, column]), None))
            continue
        split = grid[0, column].subgridspec(1, 2, width_ratios=(4, 1), wspace=0.05)
        axes = figure.add_subplot(split[0, 0])
        made.append((axes, figure.add_subplot(split[0, 1], sharey=axes)))
    return made


def render_parity(
    panels: Sequence[ParitySeries],
    output_path: Path | None = None,
    show: bool = False,
    residuals: bool = False,
    legend: bool | None = None,
) -> None:
    """Draw one run's panels, one per quantity, side by side, and write or show the figure.

    One occupancy key serves every panel, drawn above them rather than
    inside any one panel's own data area.

    :param panels: the run's series, one per quantity, in the order the
        panels are drawn.
    :param output_path: where to write the figure; the extension sets the
        format. ``None`` writes nothing.
    :param show: open an interactive window.
    :param residuals: draw predicted minus computed, with a histogram of
        the residuals beside each panel.
    :param legend: draw the occupancy key, or leave it out. ``None`` draws
        it only when some panel's orbitals split by occupancy.
    """
    import matplotlib

    if not show:
        # Chosen before pyplot is imported: a run that only writes a file must
        # not depend on a display, so that it works over ssh and in CI.
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = 4.5 * len(panels)
    # Constrained rather than tight: a square parity frame and a marginal
    # sharing its neighbour's y axis are both beyond tight_layout, which
    # says so and lays the figure out wrong.
    figure = plt.figure(figsize=(width, 4.5), layout="constrained")
    made = _parity_panels(figure, len(panels), residuals)
    split = False
    for (axes, marginal), item in zip(made, panels, strict=True):
        split = draw_parity(axes, item, residuals=residuals, marginal=marginal) or split

    wanted = split if legend is None else legend
    if wanted:
        # "outside upper center" reserves its own row above the panels
        # under constrained layout, rather than overlapping their data.
        figure.legend(
            handles=_occupancy_legend_handles(),
            loc="outside upper center",
            ncol=2,
            frameon=False,
            fontsize="small",
        )

    if output_path is not None:
        figure.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(figure)


def render_spectra(
    series: Sequence[SpectrumSeries],
    output_path: Path | None = None,
    show: bool = False,
    real: bool = False,
    ip: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
) -> None:
    """Draw the spectra and write or show the figure.

    :param series: the spectra to draw.
    :param output_path: where to write the figure; the extension sets the
        format. ``None`` writes nothing.
    :param show: open an interactive window.
    :param real: draw Re ε instead of Im ε.
    :param ip: overlay the independent-particle spectrum.
    :param xlim: the energy range to show, in eV. ``None`` is tight to the
        energies drawn.
    :param ylim: the range to show. ``None`` starts Im ε at 0 and leaves
        Re ε automatic.
    :param legend: draw the key, or leave it out. ``None`` always draws it.
    """
    import matplotlib

    if not show:
        # Chosen before pyplot is imported: a run that only writes a file must
        # not depend on a display, so that it works over ssh and in CI.
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6.0, 4.5))
    draw_spectra(axes, series, real=real, ip=ip, xlim=xlim, ylim=ylim, legend=legend)
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(figure)


def render_band_structures(
    series: Sequence[BandSeries],
    output_path: Path | None = None,
    show: bool = False,
    zero: EnergyZero = EnergyZero.NONE,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
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
    """
    import matplotlib

    if not show:
        # Chosen before pyplot is imported: a run that only writes a file must
        # not depend on a display, so that it works over ssh and in CI.
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6.0, 4.5))
    draw_band_structures(axes, series, zero=zero, ylim=ylim, legend=legend)
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(figure)
