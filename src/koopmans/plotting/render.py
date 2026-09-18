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
    SpectrumSeries,
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
    "draw_spectra",
    "path_distances",
    "render_band_structures",
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


#: How close two arrows' conduction-band-minimum positions may sit — as a
#: fraction of the path's drawn length — before they count as landing at the
#: same k-point rather than two nearby ones.
_GAP_COLLISION_TOL_FRACTION = 1e-6

#: How far apart two coinciding arrows are nudged, as a fraction of the
#: path's drawn length, spread symmetrically about the shared k-point.
_GAP_NUDGE_FRACTION = 0.03


def _draw_gap(
    axes: Axes,
    item: BandSeries,
    edge: BandGap,
    distances: np.ndarray,
    color: Any,
    arrow_x: float,
    outward: bool = False,
) -> None:
    """Draw one series' band gap in the conventional textbook form.

    A vertical double-headed arrow, at ``arrow_x``, runs from the valence
    band maximum's energy to the conduction band minimum's; for an indirect
    gap a dashed rule at the valence level marks where it sits, reaching
    from its own k-point to the arrow. A direct gap needs no such rule,
    since the arrow's own foot already sits at the valence band maximum's
    k-point. The label reads the gap's value beside the arrow, to the right
    unless ``outward``, and never joins the legend.

    :param arrow_x: the arrow's x position, nudged away from ``edge``'s own
        conduction-band-minimum position when another series' arrow lands
        at the same k-point.
    :param outward: offset the label, and the dashed rule's reach, to the
        left of the arrow instead of the right.
    """
    vbm_x = float(distances[edge.vbm_kpoint_index])
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

    axes.annotate(
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
        annotation_clip=False,
    )
    offset, alignment = ((-8, 0), "right") if outward else ((8, 0), "left")
    axes.annotate(
        f"{edge.value:.2f} {item.units}",
        xy=(arrow_x, (vbm_y + cbm_y) / 2),
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


def _spread_arrow_positions(
    positions: Sequence[float], path_length: float
) -> tuple[list[float], list[bool]]:
    """Return each arrow's x position, nudged apart within its collision cluster.

    Positions within ``_GAP_COLLISION_TOL_FRACTION`` of the path length of
    each other land at the same k-point; each such cluster is spread
    symmetrically about it by ``_GAP_NUDGE_FRACTION``, and the accompanying
    flags say which member reads its label from the arrow's outer (left)
    side — the ones nudged left — so that neither the arrows nor their
    labels overlap.
    """
    tolerance = _GAP_COLLISION_TOL_FRACTION * path_length
    nudged = list(positions)
    outward = [False] * len(positions)
    placed = [False] * len(positions)
    for index, position in enumerate(positions):
        if placed[index]:
            continue
        cluster = [
            other
            for other, candidate in enumerate(positions)
            if not placed[other] and abs(candidate - position) <= tolerance
        ]
        if len(cluster) > 1:
            spread = _GAP_NUDGE_FRACTION * path_length
            offsets = np.linspace(-spread / 2, spread / 2, len(cluster))
            for member, offset in zip(cluster, offsets, strict=True):
                nudged[member] = position + float(offset)
                outward[member] = offset < 0
                placed[member] = True
        else:
            placed[index] = True
    return nudged, outward


def _draw_gaps(
    axes: Axes,
    candidates: Sequence[tuple[BandSeries, BandGap, np.ndarray, Any]],
    path_length: float,
) -> None:
    """Draw every series' gap annotation, nudging apart ones sharing a k-point."""
    if not candidates:
        return
    positions = [float(distances[edge.cbm_kpoint_index]) for _, edge, distances, _ in candidates]
    arrow_positions, outward_flags = _spread_arrow_positions(positions, path_length)
    for (item, edge, distances, color), arrow_x, outward in zip(
        candidates, arrow_positions, outward_flags, strict=True
    ):
        _draw_gap(axes, item, edge, distances, color, arrow_x, outward)


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
    skipped silently if it reports no valence band edge — nudged apart from
    another series' gap arrow landing at the same conduction-band-minimum
    k-point.

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
    path_length = limits[1] - limits[0] if limits is not None else 0.0
    _draw_gaps(axes, gap_candidates, path_length)
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
