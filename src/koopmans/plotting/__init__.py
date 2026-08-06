"""Plot the results of a finished koopmans run.

Two halves that meet at :class:`~koopmans.plotting.series.BandSeries`: a
resolver that turns run folders into series records, and a renderer that draws
records onto axes. Nothing but the resolver touches AiiDA, so a future
self-contained dump is a change of source rather than a rewrite.
"""

from __future__ import annotations

from koopmans.plotting.render import (
    DIVIDER_LABEL,
    draw_band_structures,
    path_distances,
    render_band_structures,
)
from koopmans.plotting.resolve import (
    BAND_PRODUCERS,
    BandProducer,
    PlottingError,
    resolve_band_series,
    run_node,
)
from koopmans.plotting.series import (
    BandSeries,
    EnergyZero,
    NoEnergyZeroError,
    PathMismatchError,
    apply_energy_zero,
    apply_labels,
    check_paths_agree,
    describe_energy_zero,
    write_series_json,
)

__all__ = [
    "BAND_PRODUCERS",
    "DIVIDER_LABEL",
    "BandProducer",
    "BandSeries",
    "EnergyZero",
    "NoEnergyZeroError",
    "PathMismatchError",
    "PlottingError",
    "apply_energy_zero",
    "apply_labels",
    "check_paths_agree",
    "describe_energy_zero",
    "draw_band_structures",
    "path_distances",
    "render_band_structures",
    "resolve_band_series",
    "run_node",
    "write_series_json",
]
