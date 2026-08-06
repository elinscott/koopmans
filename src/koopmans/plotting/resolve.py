"""Turn the folders a run wrote into the series records a figure is drawn from.

The folder names the run; the AiiDA database holds it. Each dumped root
carries an ``aiida_node_metadata.yaml`` recording the uuid of the process it
came from, which is the only handle this module needs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from koopmans.aiida.dumping import NODE_METADATA_FILE
from koopmans.plotting.series import BandSeries

if TYPE_CHECKING:
    from aiida import orm

__all__ = [
    "BAND_PRODUCERS",
    "BandProducer",
    "PlottingError",
    "resolve_band_series",
    "run_node",
]


class PlottingError(Exception):
    """A folder cannot be turned into a figure."""


References = Callable[["orm.ProcessNode", "orm.BandsData"], "tuple[float | None, float | None]"]


def _vbm_from_occupations(bands: orm.BandsData) -> float | None:
    """Return the highest energy carrying more than half the peak occupation.

    Quantum ESPRESSO's occupations are normalized to one or to two depending on
    the spin treatment, so the threshold is relative. For a metal the states it
    selects reach up to the Fermi level.
    """
    try:
        energies, occupations = bands.get_bands(also_occupations=True)  # type: ignore[no-untyped-call]
    except (KeyError, AttributeError, ValueError):
        return None
    occupied = np.asarray(occupations, dtype=np.float64)
    if occupied.size == 0 or occupied.max() <= 0.0:
        return None
    mask = occupied > 0.5 * occupied.max()
    if not mask.any():
        return None
    return float(np.asarray(energies, dtype=np.float64)[mask].max())


def _output_dict(node: orm.ProcessNode, socket: str) -> dict[str, Any]:
    """Return a ``Dict`` output of ``node`` as a plain dict, empty if absent."""
    output = getattr(node.outputs, socket, None)
    return output.get_dict() if output is not None else {}


def _pw_bands_references(
    node: orm.ProcessNode, bands: orm.BandsData
) -> tuple[float | None, float | None]:
    """Return the valence band edge and Fermi level of a pw.x bands run."""
    fermi = _output_dict(node, "band_parameters").get("fermi_energy")
    return _vbm_from_occupations(bands), fermi


def _kcw_ham_references(
    node: orm.ProcessNode, bands: orm.BandsData
) -> tuple[float | None, float | None]:
    """Return the Koopmans valence band edge reported by a kcw.x ham run."""
    return _output_dict(node, "output_parameters").get("ki_homo_energy"), None


def _no_references(
    node: orm.ProcessNode, bands: orm.BandsData
) -> tuple[float | None, float | None]:
    """Return no reference energies: an interpolation reports neither."""
    return None, None


@dataclass(frozen=True)
class BandProducer:
    """A step whose output socket holds a band structure along a k-path.

    Membership is declared, not inferred. An scf mesh, the explicit
    eigenvalues split-mode wannierization detects its band groups on, and an
    interpolated path are all the same shape of data; only the step that
    produced a socket says which it is.
    """

    process_type: str
    socket: str
    series: str
    references: References


BAND_PRODUCERS: tuple[BandProducer, ...] = (
    BandProducer(
        process_type="aiida.workflows:quantumespresso.pw.bands",
        socket="band_structure",
        series="DFT",
        references=_pw_bands_references,
    ),
    BandProducer(
        process_type="aiida.calculations:koopmans.kcw_ham",
        socket="bands",
        series="KI",
        references=_kcw_ham_references,
    ),
    BandProducer(
        process_type="aiida.workflows:wannier90_workflows.base.wannier90",
        socket="interpolated_bands",
        series="Wannier interpolation",
        references=_no_references,
    ),
)

_PRODUCERS_BY_TYPE = {producer.process_type: producer for producer in BAND_PRODUCERS}

#: Why a route can finish and still have no band structure to draw, keyed by
#: the name of the workgraph the run built.
_EMPTY_REASONS = {
    "KoopmansDSCFWorkflow": (
        "the ΔSCF route computes on a supercell, and recovering primitive-cell "
        "bands from it needs unfold-and-interpolate, which no route calls"
    ),
    "TrajectoryWorkflow": (
        "the ΔSCF route computes on a supercell, and recovering primitive-cell "
        "bands from it needs unfold-and-interpolate, which no route calls"
    ),
    "SinglepointDFPTWorkflow": (
        "kcw.x interpolates a band structure only when it is given a k-point "
        "path; add `kpoints: {path: ...}` to the input file and rerun"
    ),
    "Wannierize": (
        "no route hands a k-point path to wannier90, so the Wannier "
        "interpolation is never computed (koopmans issue #80)"
    ),
    "WannierizeBlocks": (
        "no route hands a k-point path to wannier90, so the Wannier "
        "interpolation is never computed (koopmans issue #80)"
    ),
    "DielectricTask": "the dielectric route computes no band structure",
}


def _read_uuid(folder: Path) -> str:
    """Return the uuid of the process the run in ``folder`` was dumped from."""
    import yaml

    metadata_path = folder / NODE_METADATA_FILE
    if not metadata_path.is_file():
        raise PlottingError(
            f"{folder} is not a koopmans run directory: it holds no "
            f"{NODE_METADATA_FILE}. Pass the folder `koopmans run` wrote, which "
            "is named after the input file it was given."
        )
    try:
        parsed = yaml.safe_load(metadata_path.read_text())
        uuid = parsed["Node data"]["uuid"]
    except (OSError, UnicodeDecodeError, yaml.YAMLError, TypeError, KeyError):
        raise PlottingError(
            f"{metadata_path} does not record which run it came from. Rerun "
            "`koopmans run` to write the folder again."
        ) from None
    return str(uuid)


def run_node(folder: Path) -> orm.ProcessNode:
    """Return the process node the run in ``folder`` was dumped from.

    Expects an AiiDA profile to be loaded.
    """
    from aiida import orm
    from aiida.common.exceptions import NotExistent

    uuid = _read_uuid(folder)
    try:
        node = orm.load_node(uuid)
    except NotExistent:
        raise PlottingError(
            f"The run in {folder} (uuid {uuid}) is not in this AiiDA profile: it was "
            "made under a different profile, or on a different machine. A dumped "
            "folder does not yet carry its own results, so it can only be plotted "
            "where it was run."
        ) from None
    if not isinstance(node, orm.ProcessNode):
        raise PlottingError(
            f"The uuid recorded in {folder} names a {type(node).__name__}, not a run."
        )
    return node


def _route_name(node: orm.ProcessNode) -> str:
    """Return the name of the workgraph a run built."""
    if node.label:
        return str(node.label)
    label = str(node.process_label or "")
    if label.startswith("WorkGraph<") and label.endswith(">"):
        return label[len("WorkGraph<") : -1]
    return label


def _step_name(node: orm.ProcessNode) -> str:
    """Return the name the run's caller gave this step."""
    from aiida.common.links import LinkType

    incoming = node.base.links.get_incoming(
        link_type=(LinkType.CALL_WORK, LinkType.CALL_CALC)
    ).all()
    return str(incoming[0].link_label) if incoming else str(node.process_label)


def _failure_warning(folder: Path, node: orm.ProcessNode) -> str | None:
    """Return a warning naming the step that failed, if the run did not finish."""
    if node.is_finished_ok:
        return None

    from aiida import orm

    failed = sorted(
        (child for child in node.called_descendants if child.is_failed),
        key=lambda child: (child.ctime, child.pk),
    )
    # The calculation that failed, rather than the chain of workflows that
    # reported the failure upwards; the last one to start, when several did.
    calculations = [child for child in failed if isinstance(child, orm.CalcJobNode)]
    culprit = (calculations or failed or [node])[-1]
    detail = culprit.exit_message or f"exit status {culprit.exit_status}"
    return (
        f"{folder}: the run did not finish — {_step_name(culprit)} failed ({detail}). "
        "Plotting what is there."
    )


def _cell_of(bands: orm.BandsData) -> list[list[float]] | None:
    """Return the band structure's cell, or ``None`` if it carries none."""
    try:
        return [list(map(float, row)) for row in bands.cell]
    except AttributeError:
        return None


def _series_from_bands(
    node: orm.ProcessNode, producer: BandProducer, bands: orm.BandsData, name: str
) -> list[BandSeries]:
    """Return the series a single ``BandsData`` output contributes.

    A collinear calculation stores one table per spin channel and becomes one
    series per channel.
    """
    vbm, fermi = producer.references(node, bands)
    energies = np.asarray(bands.get_bands(), dtype=np.float64)  # type: ignore[no-untyped-call]
    channels: list[tuple[str, np.ndarray[Any, Any]]] = (
        [(f"{name} ({label})", energies[index]) for index, label in enumerate(("up", "down"))]
        if energies.ndim == 3
        else [(name, energies)]
    )
    return [
        BandSeries(
            label=label,
            kpoints=[list(map(float, kpoint)) for kpoint in bands.get_kpoints()],  # type: ignore[no-untyped-call]
            energies=table.tolist(),
            cell=_cell_of(bands),
            path_labels=[(int(index), str(text)) for index, text in (bands.labels or [])],
            units=str(getattr(bands, "units", None) or "eV"),
            vbm=vbm,
            fermi=fermi,
        )
        for label, table in channels
    ]


def _series_from_node(node: orm.ProcessNode) -> list[BandSeries]:
    """Return every declared band structure a run produced, in run order."""
    steps = sorted(
        (
            step
            for step in [node, *node.called_descendants]
            if step.process_type in _PRODUCERS_BY_TYPE
        ),
        key=lambda step: (step.ctime, step.pk),
    )

    matches = []
    for step in steps:
        producer = _PRODUCERS_BY_TYPE[str(step.process_type)]
        bands = getattr(step.outputs, producer.socket, None)
        if bands is not None:
            matches.append((step, producer, bands))

    # One step per producer needs no disambiguation; several — a per-spin or
    # per-block fan-out — are told apart by the name each step runs under.
    counts: dict[str, int] = {}
    for _, producer, _ in matches:
        counts[producer.series] = counts.get(producer.series, 0) + 1

    series: list[BandSeries] = []
    for step, producer, bands in matches:
        name = producer.series
        if counts[name] > 1:
            name = f"{name} ({_step_name(step)})"
        series += _series_from_bands(step, producer, bands, name)
    return series


def _nothing_plottable(folders: Sequence[Path], nodes: Sequence[orm.ProcessNode]) -> PlottingError:
    """Return the error naming each route that ran and why it drew a blank."""
    lines = [f"No band structure to plot in {', '.join(str(folder) for folder in folders)}."]
    for folder, node in zip(folders, nodes, strict=True):
        route = _route_name(node)
        reason = _EMPTY_REASONS.get(
            route,
            "no step of it produced one of the outputs koopmans plots ("
            + ", ".join(f"{p.series}: {p.socket}" for p in BAND_PRODUCERS)
            + ")",
        )
        lines.append(f"  {folder} ran {route}, and {reason}.")
    return PlottingError("\n".join(lines))


def resolve_band_series(folders: Sequence[Path]) -> tuple[list[BandSeries], list[str]]:
    """Return the band structures of the given runs, and any warnings.

    Series are labelled by the step that produced them, prefixed by the folder
    name when more than one folder is on the axes.

    :raises PlottingError: if a folder is not a run directory, its run is not
        in this profile, or nothing plottable was found in any of them.
    """
    nodes = [run_node(folder) for folder in folders]

    series: list[BandSeries] = []
    warnings: list[str] = []
    for folder, node in zip(folders, nodes, strict=True):
        warning = _failure_warning(folder, node)
        if warning is not None:
            warnings.append(warning)
        found = _series_from_node(node)
        if len(folders) > 1:
            prefix = folder.name or folder.resolve().name
            for item in found:
                item.label = f"{prefix}: {item.label}"
        series += found

    if not series:
        raise _nothing_plottable(folders, nodes)
    return series, warnings
