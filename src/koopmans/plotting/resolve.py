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
    """Return the valence band edge and Fermi level of a pw.x bands run.

    The bands step reports a Fermi level only when it inherited one; the scf
    that preceded it always does.
    """
    fermi = _output_dict(node, "band_parameters").get("fermi_energy")
    if fermi is None:
        fermi = _output_dict(node, "scf_parameters").get("fermi_energy")
    return _vbm_from_occupations(bands), fermi


def _pw_base_bands_references(
    node: orm.ProcessNode, bands: orm.BandsData
) -> tuple[float | None, float | None]:
    """Return the band edge and Fermi level a bare pw.x bands run reports.

    A bands calculation recomputes no occupations; the Fermi level it
    reports in ``output_parameters`` is the one read back from the parent
    density's restart.
    """
    fermi = _output_dict(node, "output_parameters").get("fermi_energy")
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


def _is_path_bands_run(node: orm.ProcessNode) -> bool:
    """Whether a pw.x base run declared ``calculation = 'bands'`` in its inputs.

    An scf or nscf run publishes the same ``output_band`` socket for its
    mesh eigenvalues; only the declared calculation type says a run sampled
    a path.
    """
    try:
        parameters = node.inputs.pw.parameters.get_dict()
    except AttributeError:
        return False
    return bool(parameters.get("CONTROL", {}).get("calculation") == "bands")


@dataclass(frozen=True)
class BandProducer:
    """A step whose output socket holds a band structure along a k-path.

    Membership is declared, not inferred. An scf mesh, the explicit
    eigenvalues split-mode wannierization detects its band groups on, and an
    interpolated path are all the same shape of data; only the step that
    produced a socket says which it is. Where one process type covers
    several of those shapes, ``applies`` narrows membership by what the run
    itself declared in its inputs.

    ``socket`` is a dotted output path, so a workflow that publishes its result
    under a namespace can name it.
    """

    process_type: str
    socket: str
    series: str
    references: References
    applies: Callable[[orm.ProcessNode], bool] | None = None


#: The optimize workchain's own wannierization outputs, and the series each
#: names. Its scan reruns the plain wannierization once per trial frozen
#: window, so only these namespaces say which one it chose; at most one of the
#: ``optimal``/``plot`` pair carries interpolated bands, because
#: ``separate_plotting`` moves the plotting keywords out of the optimal run.
_OPTIMIZE_OUTPUTS = (
    ("wannier90_optimal", "Wannier interpolation"),
    ("wannier90_plot", "Wannier interpolation"),
    ("wannier90_optimal_up", "Wannier interpolation (up)"),
    ("wannier90_plot_up", "Wannier interpolation (up)"),
    ("wannier90_optimal_down", "Wannier interpolation (down)"),
    ("wannier90_plot_down", "Wannier interpolation (down)"),
)

BAND_PRODUCERS: tuple[BandProducer, ...] = (
    BandProducer(
        process_type="aiida.workflows:quantumespresso.pw.bands",
        socket="band_structure",
        series="DFT",
        references=_pw_bands_references,
    ),
    # The bare bands run a wannierize route runs off its scf density: its
    # explicit eigenvalues along the path are what the Wannier
    # interpolation is judged against.
    BandProducer(
        process_type="aiida.workflows:quantumespresso.pw.base",
        socket="output_band",
        series="DFT",
        references=_pw_base_bands_references,
        applies=_is_path_bands_run,
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
    *(
        BandProducer(
            process_type="aiida.workflows:wannier90_workflows.optimize",
            socket=f"{namespace}.interpolated_bands",
            series=name,
            references=_no_references,
        )
        for namespace, name in _OPTIMIZE_OUTPUTS
    ),
)


def _producers_for(step: orm.ProcessNode) -> list[BandProducer]:
    """Return the producers whose declared membership ``step`` satisfies."""
    return [
        producer
        for producer in BAND_PRODUCERS
        if producer.process_type == step.process_type
        and (producer.applies is None or producer.applies(step))
    ]


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
        "wannier90 interpolates a band structure only when it is given a "
        "k-point path; add `kpoints: {path: ...}` to the input file and rerun"
    ),
    "WannierizeBlocks": (
        "wannier90 interpolates a band structure only when it is given a "
        "k-point path; add `kpoints: {path: ...}` to the input file and rerun"
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


def _incoming_call(node: orm.ProcessNode) -> tuple[str, orm.ProcessNode] | None:
    """Return the link label and caller of the CALL link that produced ``node``."""
    from aiida import orm
    from aiida.common.links import LinkType

    incoming = node.base.links.get_incoming(
        link_type=(LinkType.CALL_WORK, LinkType.CALL_CALC)
    ).all()
    if not incoming:
        return None
    link = incoming[0]
    if not isinstance(link.node, orm.ProcessNode):
        raise TypeError(f"{node} has a CALL link from {link.node}, which is not a process.")
    return str(link.link_label), link.node


def _step_name(node: orm.ProcessNode) -> str:
    """Return the name the run's caller gave this step."""
    found = _incoming_call(node)
    return found[0] if found else str(node.process_label)


def _call_chain(step: orm.ProcessNode, root: orm.ProcessNode) -> list[str]:
    """Return the CALL link labels from ``step`` up to ``root``, nearest first."""
    chain: list[str] = []
    node = step
    while node.pk != root.pk:
        found = _incoming_call(node)
        if found is None:
            break
        label, node = found
        chain.append(label)
    return chain


def _shared_prefix(labels: Sequence[str]) -> str:
    """Return the longest string every one of ``labels`` starts with."""
    shortest = min(labels, key=len)
    for length in range(len(shortest), 0, -1):
        prefix = shortest[:length]
        if all(label.startswith(prefix) for label in labels):
            return prefix
    return ""


def _prettify_link_label(label: str) -> str:
    """Render a call link label for a legend: underscores become spaces."""
    return label.replace("_", " ")


def _disambiguating_labels(steps: Sequence[orm.ProcessNode], root: orm.ProcessNode) -> list[str]:
    """Return one distinguishing legend suffix per step in a tied group.

    Walks each step's chain of CALL links toward ``root`` and stops at the
    first depth where every step's label there is distinct from the rest.
    The immediate call label, when it already distinguishes the group, is
    used as-is: the run's caller named that step directly. Escalating past
    it means every branch shares that immediate label (a sub-graph that
    calls the same producer under the same name once per block), so the
    prefix the group shares at the depth that does distinguish them (e.g.
    the ``wannierize_`` common to a set of per-block sub-graph calls) is
    stripped before the label is rendered, leaving the block token. Falls
    back to plain numbering when no depth distinguishes every step.
    """
    chains = [_call_chain(step, root) for step in steps]
    depth = 0
    while all(depth < len(chain) for chain in chains):
        labels = [chain[depth] for chain in chains]
        if len(set(labels)) == len(labels):
            if depth == 0:
                return list(labels)
            prefix = _shared_prefix(labels)
            if "_" in prefix:
                prefix = prefix.rsplit("_", 1)[0] + "_"
            else:
                prefix = ""
            return [_prettify_link_label(label[len(prefix) :] or label) for label in labels]
        depth += 1
    return [str(index) for index in range(1, len(steps) + 1)]


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


def _declared_pw_system(node: orm.ProcessNode) -> dict[str, Any]:
    """Return the pw.x ``&SYSTEM`` namelist ``node`` declared, if any."""
    try:
        return dict(node.inputs.pw.parameters.get_dict().get("SYSTEM", {}))
    except AttributeError:
        return {}


def _spin_channels_are_degenerate(node: orm.ProcessNode) -> bool:
    """Whether ``node`` declared nspin=2 with no nonzero starting magnetization.

    A run that inherits its density from a magnetic-capable restart can carry
    two identical channels without itself declaring any magnetic moment; the
    channels are not compared numerically, so declared inputs are the only
    way to tell a real spin split from an inherited one. A magnetization that
    cannot be determined from the declared inputs keeps both channels.
    """
    system = _declared_pw_system(node)
    if system.get("nspin") != 2:
        return False
    magnetization = system.get("starting_magnetization", {})
    if not isinstance(magnetization, dict):
        return False
    return all(float(value) == 0.0 for value in magnetization.values())


def _series_from_bands(
    node: orm.ProcessNode, producer: BandProducer, bands: orm.BandsData, name: str
) -> list[BandSeries]:
    """Return the series a single ``BandsData`` output contributes.

    A collinear calculation stores one table per spin channel and becomes one
    series per channel, unless the declared inputs show the channels are
    degenerate by construction, in which case one series carries channel 0.
    """
    vbm, fermi = producer.references(node, bands)
    energies = np.asarray(bands.get_bands(), dtype=np.float64)  # type: ignore[no-untyped-call]
    if energies.ndim == 3 and _spin_channels_are_degenerate(node):
        channels: list[tuple[str, np.ndarray[Any, Any]]] = [(name, energies[0])]
    elif energies.ndim == 3:
        channels = [
            (f"{name} ({label})", energies[index]) for index, label in enumerate(("up", "down"))
        ]
    else:
        channels = [(name, energies)]
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


def _output_at(node: orm.ProcessNode, socket: str) -> orm.BandsData | None:
    """Return the band structure ``node`` publishes at a dotted socket path.

    ``None`` when the socket is absent, or holds something that is not a band
    structure.
    """
    from aiida import orm

    found: Any = node.outputs
    for part in socket.split("."):
        found = getattr(found, part, None)
        if found is None:
            return None
    return found if isinstance(found, orm.BandsData) else None


def _producing_steps(root: orm.ProcessNode) -> list[orm.ProcessNode]:
    """Return the run's declared band-structure steps, in run order.

    A step that declares a band structure owns everything below it, so the
    walk stops there: the optimize workchain reruns the plain wannierization
    once per trial frozen window, and only its own outputs name the one it
    chose.
    """
    found: list[orm.ProcessNode] = []
    frontier = [root]
    while frontier:
        step = frontier.pop()
        if _producers_for(step):
            found.append(step)
        else:
            frontier += step.called
    return sorted(found, key=lambda step: (step.ctime, step.pk))


def _series_from_node(node: orm.ProcessNode) -> list[BandSeries]:
    """Return every declared band structure a run produced, in run order."""
    matches = []
    for step in _producing_steps(node):
        for producer in _producers_for(step):
            bands = _output_at(step, producer.socket)
            if bands is not None:
                matches.append((step, producer, bands))

    # One step per series name needs no disambiguation; several — a per-spin
    # or per-block fan-out — are told apart by the call chain that led there.
    grouped: dict[str, list[int]] = {}
    for index, (_, producer, _) in enumerate(matches):
        grouped.setdefault(producer.series, []).append(index)

    names = [producer.series for _, producer, _ in matches]
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        suffixes = _disambiguating_labels([matches[i][0] for i in indices], node)
        for index, suffix in zip(indices, suffixes, strict=True):
            names[index] = f"{names[index]} ({suffix})"

    series: list[BandSeries] = []
    for (step, producer, bands), name in zip(matches, names, strict=True):
        series += _series_from_bands(step, producer, bands, name)
    return series


def _plotted_sockets() -> str:
    """Return the declared output names, once each, in table order."""
    names = dict.fromkeys(producer.socket.rsplit(".", 1)[-1] for producer in BAND_PRODUCERS)
    return ", ".join(names)


def _nothing_plottable(folders: Sequence[Path], nodes: Sequence[orm.ProcessNode]) -> PlottingError:
    """Return the error naming each route that ran and why it drew a blank."""
    lines = [f"No band structure to plot in {', '.join(str(folder) for folder in folders)}."]
    for folder, node in zip(folders, nodes, strict=True):
        route = _route_name(node)
        reason = _EMPTY_REASONS.get(
            route,
            f"no step of it produced one of the outputs koopmans plots ({_plotted_sockets()})",
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
