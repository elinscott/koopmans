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
    "RunNotInProfileError",
    "resolve_band_series",
    "run_node",
]


class PlottingError(Exception):
    """A folder cannot be turned into a figure."""


class RunNotInProfileError(PlottingError):
    """A folder names a run this AiiDA profile does not hold."""


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


def _unfolded_band_references(
    node: orm.ProcessNode, bands: orm.BandsData
) -> tuple[float | None, float | None]:
    """Return the valence band edge the unfold-and-interpolate stage computed.

    A ΔSCF interpolation carries no occupations, so the edge cannot be read
    off the bands; it travels as an input of the step that built them.
    """
    reference = getattr(node.inputs, "reference", None)
    return (None if reference is None else float(reference.value)), None


def _declared_pw_parameters(node: orm.ProcessNode) -> dict[str, Any]:
    """Return the pw.x namelists ``node`` declared, if any.

    A base workchain carries them under its ``pw`` namespace and a bare
    calculation directly, so a dumped step reads the same as the run it
    belongs to.
    """
    for holder in (getattr(node.inputs, "pw", None), node.inputs):
        parameters = getattr(holder, "parameters", None)
        if parameters is not None:
            try:
                return dict(parameters.get_dict())
            except (AttributeError, TypeError):
                return {}
    return {}


def _is_path_bands_run(node: orm.ProcessNode) -> bool:
    """Whether a pw.x run declared ``calculation = 'bands'`` in its inputs.

    An scf or nscf run publishes the same ``output_band`` socket for its
    mesh eigenvalues; only the declared calculation type says a run sampled
    a path.
    """
    parameters = _declared_pw_parameters(node)
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
    # The pw.x calculation itself, so that a dumped bands step plots on its
    # own: a dump keeps a metadata file beside each calculation, and the
    # workchains above it leave none behind.
    BandProducer(
        process_type="aiida.calculations:quantumespresso.pw",
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
    # A dump writes a folder per calculation, so the calculation is the only
    # wannier90 step a folder can name; the base workchain above it owns the
    # walk whenever a whole run is plotted, so the two never both match.
    # aiida-wannier90 registers no reverse-resolvable entry point, so AiiDA
    # stores its class path rather than an ``aiida.calculations:`` name.
    BandProducer(
        process_type="aiida_wannier90.calculations.wannier90.Wannier90Calculation",
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
    # The split-mode auto-Wannierization merges each group's re-interpolated
    # bands into one block-wide structure; it is the same shape of curve as
    # the groups it concatenates, so it shares their series name and is told
    # apart from them by the call chain, same as any other tied producer.
    BandProducer(
        process_type="aiida_koopmans.workgraphs.auto_wannierize.merge_interpolated_bands",
        socket="result",
        series="Wannier interpolation",
        references=_no_references,
    ),
    # The ΔSCF route's unfold-and-interpolate stage: the step that attaches
    # the interpolated eigenvalues to their k-path is the one that names a
    # band structure, and it carries the valence band edge as an input.
    BandProducer(
        process_type="aiida_koopmans.workgraphs.ui.dscf.build_band_structure",
        socket="result",
        series="KI",
        references=_unfolded_band_references,
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


#: Why a ΔSCF singlepoint draws a blank: it computes on a supercell, so its
#: bands exist only once the unfold-and-interpolate stage has been asked for.
_SUPERCELL_REASON = (
    "the ΔSCF route computes on a supercell, so a primitive-cell band structure "
    "has to be asked for; add the path to interpolate along as "
    "`kpoints: {path: ...}` and rerun"
)

#: The same blank on the trajectory route, which runs no interpolation stage.
_TRAJECTORY_REASON = (
    "the trajectory route screens each snapshot on a supercell and reports "
    "screening parameters and eigenvalues, not band structures"
)

#: Why a wannierization draws a blank, whether the folder names the koopmans
#: workgraph or the calculation under it.
_NO_WANNIER_PATH_REASON = (
    "wannier90 writes an interpolated band structure only when the input "
    "provides a k-point path; add `kpoints: {path: ...}` to the input file "
    "and rerun"
)

#: Why a route can finish and still have no band structure to draw, keyed by
#: the name of the workgraph the run built, or the process label of the
#: calculation a folder inside it names.
_EMPTY_REASONS = {
    "KoopmansDSCFWorkflow": _SUPERCELL_REASON,
    "TrajectoryWorkflow": _TRAJECTORY_REASON,
    "SinglepointDFPTWorkflow": (
        "kcw.x interpolates a band structure only when it is given a k-point "
        "path; add `kpoints: {path: ...}` to the input file and rerun"
    ),
    "Wannierize": _NO_WANNIER_PATH_REASON,
    "WannierizeBlocks": _NO_WANNIER_PATH_REASON,
    "Wannier90Calculation": _NO_WANNIER_PATH_REASON,
    "DielectricTask": "the dielectric route computes no band structure",
}


#: How many plottable directories a rejected folder's error names. Long enough
#: for a per-block fan-out, short enough to read.
SUGGESTION_LIMIT = 10


def _read_uuid(folder: Path) -> str | None:
    """Return the uuid ``folder``'s metadata records, or ``None`` if it holds none.

    :raises PlottingError: if the file is there and records no uuid.
    """
    import yaml

    metadata_path = folder / NODE_METADATA_FILE
    if not metadata_path.is_file():
        return None
    try:
        parsed = yaml.safe_load(metadata_path.read_text())
        uuid = parsed["Node data"]["uuid"]
    except (OSError, UnicodeDecodeError, yaml.YAMLError, TypeError, KeyError):
        raise PlottingError(
            f"{folder} does not record which run it came from. Rerun `koopmans run` "
            "to write the folder again."
        ) from None
    return str(uuid)


def _has_band_structure(node: orm.ProcessNode) -> bool:
    """Whether the run under ``node`` published any band structure to plot."""
    return any(
        _output_at(step, producer.socket) is not None
        for step in _producing_steps(node)
        for producer in _producers_for(step)
    )


def _plottable_below(folder: Path) -> tuple[list[Path], int]:
    """Return what under ``folder`` can be plotted, and how much is elsewhere.

    The second value counts the directories skipped because this profile does
    not hold the runs they name.

    A dump keeps a metadata file beside each calculation and none beside the
    step folders grouping them, so a rejected folder can still say which of
    the directories under it can be given to the command. Shallowest first, so
    that a whole run outranks its own steps when the list is cut short.

    Expects an AiiDA profile to be loaded.
    """
    found: list[Path] = []
    absent = 0
    candidates = sorted(folder.rglob(NODE_METADATA_FILE), key=lambda path: (len(path.parts), path))
    for metadata_path in candidates:
        directory = metadata_path.parent
        if directory == folder:
            continue
        try:
            node = run_node(directory)
        except RunNotInProfileError:
            absent += 1
            continue
        except PlottingError:
            continue
        if _has_band_structure(node):
            found.append(directory)
    return found, absent


def _not_a_run_directory(folder: Path) -> PlottingError:
    """Return the error for a directory holding no metadata of its own.

    Names the directories beneath it that can be plotted, since a step folder
    grouping calculations is the one thing a reader is likely to have typed.
    """
    opening = f"{folder} is not a koopmans run directory"
    plottable, absent = _plottable_below(folder)
    if not plottable and absent:
        return PlottingError(
            f"{opening}, and the {absent} run(s) beneath it are not in this AiiDA "
            "profile: they were made under a different profile, or on a different "
            "machine. A dumped folder does not yet carry its own results, so it can "
            "only be plotted where it was run."
        )
    if not plottable:
        return PlottingError(
            f"{opening}, and nothing beneath it has a band structure to plot. Pass a "
            "directory `koopmans run` wrote, or a calculation directory inside one."
        )
    lines = [f"{opening}. These directories beneath it have band structures to plot:"]
    lines += [f"  {path}" for path in plottable[:SUGGESTION_LIMIT]]
    if len(plottable) > SUGGESTION_LIMIT:
        lines.append(f"  ... and {len(plottable) - SUGGESTION_LIMIT} more.")
    return PlottingError("\n".join(lines))


def run_node(folder: Path) -> orm.ProcessNode:
    """Return the process node the run in ``folder`` was dumped from.

    Expects an AiiDA profile to be loaded.

    :raises PlottingError: if ``folder`` records no run of its own, or the run
        it records is not in this profile.
    """
    from aiida import orm
    from aiida.common.exceptions import NotExistent

    uuid = _read_uuid(folder)
    if uuid is None:
        raise _not_a_run_directory(folder)
    try:
        node = orm.load_node(uuid)
    except NotExistent:
        raise RunNotInProfileError(
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
    if not isinstance(link.node, orm.ProcessNode):  # pragma: no cover
        # AiiDA's own link validation requires a CALL_WORK/CALL_CALC source to
        # already be a ProcessNode, so this never fires; it exists to narrow
        # the type rather than to guard against a reachable state.
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
        if found is None:  # pragma: no cover
            # ``_producing_steps`` only ever finds steps by walking down from
            # ``root`` via ``.called``, the same CALL links this walks back up
            # through, so the walk is guaranteed to reach ``root`` first.
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


def _stripped_hop_labels(labels: Sequence[str], depth: int) -> dict[str, str]:
    """Render one call-chain hop's sibling labels as legend text.

    ``labels`` are the distinct labels a single depth split a tied group
    into. At depth 0 they are the run's caller's own immediate names for
    each step, used exactly as given. Any deeper hop is a level of
    sub-graph wrapping the run itself chose the same way for every
    branch, so the lead-in every label there shares (e.g. the
    ``wannierize_`` common to a set of per-block sub-graph calls) is
    stripped before rendering, leaving the token that actually varies;
    labels with no shared lead-in are used exactly as they are.
    """
    if depth == 0:
        return dict(zip(labels, labels, strict=True))
    prefix = _shared_prefix(labels)
    if "_" in prefix:
        prefix = prefix.rsplit("_", 1)[0] + "_"
    else:
        prefix = ""
    return {label: _prettify_link_label(label[len(prefix) :] or label) for label in labels}


def _split_by_call_chain(
    steps: Sequence[orm.ProcessNode], root: orm.ProcessNode
) -> tuple[dict[int, list[str]], dict[int, int]]:
    """Partition ``steps`` by their call chain to ``root``.

    Returns, per step index: a history of the rendered hop labels that
    separated it from its call-chain siblings, in root-to-leaf order, for
    every step some depth eventually isolates; and a 1-based rank, scoped
    to the group of steps no depth ever isolates from each other (their
    chains run out while still tied), in call order.
    """
    chains = [_call_chain(step, root) for step in steps]
    histories: dict[int, list[str]] = {}
    unresolved: dict[int, int] = {}

    def resolve(indices: list[int], depth: int, path: list[str]) -> None:
        """Assign each of ``indices`` a history in ``histories`` or a rank in ``unresolved``."""
        if len(indices) == 1:
            histories[indices[0]] = path
            return
        if any(depth >= len(chains[i]) for i in indices):
            for rank, index in enumerate(sorted(indices), start=1):
                unresolved[index] = rank
            return
        buckets: dict[str, list[int]] = {}
        for i in indices:
            buckets.setdefault(chains[i][depth], []).append(i)
        if len(buckets) == 1:
            resolve(indices, depth + 1, path)
            return
        rendered = _stripped_hop_labels(list(buckets), depth)
        for label, members in buckets.items():
            resolve(members, depth + 1, [*path, rendered[label]])

    resolve(list(range(len(steps))), 0, [])
    return histories, unresolved


def _minimal_unique_suffixes(histories: dict[int, list[str]]) -> dict[int, str]:
    """Return the shortest hop-history suffix, per step, that tells it from the rest.

    Steps already unique on their own deepest hop keep just that one label
    (the common case: each block, group, or merge names itself distinctly).
    A hop shared by several steps' histories (e.g. several blocks each
    producing the same per-group fragment index) is not enough on its own,
    so those steps grow their rendered suffix one hop further toward the
    root, in lockstep, until every candidate in the run is distinct.
    """
    remaining = dict(histories)
    resolved: dict[int, str] = {}
    depth = 1
    while remaining:
        candidates = {index: ", ".join(hops[-depth:]) for index, hops in remaining.items()}
        counts: dict[str, int] = {}
        for text in candidates.values():
            counts[text] = counts.get(text, 0) + 1
        newly_resolved = [index for index, text in candidates.items() if counts[text] == 1]
        for index in newly_resolved:
            resolved[index] = candidates[index]
            del remaining[index]
        if not remaining:
            break
        if all(depth >= len(hops) for hops in remaining.values()):
            # Every remaining step has used its full history and some are
            # still tied by pure string coincidence between unrelated
            # branches; there is nothing more to combine.
            for index, hops in remaining.items():
                resolved[index] = ", ".join(hops)
            break
        depth += 1
    return resolved


def _disambiguating_labels(steps: Sequence[orm.ProcessNode], root: orm.ProcessNode) -> list[str]:
    """Return one distinguishing legend suffix per step in a tied group.

    Walks each step's chain of CALL links toward ``root``, splitting the
    group at each depth where a label there varies (a per-block sub-graph
    name, a per-group fragment index, a merge step's own literal call
    label, ...). A split-mode run nests several of these depths — block
    identity two levels above a per-group label that itself restarts at
    each block — so a step's legend suffix combines every depth that
    contributed to isolating it, not just one; the combination is grown
    only as far as needed; the common case (one producer, one distinguishing
    depth) still renders that depth's label alone. Falls back to plain
    numbering for a subset no depth distinguishes at all.
    """
    histories, unresolved = _split_by_call_chain(steps, root)
    suffixes = _minimal_unique_suffixes(histories) if histories else {}
    return [
        suffixes[index] if index in suffixes else str(unresolved[index])
        for index in range(len(steps))
    ]


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
    system = _declared_pw_parameters(node).get("SYSTEM", {})
    return dict(system) if isinstance(system, dict) else {}


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
    node: orm.ProcessNode,
    producer: BandProducer,
    bands: orm.BandsData,
    name: str,
    qualifier: str,
) -> list[tuple[BandSeries, str]]:
    """Return the series a single ``BandsData`` output contributes.

    A collinear calculation stores one table per spin channel and becomes one
    series per channel, unless the declared inputs show the channels are
    degenerate by construction, in which case one series carries channel 0.
    Each series is paired with the qualifier that tells it from its siblings
    within the run, which a name the caller chose keeps.
    """
    vbm, fermi = producer.references(node, bands)
    energies = np.asarray(bands.get_bands(), dtype=np.float64)  # type: ignore[no-untyped-call]
    if energies.ndim == 3 and _spin_channels_are_degenerate(node):
        channels: list[tuple[str, np.ndarray[Any, Any]]] = [(qualifier, energies[0])]
    elif energies.ndim == 3:
        channels = [
            (f"{qualifier} ({label})", energies[index])
            for index, label in enumerate(("up", "down"))
        ]
    else:
        channels = [(qualifier, energies)]
    return [
        (
            BandSeries(
                label=f"{name}{suffix}",
                kpoints=[list(map(float, kpoint)) for kpoint in bands.get_kpoints()],  # type: ignore[no-untyped-call]
                energies=table.tolist(),
                cell=_cell_of(bands),
                path_labels=[(int(index), str(text)) for index, text in (bands.labels or [])],
                units=str(getattr(bands, "units", None) or "eV"),
                vbm=vbm,
                fermi=fermi,
            ),
            suffix,
        )
        for suffix, table in channels
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


def _series_from_node(node: orm.ProcessNode) -> list[tuple[BandSeries, str]]:
    """Return every declared band structure a run produced, in run order.

    Each series is paired with its qualifier: the parenthesised text that tells
    it from the other series of the same run, empty when the run produced only
    one.
    """
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

    qualifiers = [""] * len(matches)
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        suffixes = _disambiguating_labels([matches[i][0] for i in indices], node)
        for index, suffix in zip(indices, suffixes, strict=True):
            qualifiers[index] = f" ({suffix})"

    series: list[tuple[BandSeries, str]] = []
    for (step, producer, bands), qualifier in zip(matches, qualifiers, strict=True):
        series += _series_from_bands(step, producer, bands, producer.series, qualifier)
    return series


def _plotted_sockets() -> str:
    """Return the declared output names, once each, in table order."""
    names = dict.fromkeys(producer.socket.rsplit(".", 1)[-1] for producer in BAND_PRODUCERS)
    return ", ".join(names)


def _nothing_plottable(empty: Sequence[tuple[Path, orm.ProcessNode]], total: int) -> PlottingError:
    """Return the error naming each route that ran and why it drew a blank.

    ``total`` is how many folders were asked for, so that the message can say
    whether any of the others carry a band structure.
    """
    lines = [f"No band structure to plot in {', '.join(str(folder) for folder, _ in empty)}."]
    for folder, node in empty:
        route = _route_name(node)
        reason = _EMPTY_REASONS.get(
            route,
            f"no step of it produced one of the outputs koopmans plots ({_plotted_sockets()})",
        )
        lines.append(f"  {folder} ran {route}, and {reason}.")
    if len(empty) < total:
        lines.append("Leave out the folders above to draw the rest.")
    return PlottingError("\n".join(lines))


def _name_after_folder(found: Sequence[tuple[BandSeries, str]], label: str) -> None:
    """Rename every series one folder contributed after that folder.

    Each keeps the qualifier telling it from its siblings. Curves told apart by
    their series names alone carry none, and their own derived names stand in,
    so that no two curves of one folder end up sharing a name.
    """
    for item, qualifier in found:
        if not qualifier and len(found) > 1:
            qualifier = f" ({item.label})"
        item.label = f"{label}{qualifier}"


def _check_one_per_folder(values: Sequence[str | None], folders: int, option: str) -> None:
    """Reject a per-folder option given for some but not all of the folders.

    :raises ValueError: if any values were given and they do not number ``folders``.
    """
    if values and len(values) != folders:
        raise ValueError(
            f"{len(values)} {option} value(s) were given for {folders} folder(s). "
            f"Give one {option} per folder, in the order the folders are listed, or "
            "none at all."
        )


def resolve_band_series(
    folders: Sequence[Path],
    labels: Sequence[str | None] = (),
    styles: Sequence[str | None] = (),
) -> tuple[list[BandSeries], list[str]]:
    """Return the band structures of the given runs, and any warnings.

    Series are labelled by the step that produced them, prefixed by the folder
    name when more than one folder is on the axes. ``labels`` names the folders
    instead, one per folder in the order they were given; a folder that yields
    several series keeps whatever tells them apart, so one name covers a
    per-spin or per-block fan-out and no two curves end up sharing a name.
    ``None`` leaves that folder's own series named or styled as if
    ``labels``/``styles`` had not been given for it at all, which is how a
    caller pairing values with only some of the folders spells "no value
    here" — an empty string is a value of its own (a no-op matplotlib format
    string counts as a style given), not a stand-in for "none". ``styles``
    are matplotlib format strings, given the same way and covering a fan-out
    the same way: every curve one folder contributes is drawn alike. Every
    folder must carry a band structure: drawing fewer curves than folders
    asked for reads as a figure of them all.

    :raises ValueError: if given, ``labels``/``styles`` do not number the
        folders.
    :raises PlottingError: if a folder is not a run directory, its run is not
        in this profile, or any of them holds nothing plottable.
    """
    _check_one_per_folder(labels, len(folders), "--label")
    _check_one_per_folder(styles, len(folders), "--style")

    nodes = [run_node(folder) for folder in folders]

    series: list[BandSeries] = []
    warnings: list[str] = []
    empty: list[tuple[Path, orm.ProcessNode]] = []
    for index, (folder, node) in enumerate(zip(folders, nodes, strict=True)):
        warning = _failure_warning(folder, node)
        if warning is not None:
            warnings.append(warning)
        found = _series_from_node(node)
        if not found:
            empty.append((folder, node))
        style_value = styles[index] if styles else None
        if style_value is not None:
            for item, _ in found:
                item.style = style_value

        label_value = labels[index] if labels else None
        if label_value is not None:
            _name_after_folder(found, label_value)
        elif len(folders) > 1:
            prefix = folder.name or folder.resolve().name
            for item, _ in found:
                item.label = f"{prefix}: {item.label}"
        series += [item for item, _ in found]

    if empty:
        raise _nothing_plottable(empty, len(folders))
    return series, warnings
