"""Tests for ``koopmans plot``: the folder resolver and the renderer.

The resolver tests build process nodes directly rather than running
workflows: what is under test is which producer/socket pairs count as a band
structure, and that is decided by ``process_type`` plus, for the pw.x base
run, the calculation type its own inputs declare.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from aiida import orm
from aiida.common.links import LinkType

from koopmans.aiida.dumping import NODE_METADATA_FILE
from koopmans.plotting import (
    DIVIDER_LABEL,
    BandSeries,
    EnergyZero,
    NoEnergyZeroError,
    PathMismatchError,
    PlottingError,
    StyleError,
    apply_energy_zero,
    check_paths_agree,
    check_style,
    describe_energy_zero,
    draw_band_structures,
    path_distances,
    render_band_structures,
    resolve_band_series,
    write_series_json,
)
from koopmans.plotting.resolve import SUGGESTION_LIMIT
from tests.fixtures import make_process

PW_BANDS = "aiida.workflows:quantumespresso.pw.bands"
PW_BASE = "aiida.workflows:quantumespresso.pw.base"
KCW_HAM = "aiida.calculations:koopmans.kcw_ham"
W90_BASE = "aiida.workflows:wannier90_workflows.base.wannier90"
W90_CALC = "aiida_wannier90.calculations.wannier90.Wannier90Calculation"
PW_CALC = "aiida.calculations:quantumespresso.pw"
W90_OPTIMIZE = "aiida.workflows:wannier90_workflows.optimize"
MERGE_INTERPOLATED_BANDS = "aiida_koopmans.workgraphs.auto_wannierize.merge_interpolated_bands"

#: A cubic cell, so that reciprocal-space distances are easy to reason about.
CUBIC = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------


def make_bands(
    kpoints: list[list[float]],
    energies: list[list[float]] | list[list[list[float]]],
    cell: list[list[float]] | None = None,
    labels: list[tuple[int, str]] | None = None,
    occupations: list[list[float]] | None = None,
) -> orm.BandsData:
    """Return an unstored ``BandsData`` holding the given eigenvalues.

    ``energies`` is ``[N_kpoints, N_bands]``, or ``[N_spin, N_kpoints,
    N_bands]`` for a collinear calculation's two channels.
    """
    bands = orm.BandsData()
    if cell is not None:
        bands.set_cell(cell)  # type: ignore[no-untyped-call]
    bands.set_kpoints(kpoints)  # type: ignore[no-untyped-call]
    bands.set_bands(energies, units="eV", occupations=occupations)  # type: ignore[no-untyped-call]
    if labels is not None:
        bands.labels = labels
    return bands


def make_spin_bands(kpoints: list[list[float]], energies: list[list[list[float]]]) -> orm.BandsData:
    """Return a ``BandsData`` holding one table per spin channel."""
    bands = orm.BandsData()
    bands.set_cell(CUBIC)  # type: ignore[no-untyped-call]
    bands.set_kpoints(kpoints)  # type: ignore[no-untyped-call]
    bands.set_bands(np.asarray(energies, dtype=float), units="eV")  # type: ignore[no-untyped-call]
    return bands


def attach(node: orm.ProcessNode, socket: str, data: orm.Data) -> orm.Data:
    """Link ``data`` as an output of ``node`` under the link label ``socket``."""
    if isinstance(node, orm.CalcJobNode):
        # A calculation creates its outputs, so the node must still be unstored.
        data.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label=socket)
        return data.store()
    # A workflow only returns data that already exists.
    data.store()
    data.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label=socket)
    return data


def write_run_folder(root: Path, name: str, node: orm.ProcessNode | None) -> Path:
    """Write a run folder whose metadata names ``node``, and return it."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    if node is not None:
        payload = {
            "Node data": {
                "label": node.label,
                "pk": node.pk,
                "uuid": node.uuid,
                "node_type": node.node_type,
                "is_finished_ok": node.is_finished_ok,
            }
        }
        (folder / NODE_METADATA_FILE).write_text(yaml.dump(payload, sort_keys=False))
    return folder


def band_lines(axes: Any) -> list[Any]:
    """Return the curves drawn on ``axes``, without the path dividers."""
    return [line for line in axes.get_lines() if line.get_label() != DIVIDER_LABEL]


def divider_positions(axes: Any) -> list[float]:
    """Return the x of every rule drawn at a high-symmetry point, in order.

    A rule is vertical, so both of its x data are the position; taking the
    first would pass a horizontal line off as one.
    """
    positions = []
    for line in axes.get_lines():
        if line.get_label() != DIVIDER_LABEL:
            continue
        xdata = list(line.get_xdata())
        assert xdata[0] == pytest.approx(xdata[-1]), "a path divider must be vertical"
        positions.append(float(xdata[0]))
    return sorted(positions)


def blank_axes() -> Any:
    """Return a fresh set of axes on a headless backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots()[1]


@pytest.fixture(autouse=True)
def close_figures() -> Any:
    """Close every figure a test opened.

    Held-open figures accumulate across the module until matplotlib warns
    about it, and the warning lands in the output a ``CliRunner`` test then
    asserts on.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def broken_path() -> BandSeries:
    """Return a series whose path runs G-X, then restarts at L and ends at G."""
    return series(
        kpoints=[
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
        ],
        energies=[[-5.0], [-4.5], [-4.0], [-3.0], [-3.5], [-4.0]],
        path_labels=[(0, "G"), (2, "X"), (3, "L"), (5, "G")],
    )


def three_corner_path() -> BandSeries:
    """Return a series whose unbroken path runs G-X-M, so X is interior to it.

    The 4 Angstrom cubic cell puts X at pi/4 and M at pi/2.
    """
    return series(
        kpoints=[
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.25, 0.0],
            [0.5, 0.5, 0.0],
        ],
        energies=[[-5.0], [-4.5], [-4.0], [-3.5], [-3.0]],
        path_labels=[(0, "G"), (2, "X"), (4, "M")],
    )


def series(label: str = "DFT", **overrides: Any) -> BandSeries:
    """Return a two-k-point, two-band series, with fields overridden."""
    fields: dict[str, Any] = {
        "label": label,
        "kpoints": [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "energies": [[-5.0, 5.0], [-4.5, 5.5], [-4.0, 6.0]],
        "cell": CUBIC,
        "path_labels": [(0, "G"), (2, "X")],
    }
    fields.update(overrides)
    return BandSeries(**fields)


# ----------------------------------------------------------------------
# The energy zero
# ----------------------------------------------------------------------


class TestEnergyZero:
    """One shift for the whole figure, taken from one designated series."""

    def test_zero_is_shared_across_series(self) -> None:
        """Every series is shifted by the first series' valence band edge.

        Referencing each series to its own edge would move both curves' edges
        to zero and erase the 0.8 eV shift between them, which is the physical
        result the overlay exists to show. Asserting on the *shifted* edges is
        what discriminates: a per-series zero passes any test that only checks
        that a shift happened.
        """
        dft = series("DFT", vbm=6.0)
        ki = series("KI", vbm=5.2)

        value, reference = apply_energy_zero([dft, ki], EnergyZero.VBM)

        assert value == pytest.approx(6.0)
        assert reference is dft
        assert dft.zero == ki.zero == pytest.approx(6.0)
        # The band-edge shift survives the referencing.
        assert (5.2 - ki.zero) - (6.0 - dft.zero) == pytest.approx(-0.8)

    def test_zero_falls_through_to_a_series_that_reports_one(self) -> None:
        """A leading series with no edge does not veto the ones behind it."""
        interpolation = series("Wannier interpolation")
        ki = series("KI", vbm=5.2)

        value, reference = apply_energy_zero([interpolation, ki], EnergyZero.VBM)

        assert value == pytest.approx(5.2)
        assert reference is ki
        assert interpolation.zero == pytest.approx(5.2)

    def test_fermi_uses_the_fermi_level(self) -> None:
        """``--zero fermi`` ignores a valence band edge that is also present."""
        dft = series("DFT", vbm=6.0, fermi=6.4)

        value, _ = apply_energy_zero([dft], EnergyZero.FERMI)

        assert value == pytest.approx(6.4)

    def test_zero_none_leaves_the_energies_alone(self) -> None:
        """``--zero none`` applies no shift even when an edge is available."""
        dft = series("DFT", vbm=6.0)

        value, reference = apply_energy_zero([dft], EnergyZero.NONE)

        assert value == 0.0
        assert reference is None
        assert dft.zero == 0.0

    def test_missing_reference_names_the_alternatives(self) -> None:
        """Nothing to subtract is an error that says what to pass instead."""
        with pytest.raises(NoEnergyZeroError) as excinfo:
            apply_energy_zero([series("Wannier interpolation")], EnergyZero.VBM)

        assert "--zero fermi" in str(excinfo.value)
        assert "--zero none" in str(excinfo.value)

    def test_description_states_the_value_and_its_source(self) -> None:
        """The figure's caption names both the reference series and the shift."""
        dft = series("DFT", vbm=6.2452)
        value, reference = apply_energy_zero([dft], EnergyZero.VBM)

        caption = describe_energy_zero(EnergyZero.VBM, value, reference)

        assert caption == "energies relative to the valence band edge of 'DFT' at 6.2452 eV"


# ----------------------------------------------------------------------
# Labels and the data file
# ----------------------------------------------------------------------


class TestSeriesRecords:
    """The records the resolver hands the renderer."""

    def test_data_file_holds_the_energies_and_the_zero(self, tmp_path: Path) -> None:
        """``--data`` writes energies as computed plus the shift that was applied.

        Keeping the two separate is what makes the file enough to redraw the
        figure and to re-reference it.
        """
        dft = series("DFT", vbm=6.0)
        apply_energy_zero([dft], EnergyZero.VBM)

        target = tmp_path / "bands.json"
        write_series_json([dft], target)
        payload = json.loads(target.read_text())

        (record,) = payload["series"]
        assert record["label"] == "DFT"
        assert record["zero"] == pytest.approx(6.0)
        assert record["energies"] == [[-5.0, 5.0], [-4.5, 5.5], [-4.0, 6.0]]
        assert record["path_labels"] == [[0, "G"], [2, "X"]]


# ----------------------------------------------------------------------
# The path axis
# ----------------------------------------------------------------------


class TestPathDistances:
    """How far along the x axis each k-point sits."""

    def test_distance_uses_the_reciprocal_cell(self) -> None:
        """A cell turns crystal coordinates into reciprocal-space distance.

        With a 4 Angstrom cubic cell the reciprocal vectors have length
        2*pi/4, so a half-way k-point sits at pi/4.
        """
        distances = path_distances(series())

        assert distances[-1] == pytest.approx(np.pi / 4)

    def test_without_a_cell_the_axis_is_crystal_coordinates(self) -> None:
        """Falling back distorts the axis, which is why the cell is threaded."""
        distances = path_distances(series(cell=None))

        assert distances[-1] == pytest.approx(0.5)

    def test_a_jump_contributes_no_distance(self) -> None:
        """Two adjacent labelled points are a discontinuity, not a step.

        Without this the gap between the end of one branch and the start of
        the next would stretch the axis by the distance between two unrelated
        corners of the Brillouin zone.
        """
        distances = path_distances(broken_path())

        assert distances[3] == pytest.approx(distances[2])


# ----------------------------------------------------------------------
# The resolver
# ----------------------------------------------------------------------


class TestResolver:
    """Turning run folders into series."""

    def test_pw_bands_becomes_a_dft_series(self, aiida_profile: Any, tmp_path: Path) -> None:
        """A ``PwBandsWorkChain`` ``band_structure`` is the declared DFT series."""
        root = make_process("aiida.workflows:workgraph.engine", label="RunPwBands")
        bands_chain = make_process(PW_BANDS, caller=root, link_label="bands")
        attach(
            bands_chain,
            "band_structure",
            make_bands(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                [[-5.0, 5.0], [-4.0, 6.0]],
                cell=CUBIC,
                labels=[(0, "G"), (1, "X")],
                occupations=[[2.0, 0.0], [2.0, 0.0]],
            ),
        )
        folder = write_run_folder(tmp_path, "si_lda", root)

        found, warnings = resolve_band_series([folder])

        assert warnings == []
        assert [item.label for item in found] == ["DFT"]
        # The valence band edge is the highest occupied energy, not the
        # highest energy on the axes.
        assert found[0].vbm == pytest.approx(-4.0)
        assert found[0].cell == CUBIC
        assert found[0].path_labels == [(0, "G"), (1, "X")]

    def test_a_calculation_directory_is_a_folder_of_its_own(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """A dumped calculation plots on its own, without its run around it.

        A dump keeps a metadata file beside every calculation, so a single
        step of a run can be given to the command and styled apart from the
        rest. This pins that the resolver reads such a directory: a folder is
        anything holding the metadata file, not only the root of a run.
        """
        calculation = make_process(
            W90_CALC, calcjob=True, computer=aiida_localhost, process_label="Wannier90Calculation"
        )
        attach(calculation, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        folder = write_run_folder(tmp_path, "03-wannier90", calculation)

        found, warnings = resolve_band_series([folder], ("Wannier",), ("b-",))

        assert warnings == []
        assert [(item.label, item.style) for item in found] == [("Wannier", "b-")]

    def test_fermi_falls_back_to_the_scf(self, aiida_profile: Any, tmp_path: Path) -> None:
        """A bands step that inherited no Fermi level takes the scf's.

        A pw.x run along a k-path need not report one of its own, so
        ``--zero fermi`` would otherwise have nothing to subtract for the one
        route that always produces a band structure.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="RunPwBands")
        chain = make_process(PW_BANDS, caller=root, link_label="bands")
        attach(chain, "band_structure", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        attach(chain, "scf_parameters", orm.Dict({"fermi_energy": 6.31}))  # type: ignore[no-untyped-call]
        folder = write_run_folder(tmp_path, "si_lda", root)

        found, _ = resolve_band_series([folder])

        assert found[0].fermi == pytest.approx(6.31)

    def test_scf_bands_are_not_plottable(self, aiida_profile: Any, tmp_path: Path) -> None:
        """An scf ``output_band`` is eigenvalues on a mesh and is left out.

        It is the same shape of data as an interpolated path — same node type,
        same arrays — so nothing but the declared producer table tells them
        apart. Inferring plottability from the data would pick this up and
        draw a mesh against a k-path axis.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="RunPwBands")
        scf = make_process(PW_BASE, caller=root, link_label="scf")
        attach(
            scf,
            "output_band",
            make_bands([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], [[-5.0, 5.0], [-4.5, 5.5]]),
        )
        folder = write_run_folder(tmp_path, "si_scf", root)

        with pytest.raises(PlottingError, match="No band structure to plot"):
            resolve_band_series([folder])

    def test_kcw_ham_becomes_a_ki_series(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """kcw.x reports its own valence band edge in ``output_parameters``."""
        root = make_process("aiida.workflows:workgraph.engine", label="SinglepointDFPTWorkflow")
        ham = make_process(
            KCW_HAM, caller=root, link_label="ham", calcjob=True, computer=aiida_localhost
        )
        attach(
            ham,
            "bands",
            make_bands(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                [[-5.4, 5.2], [-4.4, 6.2]],
                cell=CUBIC,
                labels=[(0, "G"), (1, "X")],
            ),
        )
        attach(ham, "output_parameters", orm.Dict({"ki_homo_energy": 5.2}))  # type: ignore[no-untyped-call]
        folder = write_run_folder(tmp_path, "si_ki", root)

        found, _ = resolve_band_series([folder])

        assert [item.label for item in found] == ["KI"]
        assert found[0].vbm == pytest.approx(5.2)

    def test_two_folders_are_qualified_by_folder_name(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """One folder needs no prefix; two do, or the legend is ambiguous."""
        folders = []
        for name in ("si_lda", "si_lda_2"):
            root = make_process("aiida.workflows:workgraph.engine", label="RunPwBands")
            chain = make_process(PW_BANDS, caller=root, link_label="bands")
            attach(
                chain,
                "band_structure",
                make_bands([[0.0, 0.0, 0.0]], [[-5.0, 5.0]], occupations=[[2.0, 0.0]]),
            )
            folders.append(write_run_folder(tmp_path, name, root))

        found, _ = resolve_band_series(folders)

        assert [item.label for item in found] == ["si_lda: DFT", "si_lda_2: DFT"]

    def test_repeated_producer_is_told_apart_by_step_name(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A spin fan-out runs the same producer twice; the step names split them."""
        root = make_process("aiida.workflows:workgraph.engine", label="SinglepointDFPTWorkflow")
        for step in ("ham_up", "ham_down"):
            chain = make_process(PW_BANDS, caller=root, link_label=step)
            attach(chain, "band_structure", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        folder = write_run_folder(tmp_path, "si", root)

        found, _ = resolve_band_series([folder])

        assert sorted(item.label for item in found) == ["DFT (ham_down)", "DFT (ham_up)"]

    def test_call_chain_disambiguates_a_per_block_fan_out(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A per-block Wannierization names every wannier90 call the same.

        Each block's wannier90 run sits under a sub-graph call named after
        the block; wannier90 itself is always called "wannier90", so the
        immediate step name ties and disambiguation has to walk up to the
        sub-graph call that actually names the block.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="SinglepointDFPTWorkflow")
        for block in ("wannierize_emp", "wannierize_occ_1"):
            sub_graph = make_process(
                "aiida.workflows:workgraph.engine", caller=root, link_label=block
            )
            base = make_process(W90_BASE, caller=sub_graph, link_label="wannier90")
            attach(base, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        folder = write_run_folder(tmp_path, "zno", root)

        found, _ = resolve_band_series([folder])

        assert sorted(item.label for item in found) == [
            "Wannier interpolation (emp)",
            "Wannier interpolation (occ 1)",
        ]

    def test_nested_single_producer_stays_unqualified(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """The same sub-graph nesting with only one producer needs no suffix.

        Disambiguation is triggered by a tied series name, not by sitting
        under a sub-graph call.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="Wannierize")
        sub_graph = make_process(
            "aiida.workflows:workgraph.engine", caller=root, link_label="wannierize_occ"
        )
        base = make_process(W90_BASE, caller=sub_graph, link_label="wannier90")
        attach(base, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        folder = write_run_folder(tmp_path, "si_w90", root)

        found, _ = resolve_band_series([folder])

        assert [item.label for item in found] == ["Wannier interpolation"]

    def test_no_common_depth_falls_back_to_numbering(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A tied pair with no depth in common everywhere is told apart by number.

        One producer is called directly by the run root; another with the
        same immediate call label sits one level deeper. No depth compares
        across the whole group, so nothing structural distinguishes them.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="Wannierize")
        direct = make_process(W90_BASE, caller=root, link_label="wannier90")
        attach(direct, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        mid = make_process("aiida.workflows:workgraph.engine", caller=root, link_label="mid")
        nested = make_process(W90_BASE, caller=mid, link_label="wannier90")
        attach(nested, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.1]]))
        folder = write_run_folder(tmp_path, "si_w90", root)

        found, _ = resolve_band_series([folder])

        assert sorted(item.label for item in found) == [
            "Wannier interpolation (1)",
            "Wannier interpolation (2)",
        ]

    def test_labels_with_no_shared_prefix_are_rendered_unstripped(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """Distinguishing labels that share no lead-in are used exactly as they are.

        Prefix-stripping only fires on the part the tied group's labels
        actually share; two sub-graph calls named without any common
        lead-in still disambiguate correctly once the shared immediate
        ``wannier90`` label forces the walk to escalate.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="SinglepointDFPTWorkflow")
        for block in ("north", "south"):
            sub_graph = make_process(
                "aiida.workflows:workgraph.engine", caller=root, link_label=block
            )
            base = make_process(W90_BASE, caller=sub_graph, link_label="wannier90")
            attach(base, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        folder = write_run_folder(tmp_path, "zno", root)

        found, _ = resolve_band_series([folder])

        assert sorted(item.label for item in found) == [
            "Wannier interpolation (north)",
            "Wannier interpolation (south)",
        ]

    def test_merge_interpolated_bands_is_recognized(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """The split-mode merge calcfunction's output is a plottable band structure.

        Before this producer is registered, a folder addressing this
        calcfunction directly reports that no step of it produced a
        plottable output.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="WannierizeBlocks")
        merged = make_process(
            MERGE_INTERPOLATED_BANDS,
            caller=root,
            link_label="merge_interpolated_bands",
            calcjob=True,
            computer=aiida_localhost,
        )
        attach(merged, "result", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        folder = write_run_folder(tmp_path, "zno", root)

        found, _ = resolve_band_series([folder])

        assert [item.label for item in found] == ["Wannier interpolation"]

    def test_split_mode_names_the_gauge_fragments_and_merge(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """A whole-run plot of one split block shows every reachable curve.

        The pre-split whole-block gauge, each re-Wannierised group, and the
        block-wide merge all share the "Wannier interpolation" series name
        and are told apart structurally, without falling back to bare
        numbering; a sibling block wannierized directly is unaffected.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="WannierizeBlocks")

        direct = make_process(
            "aiida.workflows:workgraph.engine", caller=root, link_label="wannierize_occ"
        )
        direct_leaf = make_process(W90_BASE, caller=direct, link_label="wannier90")
        attach(direct_leaf, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))

        split = make_process(
            "aiida.workflows:workgraph.engine", caller=root, link_label="wannierize_split_emp"
        )
        gauge = make_process(
            "aiida.workflows:workgraph.engine", caller=split, link_label="wannierize_whole_block"
        )
        gauge_leaf = make_process(W90_BASE, caller=gauge, link_label="wannier90")
        attach(gauge_leaf, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.1]]))

        rewannierize = make_process(
            "aiida.workflows:workgraph.engine", caller=split, link_label="rewannierize_split_blocks"
        )
        for i in range(2):
            fragment = make_process(
                W90_CALC,
                caller=rewannierize,
                link_label=f"wannier90_split_block_{i}",
                calcjob=True,
                computer=aiida_localhost,
            )
            attach(fragment, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.2 - i]]))
        merged = make_process(
            MERGE_INTERPOLATED_BANDS,
            caller=rewannierize,
            link_label="merge_interpolated_bands",
            calcjob=True,
            computer=aiida_localhost,
        )
        attach(merged, "result", make_bands([[0.0, 0.0, 0.0]], [[-5.4]]))

        folder = write_run_folder(tmp_path, "zno", root)

        found, _ = resolve_band_series([folder])
        labels = sorted(item.label for item in found)

        # Five curves, every one distinct, and none of them bare numbering:
        # a whole-block gauge, two split fragments, a merge, and the
        # sibling block wannierized directly.
        assert len(labels) == len(set(labels)) == 5
        assert not any(label.rsplit("(", 1)[-1].rstrip(")").strip().isdigit() for label in labels)
        assert "Wannier interpolation (occ)" in labels

    def test_split_mode_two_blocks_combines_depths_for_matching_fragments(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """Two split blocks each contribute a same-indexed fragment and a merge.

        Fragment 0 of one block and fragment 0 of the other tie at every
        depth their own sub-graph shares (the fragment index, the
        "rewannierize_split_blocks" wrapper); only the block name two
        levels up (which itself only distinguishes once combined with the
        fragment index) tells them apart, so disambiguation has to combine
        more than one depth rather than stopping at the first that splits
        anything.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="WannierizeBlocks")

        for block in ("occ", "emp"):
            split = make_process(
                "aiida.workflows:workgraph.engine",
                caller=root,
                link_label=f"wannierize_split_{block}",
            )
            rewannierize = make_process(
                "aiida.workflows:workgraph.engine",
                caller=split,
                link_label="rewannierize_split_blocks",
            )
            for i in range(2):
                fragment = make_process(
                    W90_CALC,
                    caller=rewannierize,
                    link_label=f"wannier90_split_block_{i}",
                    calcjob=True,
                    computer=aiida_localhost,
                )
                attach(fragment, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0 - i]]))
            merged = make_process(
                MERGE_INTERPOLATED_BANDS,
                caller=rewannierize,
                link_label="merge_interpolated_bands",
                calcjob=True,
                computer=aiida_localhost,
            )
            attach(merged, "result", make_bands([[0.0, 0.0, 0.0]], [[-5.5]]))

        folder = write_run_folder(tmp_path, "zno", root)

        found, _ = resolve_band_series([folder])
        labels = sorted(item.label for item in found)

        # Six curves: two fragments and a merge, per block, all distinct.
        assert len(labels) == len(set(labels)) == 6
        assert not any(label.rsplit("(", 1)[-1].rstrip(")").strip().isdigit() for label in labels)

    def test_not_a_run_directory(self, aiida_profile: Any, tmp_path: Path) -> None:
        """A folder that is not a run is named, along with what to pass.

        The bookkeeping file the dump writes is not named: the reader neither
        writes it nor can act on it, and every message on this path says the
        same thing.
        """
        folder = tmp_path / "somewhere"
        folder.mkdir()

        with pytest.raises(PlottingError) as excinfo:
            resolve_band_series([folder])

        assert "is not a koopmans run directory" in str(excinfo.value)
        assert "koopmans run" in str(excinfo.value)
        assert NODE_METADATA_FILE not in str(excinfo.value)

    def test_unreadable_metadata_names_the_folder_not_the_file(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A folder whose metadata records no uuid is named by its own path.

        The reader passed the folder, not the file inside it, and rerunning is
        the only thing they can do about either.
        """
        folder = write_run_folder(tmp_path, "zno", None)
        (folder / NODE_METADATA_FILE).write_text(yaml.dump({"Node data": {"pk": 1}}))

        with pytest.raises(PlottingError) as excinfo:
            resolve_band_series([folder])

        message = str(excinfo.value)
        assert "does not record which run it came from" in message
        assert str(folder) in message
        assert NODE_METADATA_FILE not in message

    def test_uuid_absent_from_this_profile(self, aiida_profile: Any, tmp_path: Path) -> None:
        """A folder from another machine says so, rather than "node not found"."""
        folder = write_run_folder(tmp_path, "elsewhere", None)
        (folder / NODE_METADATA_FILE).write_text(
            yaml.dump({"Node data": {"uuid": "00000000-0000-0000-0000-000000000000"}})
        )

        with pytest.raises(PlottingError) as excinfo:
            resolve_band_series([folder])

        assert "not in this AiiDA profile" in str(excinfo.value)

    def test_unfinished_run_warns_and_plots_what_is_there(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A killed run's partial bands are usually what the user wants to see."""
        root = make_process(
            "aiida.workflows:workgraph.engine",
            label="RunPwBands",
            exit_status=402,
            exit_message="The scf PwBaseWorkChain sub process failed",
        )
        failed = make_process(PW_BASE, caller=root, link_label="scf", exit_status=402)
        failed.set_exit_message("Out of walltime")
        chain = make_process(PW_BANDS, caller=root, link_label="bands")
        attach(chain, "band_structure", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        folder = write_run_folder(tmp_path, "si_partial", root)

        found, warnings = resolve_band_series([folder])

        assert len(found) == 1
        assert len(warnings) == 1
        assert "scf failed" in warnings[0]
        assert "Out of walltime" in warnings[0]

    def test_a_root_level_failure_with_no_failed_step_blames_the_run_itself(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A run that fails with no failed descendant blames the run, not a step.

        The engine can except before any child it started went on to fail —
        every descendant still finishes fine, so the run's own root is the
        only node left to name. The root carries no CALL link of its own, so
        the step-naming fallback has to use its process label instead.
        """
        root = make_process(
            "aiida.workflows:workgraph.engine",
            label="RunPwBands",
            exit_status=500,
            exit_message="WorkGraph excepted",
        )
        chain = make_process(PW_BANDS, caller=root, link_label="bands")
        attach(chain, "band_structure", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        folder = write_run_folder(tmp_path, "si_excepted", root)

        found, warnings = resolve_band_series([folder])

        assert len(found) == 1
        assert len(warnings) == 1
        assert "did not finish" in warnings[0]
        assert "WorkGraph excepted" in warnings[0]

    def test_nothing_plottable_names_the_route_and_the_reason(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """The ΔSCF route has a known, specific reason for drawing a blank."""
        root = make_process("aiida.workflows:workgraph.engine", label="KoopmansDSCFWorkflow")
        folder = write_run_folder(tmp_path, "si_dscf", root)

        with pytest.raises(PlottingError) as excinfo:
            resolve_band_series([folder])

        message = str(excinfo.value)
        assert "KoopmansDSCFWorkflow" in message
        assert "supercell" in message

    def test_unknown_route_lists_what_koopmans_plots(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A route with no recorded reason still names the sockets it looked for."""
        root = make_process("aiida.workflows:workgraph.engine", label="SomethingElse")
        folder = write_run_folder(tmp_path, "si_other", root)

        with pytest.raises(PlottingError) as excinfo:
            resolve_band_series([folder])

        assert "interpolated_bands" in str(excinfo.value)


# ----------------------------------------------------------------------
# The renderer
# ----------------------------------------------------------------------


class TestRenderer:
    """Drawing the records."""

    def test_writes_the_requested_format(self, tmp_path: Path) -> None:
        """The extension chooses the format, and nothing opens a window."""
        target = tmp_path / "bands.pdf"

        render_band_structures([series("DFT")], output_path=target, zero=EnergyZero.VBM)

        assert target.is_file()
        assert target.read_bytes().startswith(b"%PDF")

    def test_every_series_reaches_the_axes(self) -> None:
        """Both curves are drawn: an overlay is the point of the figure."""
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT"), series("KI")])

        assert len(band_lines(axes)) == 4  # two bands each
        labels = {line.get_label() for line in band_lines(axes)}
        assert {"DFT", "KI"} <= labels

    def test_the_zero_is_subtracted_when_drawing(self) -> None:
        """The curve is drawn at ``energy - zero``, not at the raw energy."""
        shifted = series("DFT", zero=6.0)
        axes = blank_axes()

        draw_band_structures(axes, [shifted])

        assert band_lines(axes)[0].get_ydata()[0] == pytest.approx(-11.0)

    def test_special_points_become_the_ticks(self) -> None:
        """Names are drawn as symbols, at the distance their k-point sits at."""
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT")])

        assert [text.get_text() for text in axes.get_xticklabels()] == ["\u0393", "X"]

    def test_a_jump_splits_the_curve(self) -> None:
        """Across a discontinuity the line breaks rather than joining two branches."""
        axes = blank_axes()

        draw_band_structures(axes, [broken_path()])

        assert [len(line.get_xdata()) for line in band_lines(axes)] == [3, 3]

    def test_ticks_sit_at_their_k_points_distance(self) -> None:
        """A tick lands where its k-point does, not at its index.

        The 4 Angstrom cubic cell puts X at pi/4; an index-based x axis would
        put it at 2 and still carry the right name.
        """
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT")])

        assert list(axes.get_xticks()) == pytest.approx([0.0, np.pi / 4])

    def test_the_axis_spans_the_path_and_no_more(self) -> None:
        """Without limits the axes pad the path with empty margins."""
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT")])

        assert axes.get_xlim() == pytest.approx((0.0, np.pi / 4))

    def test_the_axis_spans_the_path_with_nothing_named(self) -> None:
        """The limits come from the curves, so a run naming no points gets them too.

        Bands dumped before the k-path prerequisites carry no labels at all;
        limits taken from the ticks would leave those figures padded.
        """
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT", path_labels=[])])

        assert axes.get_xlim() == pytest.approx((0.0, np.pi / 4))

    def test_the_limits_do_not_clip_the_curve(self) -> None:
        """A path sampled past its last named point still reaches the spine.

        Limits taken from the last tick would cut the unnamed tail off at
        pi/4 while the curve runs to 3*pi/8.
        """
        axes = blank_axes()

        draw_band_structures(
            axes,
            [
                series(
                    "DFT",
                    kpoints=[[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0], [0.75, 0.0, 0.0]],
                    energies=[[-5.0], [-4.5], [-4.0], [-3.5]],
                    path_labels=[(0, "G"), (2, "X")],
                )
            ],
        )

        assert axes.get_xlim() == pytest.approx((0.0, 3 * np.pi / 8))

    def test_a_rule_is_drawn_at_each_interior_special_point(self) -> None:
        """The rules are what separates one segment of the path from the next.

        Only X is interior: G and M sit on the spines, which draw them already.
        A rule at every named point would put one on each spine, and a rule
        placed by index would put X at 2 rather than at pi/4.
        """
        axes = blank_axes()

        draw_band_structures(axes, [three_corner_path()])

        assert divider_positions(axes) == pytest.approx([np.pi / 4])

    def test_a_discontinuity_draws_one_rule(self) -> None:
        """The two sides of a jump sit at one position, so they get one rule.

        Drawing one per named point would stack two rules there, at twice the
        weight of every other.
        """
        axes = blank_axes()

        draw_band_structures(axes, [broken_path()])

        assert divider_positions(axes) == pytest.approx([np.pi / 4])

    def test_a_jump_joins_two_names_on_one_tick(self) -> None:
        """The two sides of a discontinuity share a position, so they share a tick."""
        axes = blank_axes()

        draw_band_structures(axes, [broken_path()])

        assert [text.get_text() for text in axes.get_xticklabels()] == ["\u0393", "X|L", "\u0393"]

    def test_no_title_is_drawn(self) -> None:
        """The axes carry no title, whatever zero the figure was given."""
        axes = blank_axes()

        draw_band_structures(axes, [series("KI")], zero=EnergyZero.VBM)

        assert axes.get_title() == ""

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            (EnergyZero.VBM, "$E - E_\\mathrm{VBM}$ (eV)"),
            (EnergyZero.FERMI, "$E - E_\\mathrm{F}$ (eV)"),
            (EnergyZero.NONE, "Energy (eV)"),
        ],
    )
    def test_the_y_axis_names_the_energy_that_was_subtracted(
        self, kind: EnergyZero, expected: str
    ) -> None:
        """A saved figure has to disclose its zero somewhere, and the title is gone.

        A bare "Energy (eV)" over shifted energies reads as the energies as
        computed.
        """
        axes = blank_axes()

        draw_band_structures(axes, [series("KI")], zero=kind)

        assert axes.get_ylabel() == expected

    def test_one_series_carries_no_legend(self) -> None:
        """A single curve needs no key to tell it from the others."""
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT")])

        assert axes.get_legend() is None

    def test_a_split_series_carries_no_legend(self) -> None:
        """A discontinuity draws several lines, and it is still one series.

        Keying the legend off the drawn lines would bring it back for every
        band structure whose path has a break.
        """
        axes = blank_axes()

        draw_band_structures(axes, [broken_path()])

        assert len(band_lines(axes)) > 1
        assert axes.get_legend() is None

    def test_an_overlay_carries_a_legend_naming_every_series(self) -> None:
        """Two curves are told apart by their labels, so the key stays."""
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT"), series("KI")])

        legend = axes.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["DFT", "KI"]

    def test_a_series_without_a_cell_shares_the_axis(self) -> None:
        """One reciprocal basis measures every curve on the figure.

        Measuring a cell-less series in crystal coordinates would squeeze it
        into the left fraction of the axes while its neighbour spans them,
        with both curves belonging to the same path.
        """
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT"), series("KI", cell=None)])

        spans = [list(line.get_xdata()) for line in band_lines(axes)]
        assert spans[0] == pytest.approx(spans[-1])

    def test_the_ticks_come_from_a_series_that_has_them(self) -> None:
        """A series naming no special points does not cost the figure its axis."""
        axes = blank_axes()

        draw_band_structures(axes, [series("KI", path_labels=[], cell=None), series("DFT")])

        assert [text.get_text() for text in axes.get_xticklabels()] == ["\u0393", "X"]

    def test_the_legend_can_be_asked_for_on_one_curve(self) -> None:
        """A caller that named its one series gets a key naming it.

        The control is ``test_one_series_carries_no_legend`` above: the same
        single series draws no key when nothing asks for one.
        """
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT")], legend=True)

        legend = axes.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["DFT"]

    def test_the_legend_can_be_refused_on_an_overlay(self) -> None:
        """Asking for no key overrides the rule that an overlay carries one."""
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT"), series("KI")], legend=False)

        assert axes.get_legend() is None

    def test_the_energy_axis_spans_the_bands_by_default(self) -> None:
        """With no range asked for, every band stays on the figure.

        The control for the clipping tests below: the limits must start out
        wide enough that a narrower window is visibly the option's doing.
        """
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT")])

        low, high = axes.get_ylim()
        assert low <= -5.0 and high >= 6.0

    def test_the_energy_axis_is_clipped_to_the_range_asked_for(self) -> None:
        """A window narrower than the bands is what the axes show.

        A semicore band 130 eV down otherwise compresses the gap region into
        a sliver, which is the whole reason to ask for a range.
        """
        axes = blank_axes()
        deep = series("DFT", energies=[[-130.0, 5.0], [-130.0, 5.5], [-130.0, 6.0]])

        draw_band_structures(axes, [deep], ylim=(-2.0, 8.0))

        assert axes.get_ylim() == pytest.approx((-2.0, 8.0))

    def test_clipping_measures_the_range_from_the_zero(self) -> None:
        """The range is read in the shifted energies the axis is drawn in.

        Applying it to the raw energies would frame a different window for
        every choice of ``--zero``, while the axis is labelled ``E - E_VBM``.
        """
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT", zero=6.0)], ylim=(-1.0, 1.0))

        assert axes.get_ylim() == pytest.approx((-1.0, 1.0))
        # -5.0 eV as computed is -11.0 eV once the zero is subtracted, so it
        # is outside the window rather than at its lower edge.
        assert band_lines(axes)[0].get_ydata()[0] == pytest.approx(-11.0)

    def test_clipping_keeps_the_figure_conventions(self) -> None:
        """A clipped figure still carries its rules, labels and tight x limits."""
        axes = blank_axes()

        draw_band_structures(axes, [three_corner_path()], ylim=(-4.5, -3.5))

        assert divider_positions(axes) == pytest.approx([np.pi / 4])
        assert [text.get_text() for text in axes.get_xticklabels()] == ["\u0393", "X", "M"]
        assert axes.get_xlim() == pytest.approx((0.0, np.pi / 2))
        assert axes.get_title() == ""


# ----------------------------------------------------------------------
# How a series is drawn
# ----------------------------------------------------------------------


def as_rgba(color: Any) -> tuple[float, float, float, float]:
    """Return a color in the one form 'k', 'C1' and a drawn line's own all take."""
    from matplotlib.colors import to_rgba

    return to_rgba(color)


class TestSeriesStyles:
    """A series drawn in a matplotlib format string of its own."""

    def test_a_style_sets_the_marker_and_line_of_every_band(self) -> None:
        """The format string reaches each of the series' bands, not just its first.

        A band is a plot call of its own, so a style applied where the series
        is set up rather than where each curve is drawn would style one band
        and leave the rest solid.
        """
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT", style="x")])

        lines = band_lines(axes)
        assert len(lines) == 2
        assert {line.get_marker() for line in lines} == {"x"}
        assert {line.get_linestyle() for line in lines} == {"None"}

    def test_an_unstyled_series_keeps_the_plain_curve(self) -> None:
        """The control: without a style the bands are solid lines, as before."""
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT")])

        lines = band_lines(axes)
        assert {line.get_linestyle() for line in lines} == {"-"}
        assert {line.get_marker() for line in lines} == {"None"}
        assert {as_rgba(line.get_color()) for line in lines} == {as_rgba("C0")}

    def test_a_color_in_the_style_replaces_the_assigned_one(self) -> None:
        """A style naming a color owns it; the unstyled series keeps its own.

        Handing matplotlib both a format string and a ``color`` keyword lets
        the keyword win, which would make an explicit 'k-' come out in the
        automatic color it was given to escape.
        """
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT"), series("KI", style="k-")])

        first, second = band_lines(axes)[:2], band_lines(axes)[2:]
        assert {as_rgba(line.get_color()) for line in first} == {as_rgba("C0")}
        assert {as_rgba(line.get_color()) for line in second} == {as_rgba("k")}

    def test_a_style_naming_no_color_keeps_one_color_per_series(self) -> None:
        """Crosses come out in the series' own color, not one per band.

        matplotlib advances its color cycle once per plot call and a series is
        drawn a band at a time, so leaving the color to a format string that
        names none draws a single band structure in as many colors as it has
        bands.
        """
        axes = blank_axes()

        draw_band_structures(axes, [series("DFT"), series("KI", style="x")])

        first, second = band_lines(axes)[:2], band_lines(axes)[2:]
        assert {as_rgba(line.get_color()) for line in second} == {as_rgba("C1")}
        assert {as_rgba(line.get_color()) for line in first} == {as_rgba("C0")}

    def test_the_matplotlib_vocabulary_is_accepted(self) -> None:
        """The strings the option's help offers are all readable as written."""
        for style in ("x", "-", "--", "k--", "rx", "C1--", "o", "k--x"):
            check_style(style)

    def test_a_string_matplotlib_cannot_read_is_refused(self) -> None:
        """A typo is caught by the check rather than by the drawing."""
        with pytest.raises(StyleError, match="not a valid format string"):
            check_style("zz")

    def test_a_matplotlib_without_the_parser_says_so(self, monkeypatch: Any) -> None:
        """A renamed parser is reported, not raised as a bare ImportError.

        matplotlib publishes no format-string parser, so koopmans reads a
        private one and is unpinned against it moving; this is what that costs
        the reader when it does.
        """
        from matplotlib.axes import _base

        monkeypatch.delattr(_base, "_process_plot_format")

        with pytest.raises(StyleError, match="cannot say whether a format string is valid"):
            check_style("rx")


# ----------------------------------------------------------------------
# The command
# ----------------------------------------------------------------------


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return a click runner whose profile is the one pytest already loaded."""
    from click.testing import CliRunner

    import koopmans.cli

    monkeypatch.setattr(koopmans.cli, "load_koopmans_profile", lambda: None)
    return CliRunner()


@pytest.fixture
def drawn_axes(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return the axes each figure the command draws was drawn on, in order.

    The command closes its figure before returning, so the axes are only
    reachable while it runs. Drawing itself is left alone, which is what makes
    a test on these axes a test of the figure rather than of the call.
    """
    from koopmans.plotting import render

    seen: list[Any] = []
    original = render.draw_band_structures

    def record(axes: Any, *args: Any, **kwargs: Any) -> None:
        """Call through to the renderer, keeping the axes it drew."""
        seen.append(axes)
        original(axes, *args, **kwargs)

    monkeypatch.setattr(render, "draw_band_structures", record)
    return seen


def dft_run(tmp_path: Path, name: str, vbm: float, energies: list[list[float]]) -> Path:
    """Write a run folder holding one pw.x band structure along G-X."""
    root = make_process("aiida.workflows:workgraph.engine", label="RunPwBands")
    chain = make_process(PW_BANDS, caller=root, link_label="bands")
    occupations = [[2.0 if energy <= vbm else 0.0 for energy in row] for row in energies]
    attach(
        chain,
        "band_structure",
        make_bands(
            [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]],
            energies,
            cell=CUBIC,
            labels=[(0, "G"), (2, "X")],
            occupations=occupations,
        ),
    )
    return write_run_folder(tmp_path, name, root)


class TestCommand:
    """``koopmans plot bandstructure`` end to end."""

    def test_one_folder_writes_a_file_and_prints_its_path(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """The default writes a file and never blocks on a window."""
        from koopmans.cli import cli

        folder = dft_run(tmp_path, "zno", 6.2452, [[-5.0, 6.2452], [-4.5, 7.0], [-4.0, 7.5]])

        with runner.isolated_filesystem(temp_dir=tmp_path) as workdir:
            result = runner.invoke(cli, ["plot", "bandstructure", str(folder)])
            assert result.exit_code == 0, result.output
            assert (Path(workdir) / "bandstructure.png").is_file()

        assert result.output.startswith("Wrote bandstructure.png (1 series,")
        assert "6.2452" in result.output

    def test_two_folders_share_one_zero(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """Both runs land on one axes, referenced to the first one's edge."""
        from koopmans.cli import cli

        lda = dft_run(tmp_path, "si_lda", 6.2452, [[-5.0, 6.2452], [-4.5, 7.0], [-4.0, 7.5]])
        ki = dft_run(tmp_path, "si_ki", 5.4, [[-6.0, 5.4], [-5.5, 8.0], [-5.0, 8.5]])

        result = runner.invoke(
            cli,
            [
                "plot",
                "bandstructure",
                str(lda),
                str(ki),
                "-o",
                str(tmp_path / "si_bands.png"),
                "--data",
                str(tmp_path / "si_bands.json"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "2 series" in result.output
        assert "'si_lda: DFT'" in result.output
        payload = json.loads((tmp_path / "si_bands.json").read_text())
        # Both records carry the first run's edge, so the second's own
        # 5.4 eV edge is still 0.845 eV below zero on the drawn axes.
        assert [record["zero"] for record in payload["series"]] == [
            pytest.approx(6.2452),
            pytest.approx(6.2452),
        ]

    def test_ylim_clips_the_figure_it_writes(
        self, aiida_profile: Any, runner: Any, drawn_axes: Any, tmp_path: Path
    ) -> None:
        """The range reaches the axes, and the data written stays as computed."""
        from koopmans.cli import cli

        folder = dft_run(tmp_path, "zno", 6.0, [[-130.0, 6.0], [-130.0, 7.0], [-130.0, 7.5]])

        result = runner.invoke(
            cli,
            [
                "plot",
                "bandstructure",
                str(folder),
                "--ylim",
                "-2",
                "8",
                "-o",
                str(tmp_path / "zno.png"),
                "--data",
                str(tmp_path / "zno.json"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "zno.png").is_file()
        assert drawn_axes[-1].get_ylim() == pytest.approx((-2.0, 8.0))
        # Clipping frames the figure; it does not discard the bands behind it.
        payload = json.loads((tmp_path / "zno.json").read_text())
        assert payload["series"][0]["energies"][0][0] == pytest.approx(-130.0)

    def test_a_label_names_the_curve_and_brings_the_legend_back(
        self, aiida_profile: Any, runner: Any, drawn_axes: Any, tmp_path: Path
    ) -> None:
        """Naming one folder's series is asking for it to be named on the figure.

        The rule that a single curve carries no key exists to keep noise off a
        figure nobody asked to annotate; an explicit --label is that asking, and
        without this the label would be input with no effect.
        """
        from koopmans.cli import cli

        folder = dft_run(tmp_path, "zno", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        result = runner.invoke(
            cli,
            [
                "plot",
                "bandstructure",
                str(folder),
                "--label",
                "KI@LDA",
                "-o",
                str(tmp_path / "a.png"),
            ],
        )

        assert result.exit_code == 0, result.output
        legend = drawn_axes[-1].get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["KI@LDA"]

    def test_one_folder_unnamed_still_carries_no_legend(
        self, aiida_profile: Any, runner: Any, drawn_axes: Any, tmp_path: Path
    ) -> None:
        """The control: the key comes back for the label, not for the option's sake."""
        from koopmans.cli import cli

        folder = dft_run(tmp_path, "zno", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        result = runner.invoke(
            cli, ["plot", "bandstructure", str(folder), "-o", str(tmp_path / "a.png")]
        )

        assert result.exit_code == 0, result.output
        assert drawn_axes[-1].get_legend() is None

    def test_a_label_after_the_second_folder_pairs_with_it(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """A lone --label names the folder it followed, not every folder.

        Only the second folder is followed by --label here, so it takes the
        given name and the first keeps its default folder-prefixed one:
        fewer --label values than folders is not an error, it names only the
        folders that were followed by one.
        """
        from koopmans.cli import cli

        first = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        second = dft_run(tmp_path, "si_ki", 5.4, [[-6.0, 5.4], [-5.5, 8.0], [-5.0, 8.5]])

        result = runner.invoke(
            cli,
            [
                "plot",
                "bandstructure",
                str(first),
                str(second),
                "--label",
                "DFT",
                "-o",
                str(tmp_path / "si.png"),
                "--data",
                str(tmp_path / "si.json"),
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads((tmp_path / "si.json").read_text())
        assert [record["label"] for record in payload["series"]] == ["si_lda: DFT", "DFT"]

    def test_styles_draw_the_folders_in_the_order_they_are_listed(
        self, aiida_profile: Any, runner: Any, drawn_axes: Any, tmp_path: Path
    ) -> None:
        """Crosses for the computed bands, a line for the interpolated ones.

        The two folders are drawn differently and the right way round, which
        the marker of each folder's own curves shows; the data file records
        what each was drawn in, so the figure can be redrawn from it.
        """
        from koopmans.cli import cli

        pw = dft_run(tmp_path, "pw", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        wannier = dft_run(tmp_path, "wannier", 6.0, [[-5.1, 6.1], [-4.6, 7.1], [-4.1, 7.6]])

        result = runner.invoke(
            cli,
            [
                "plot",
                "bandstructure",
                str(pw),
                "--style",
                "x",
                str(wannier),
                "--style",
                "-",
                "-o",
                str(tmp_path / "si.png"),
                "--data",
                str(tmp_path / "si.json"),
            ],
        )

        assert result.exit_code == 0, result.output
        crosses, line = band_lines(drawn_axes[-1])[:2], band_lines(drawn_axes[-1])[2:]
        assert {item.get_marker() for item in crosses} == {"x"}
        assert {item.get_marker() for item in line} == {"None"}
        assert {item.get_linestyle() for item in line} == {"-"}
        payload = json.loads((tmp_path / "si.json").read_text())
        assert [record["style"] for record in payload["series"]] == ["x", "-"]

    def test_a_style_written_beside_its_folder_pairs_with_it(
        self, aiida_profile: Any, runner: Any, drawn_axes: Any, tmp_path: Path
    ) -> None:
        """Interleaving folders and styles pairs them as written.

        The help tells the reader to write each style beside its own folder;
        this is what makes that literally true, rather than true only when
        every folder happens to get one. The styles are different so that the
        wrong pairing draws a different figure rather than the same one.
        """
        from koopmans.cli import cli

        pw = dft_run(tmp_path, "pw", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        wannier = dft_run(tmp_path, "wannier", 6.0, [[-5.1, 6.1], [-4.6, 7.1], [-4.1, 7.6]])

        result = runner.invoke(
            cli,
            [
                "plot",
                "bandstructure",
                str(pw),
                "--style",
                "rx",
                "--label",
                "pw.x",
                str(wannier),
                "--style",
                "b-",
                "--label",
                "wannier90",
                "-o",
                str(tmp_path / "si.png"),
                "--data",
                str(tmp_path / "si.json"),
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads((tmp_path / "si.json").read_text())
        assert [(record["label"], record["style"]) for record in payload["series"]] == [
            ("pw.x", "rx"),
            ("wannier90", "b-"),
        ]
        crosses, line = band_lines(drawn_axes[-1])[:2], band_lines(drawn_axes[-1])[2:]
        assert {item.get_marker() for item in crosses} == {"x"}
        assert {as_rgba(item.get_color()) for item in line} == {as_rgba("b")}

    def test_a_style_after_the_second_folder_pairs_with_it(
        self, aiida_profile: Any, runner: Any, drawn_axes: Any, tmp_path: Path
    ) -> None:
        """A lone --style stays with the folder it followed, as --label does."""
        from koopmans.cli import cli

        first = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        second = dft_run(tmp_path, "si_ki", 5.4, [[-6.0, 5.4], [-5.5, 8.0], [-5.0, 8.5]])

        result = runner.invoke(
            cli,
            [
                "plot",
                "bandstructure",
                str(first),
                str(second),
                "--style",
                "x",
                "-o",
                str(tmp_path / "si.png"),
                "--data",
                str(tmp_path / "si.json"),
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads((tmp_path / "si.json").read_text())
        assert [record["style"] for record in payload["series"]] == [None, "x"]
        unstyled, styled = band_lines(drawn_axes[-1])[:2], band_lines(drawn_axes[-1])[2:]
        assert {item.get_marker() for item in unstyled} == {"None"}
        assert {item.get_marker() for item in styled} == {"x"}

    def test_mixed_styled_and_unstyled_folders_pair_by_position(
        self, aiida_profile: Any, runner: Any, drawn_axes: Any, tmp_path: Path
    ) -> None:
        """A reference band structure styled apart from a couple of default ones.

        The number of --style values need not match the number of folders:
        each one names the single folder it follows, so a folder with none
        after it is drawn as the figure would draw it on its own, whatever
        order the styled and unstyled folders come in.
        """
        from koopmans.cli import cli

        reference = dft_run(tmp_path, "bands", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        occ = dft_run(tmp_path, "occ", 6.0, [[-5.1, 6.1], [-4.6, 7.1], [-4.1, 7.6]])
        emp = dft_run(tmp_path, "emp", 6.0, [[-5.2, 6.2], [-4.7, 7.2], [-4.2, 7.7]])

        result = runner.invoke(
            cli,
            [
                "plot",
                "bandstructure",
                str(reference),
                "--style",
                "rx",
                str(occ),
                str(emp),
                "-o",
                str(tmp_path / "si.png"),
                "--data",
                str(tmp_path / "si.json"),
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads((tmp_path / "si.json").read_text())
        assert [record["style"] for record in payload["series"]] == ["rx", None, None]
        lines = band_lines(drawn_axes[-1])
        assert {item.get_marker() for item in lines[:2]} == {"x"}
        assert {item.get_marker() for item in lines[2:]} == {"None"}

    def test_style_before_any_folder_is_refused(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """--style up front is refused by name, rather than becoming a global default."""
        from koopmans.cli import cli

        folder = dft_run(tmp_path, "zno", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        result = runner.invoke(cli, ["plot", "bandstructure", "--style", "rx", str(folder)])

        assert result.exit_code == 2
        assert "--style must follow the folder it applies to" in result.output

    def test_label_before_any_folder_is_refused(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """--label is refused the same way --style is, symmetrically."""
        from koopmans.cli import cli

        folder = dft_run(tmp_path, "zno", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        result = runner.invoke(cli, ["plot", "bandstructure", "--label", "DFT", str(folder)])

        assert result.exit_code == 2
        assert "--label must follow the folder it applies to" in result.output

    def test_a_second_style_for_the_same_folder_is_refused(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """A second --style for one folder is refused, not silently overwritten.

        Silently keeping the last one would discard the first with no error —
        the kind of silent drop this codebase refuses everywhere else.
        """
        from koopmans.cli import cli

        folder = dft_run(tmp_path, "zno", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        result = runner.invoke(
            cli, ["plot", "bandstructure", str(folder), "--style", "x", "--style", "rx"]
        )

        assert result.exit_code == 2
        assert "--style was already given for" in result.output
        assert "'x'" in result.output

    def test_a_second_label_for_the_same_folder_is_refused(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """--label is refused a second time for one folder, symmetrically with --style."""
        from koopmans.cli import cli

        folder = dft_run(tmp_path, "zno", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        result = runner.invoke(
            cli, ["plot", "bandstructure", str(folder), "--label", "DFT", "--label", "KI"]
        )

        assert result.exit_code == 2
        assert "--label was already given for" in result.output
        assert "'DFT'" in result.output

    def test_a_double_dash_escapes_a_dash_prefixed_folder(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """-- ends option parsing, so a dash-prefixed folder is still a folder.

        Without honoring click's own "--" escape hatch, a folder spelled with
        a leading dash looks like an unrecognized option to the scanner that
        pairs --style with the folder before it.
        """
        from koopmans.cli import cli

        with runner.isolated_filesystem(temp_dir=tmp_path) as workdir:
            dft_run(Path(workdir), "-oldrun", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

            result = runner.invoke(cli, ["plot", "bandstructure", "--", "-oldrun"])

        assert result.exit_code == 0, result.output
        assert "1 series" in result.output

    def test_a_double_dash_hands_everything_after_it_to_click_as_folders(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """After --, --style is a folder name too, exactly as click's own parser reads it.

        Before honoring --, this raised the misleading "--style must follow
        the folder it applies to" — misleading because a folder (-oldrun) had
        in fact just been given. What's left after fixing that is click's own
        "no such folder" complaint about the two names that are not real
        directories, which is at least an honest error.
        """
        from koopmans.cli import cli

        folder = dft_run(tmp_path, "zno", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        result = runner.invoke(cli, ["plot", "bandstructure", "--", str(folder), "--style", "rx"])

        assert result.exit_code == 2
        assert "--style must follow the folder it applies to" not in result.output

    def test_an_explicit_empty_style_is_distinct_from_no_style(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """--style '' is a value the user typed, not the internal "unset" marker.

        Both draw the same plain curve, since an empty format string tells
        matplotlib nothing different from no format string at all — but the
        dumped record still says what was actually typed rather than
        silently collapsing it into "nothing was asked for here".
        """
        from koopmans.cli import cli

        first = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        second = dft_run(tmp_path, "si_ki", 5.4, [[-6.0, 5.4], [-5.5, 8.0], [-5.0, 8.5]])

        result = runner.invoke(
            cli,
            [
                "plot",
                "bandstructure",
                str(first),
                "--style",
                "",
                str(second),
                "-o",
                str(tmp_path / "si.png"),
                "--data",
                str(tmp_path / "si.json"),
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads((tmp_path / "si.json").read_text())
        assert [record["style"] for record in payload["series"]] == ["", None]

    def test_an_unreadable_style_is_refused_with_what_to_write(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """A style matplotlib cannot read is refused before anything is drawn.

        matplotlib's own complaint names the character it choked on and stops
        there, so the message goes on to say what a format string is made of.
        """
        from koopmans.cli import cli

        folder = dft_run(tmp_path, "zno", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        result = runner.invoke(cli, ["plot", "bandstructure", str(folder), "--style", "dashed"])

        assert result.exit_code == 2
        assert "not a valid format string" in result.output
        assert "'k-' is a black line" in result.output

    def test_a_style_alone_leaves_the_legend_off(
        self, aiida_profile: Any, runner: Any, drawn_axes: Any, tmp_path: Path
    ) -> None:
        """Saying how a curve is drawn is not asking for it to be named.

        A lone --label brings the key back because a name has nowhere else to
        appear; a style shows on the curve itself.
        """
        from koopmans.cli import cli

        folder = dft_run(tmp_path, "zno", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        result = runner.invoke(
            cli,
            ["plot", "bandstructure", str(folder), "--style", "x", "-o", str(tmp_path / "a.png")],
        )

        assert result.exit_code == 0, result.output
        assert drawn_axes[-1].get_legend() is None
        assert {line.get_marker() for line in band_lines(drawn_axes[-1])} == {"x"}

    def test_an_inverted_ylim_is_refused(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """MIN above MAX would silently flip the axis upside down."""
        from koopmans.cli import cli

        folder = dft_run(tmp_path, "zno", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        result = runner.invoke(cli, ["plot", "bandstructure", str(folder), "--ylim", "8", "-2"])

        assert result.exit_code == 2
        assert "MIN must be below MAX" in result.output

    def test_not_a_run_directory_is_reported(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """A wrong folder gets a message, not a traceback."""
        from koopmans.cli import cli

        (tmp_path / "elsewhere").mkdir()

        result = runner.invoke(cli, ["plot", "bandstructure", str(tmp_path / "elsewhere")])

        assert result.exit_code == 1
        assert "is not a koopmans run directory" in result.output


# ----------------------------------------------------------------------
# Occupations that are not exactly 0 or 2
# ----------------------------------------------------------------------

#: A three-k-point, three-band semiconductor as a smeared run reports it: a
#: deep valence band, the valence band edge at -0.35 eV, and a conduction band
#: whose occupations are small but not zero. The numbers are Marzari-Vanderbilt
#: cold smearing (QE's ``wgauss(x, -1)``) at ``degauss = 0.27 eV`` with the
#: Fermi level mid-gap, which is why the peak occupation exceeds 2.
SMEARED_ENERGIES = [[-6.0, -0.85, 0.85], [-6.0, -0.6, 0.6], [-6.0, -0.35, 0.35]]
SMEARED_OCCUPATIONS = [
    [2.0, 2.00151, 3.29678e-07],
    [2.0, 2.04821, 0.000184042],
    [2.0, 2.15916, 0.0190239],
]


class TestValenceBandEdge:
    """Which states count as occupied."""

    def test_smeared_tails_are_not_occupied(self, aiida_profile: Any, tmp_path: Path) -> None:
        """The edge is the top of the valence band, not the top of the plot.

        Every conduction state here carries a small positive occupation, so a
        threshold of "greater than zero" would return +0.85 eV — 1.2 eV out,
        and the whole figure with it. Only a threshold relative to the peak
        occupation, which smearing pushes above 2, picks the right state.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="RunPwBands")
        chain = make_process(PW_BANDS, caller=root, link_label="bands")
        attach(
            chain,
            "band_structure",
            make_bands(
                [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]],
                SMEARED_ENERGIES,
                cell=CUBIC,
                labels=[(0, "G"), (2, "X")],
                occupations=SMEARED_OCCUPATIONS,
            ),
        )
        folder = write_run_folder(tmp_path, "si_lda", root)

        found, _ = resolve_band_series([folder])

        assert found[0].vbm == pytest.approx(-0.35)


# ----------------------------------------------------------------------
# Which step owns a band structure
# ----------------------------------------------------------------------


def pw_calculation_run(tmp_path: Path, name: str, computer: orm.Computer, calculation: str) -> Path:
    """Write a folder naming a pw.x calculation that ran ``calculation``.

    What a dumped ``02-bands`` or ``01-scf`` directory holds: the calculation
    itself, its declared namelists, and the eigenvalues it wrote.
    """
    step = make_process(
        PW_CALC,
        calcjob=True,
        computer=computer,
        process_label="PwCalculation",
        inputs={"parameters": orm.Dict({"CONTROL": {"calculation": calculation}})},  # type: ignore[no-untyped-call]
    )
    attach(step, "output_band", make_bands([[0.0, 0.0, 0.0]], [[-5.0]], cell=CUBIC))
    return write_run_folder(tmp_path, name, step)


class TestProducerOwnership:
    """A step that declares a band structure owns everything below it."""

    def test_a_standalone_base_wannierization_is_plotted(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """The Wannier row of the table is reachable on its own."""
        root = make_process("aiida.workflows:workgraph.engine", label="Wannierize")
        base = make_process(W90_BASE, caller=root, link_label="wannier90")
        attach(base, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        folder = write_run_folder(tmp_path, "si_w90", root)

        found, _ = resolve_band_series([folder])

        assert [item.label for item in found] == ["Wannier interpolation"]

    def test_a_dumped_wannier90_calculation_is_plotted(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """A folder naming the calculation alone yields its interpolated bands.

        A dump writes ``aiida_node_metadata.yaml`` for calculations and for
        the run's root, so the calculation is the only wannier90 step a folder
        can name; recognizing the workchain above it and not the calculation
        leaves that folder unplottable.
        """
        calculation = make_process(W90_CALC, calcjob=True, computer=aiida_localhost)
        attach(calculation, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        folder = write_run_folder(tmp_path, "03-wannier90", calculation)

        found, _ = resolve_band_series([folder])

        assert [item.label for item in found] == ["Wannier interpolation"]
        assert found[0].energies == [[-5.0]]

    def test_a_dumped_bands_calculation_is_plotted(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """A pw.x calculation that sampled a path plots on its own.

        The bands step of a dumped run is a calculation directory, so without
        this the computed half of a computed-versus-interpolated figure cannot
        be named at all.
        """
        folder = pw_calculation_run(tmp_path, "02-bands", aiida_localhost, "bands")

        found, _ = resolve_band_series([folder])

        assert [item.label for item in found] == ["DFT"]

    def test_a_dumped_scf_calculation_has_nothing_to_plot(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """The control: an scf calculation publishes the same socket and is not bands.

        ``output_band`` carries the mesh eigenvalues of an scf run, which are
        not a path; only the declared calculation type tells the two apart, and
        without the guard every scf step in a dumped tree would claim a band
        structure.
        """
        folder = pw_calculation_run(tmp_path, "01-scf", aiida_localhost, "scf")

        with pytest.raises(PlottingError, match="No band structure to plot"):
            resolve_band_series([folder])

    def test_a_wannierization_yields_one_series_not_two(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """The workchain returns the bands its calculation created, once.

        Both steps carry the same socket, so a walk that did not stop at the
        first would draw one band structure twice and give the figure a
        legend keying two names to one curve.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="Wannierize")
        base = make_process(W90_BASE, caller=root, link_label="wannier90")
        calculation = make_process(
            W90_CALC,
            caller=base,
            link_label="iteration_01",
            calcjob=True,
            computer=aiida_localhost,
        )
        bands = attach(calculation, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        bands.base.links.add_incoming(
            base, link_type=LinkType.RETURN, link_label="interpolated_bands"
        )
        folder = write_run_folder(tmp_path, "si_w90", root)

        found, _ = resolve_band_series([folder])

        assert [item.label for item in found] == ["Wannier interpolation"]

    def test_a_path_bands_run_is_plotted_as_dft(self, aiida_profile: Any, tmp_path: Path) -> None:
        """A bare pw.x run declaring ``calculation='bands'`` joins the DFT series.

        The wannierize routes run it off their scf density as the explicit
        eigenvalues the Wannier interpolation is judged against; its Fermi
        level is the one ``output_parameters`` reports back from the parent
        density's restart.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="WannierizeBlocks")
        run = make_process(
            PW_BASE,
            caller=root,
            link_label="bands",
            inputs={"pw__parameters": orm.Dict({"CONTROL": {"calculation": "bands"}})},  # type: ignore[no-untyped-call]
        )
        attach(run, "output_band", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        attach(run, "output_parameters", orm.Dict({"fermi_energy": 1.25}))  # type: ignore[no-untyped-call]
        base = make_process(W90_BASE, caller=root, link_label="wannier90")
        attach(base, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.1]]))
        folder = write_run_folder(tmp_path, "si_w90", root)

        found, _ = resolve_band_series([folder])

        assert [item.label for item in found] == ["DFT", "Wannier interpolation"]
        assert found[0].fermi == pytest.approx(1.25)

    def test_a_mesh_run_with_declared_inputs_is_not_plotted(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """An scf run's ``output_band`` stays off the axes even with inputs present.

        Same process type, same socket, same shape of data as the path
        run: only the declared ``calculation`` type separates the mesh
        eigenvalues from the path eigenvalues.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="WannierizeBlocks")
        run = make_process(
            PW_BASE,
            caller=root,
            link_label="scf",
            inputs={"pw__parameters": orm.Dict({"CONTROL": {"calculation": "scf"}})},  # type: ignore[no-untyped-call]
        )
        attach(run, "output_band", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        folder = write_run_folder(tmp_path, "si_scf", root)

        with pytest.raises(PlottingError, match="No band structure to plot"):
            resolve_band_series([folder])

    def test_degenerate_spin_channels_collapse_to_one_series(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """nspin=2 with no declared magnetization draws one channel, not two.

        A quality-check bands run can inherit nspin=2 from a magnetic-capable
        restart density without declaring any magnetic moment of its own; the
        two channels are then identical by construction, so only one belongs
        on the legend. The two channels are not compared numerically — the
        declared inputs are the only evidence used.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="WannierizeBlocks")
        run = make_process(
            PW_BASE,
            caller=root,
            link_label="bands",
            inputs={
                "pw__parameters": orm.Dict(  # type: ignore[no-untyped-call]
                    {
                        "CONTROL": {"calculation": "bands"},
                        "SYSTEM": {"nspin": 2, "starting_magnetization": {"Zn": 0.0, "O": 0.0}},
                    }
                )
            },
        )
        attach(run, "output_band", make_bands([[0.0, 0.0, 0.0]], [[[-5.0, 5.0]], [[-5.0, 5.0]]]))
        folder = write_run_folder(tmp_path, "zno", root)

        found, _ = resolve_band_series([folder])

        assert [item.label for item in found] == ["DFT"]

    def test_a_dumped_calculation_reads_the_same_declared_inputs(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """A calculation directory collapses degenerate channels as its run does.

        The calculation carries its namelists at ``inputs.parameters`` and the
        workchain above it at ``inputs.pw.parameters``. Reading only the latter
        would leave a dumped step drawing two identical channels while the run
        it belongs to draws one.
        """
        calculation = make_process(
            PW_CALC,
            calcjob=True,
            computer=aiida_localhost,
            process_label="PwCalculation",
            inputs={
                "parameters": orm.Dict(  # type: ignore[no-untyped-call]
                    {
                        "CONTROL": {"calculation": "bands"},
                        "SYSTEM": {"nspin": 2, "starting_magnetization": {"Zn": 0.0, "O": 0.0}},
                    }
                )
            },
        )
        attach(
            calculation,
            "output_band",
            make_bands([[0.0, 0.0, 0.0]], [[[-5.0, 5.0]], [[-5.0, 5.0]]]),
        )
        folder = write_run_folder(tmp_path, "02-bands", calculation)

        found, _ = resolve_band_series([folder])

        assert [item.label for item in found] == ["DFT"]

    def test_declared_magnetization_keeps_the_spin_split(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A run that declares a nonzero moment keeps its (up)/(down) split."""
        root = make_process("aiida.workflows:workgraph.engine", label="WannierizeBlocks")
        run = make_process(
            PW_BASE,
            caller=root,
            link_label="bands",
            inputs={
                "pw__parameters": orm.Dict(  # type: ignore[no-untyped-call]
                    {
                        "CONTROL": {"calculation": "bands"},
                        "SYSTEM": {"nspin": 2, "starting_magnetization": {"Fe": 0.5}},
                    }
                )
            },
        )
        attach(run, "output_band", make_bands([[0.0, 0.0, 0.0]], [[[-5.0, 5.0]], [[-4.0, 6.0]]]))
        folder = write_run_folder(tmp_path, "fe", root)

        found, _ = resolve_band_series([folder])

        assert sorted(item.label for item in found) == ["DFT (down)", "DFT (up)"]

    def test_nspin_one_is_unaffected_by_the_collapse_rule(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A plain nspin=1 run stays single-channel; the rule never fires."""
        root = make_process("aiida.workflows:workgraph.engine", label="WannierizeBlocks")
        run = make_process(
            PW_BASE,
            caller=root,
            link_label="bands",
            inputs={
                "pw__parameters": orm.Dict(  # type: ignore[no-untyped-call]
                    {"CONTROL": {"calculation": "bands"}, "SYSTEM": {"nspin": 1}}
                )
            },
        )
        attach(run, "output_band", make_bands([[0.0, 0.0, 0.0]], [[-5.0, 5.0]]))
        folder = write_run_folder(tmp_path, "si", root)

        found, _ = resolve_band_series([folder])

        assert [item.label for item in found] == ["DFT"]

    def test_a_declared_system_of_none_does_not_crash_the_collapse_check(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A malformed ``SYSTEM: None`` is read as no declared inputs, not a crash.

        Not a shape QE itself would ever validate through, but the resolver
        reads the declared inputs of whatever process the profile holds, and
        should not raise for a namelist it cannot make sense of.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="WannierizeBlocks")
        run = make_process(
            PW_BASE,
            caller=root,
            link_label="bands",
            inputs={
                "pw__parameters": orm.Dict(  # type: ignore[no-untyped-call]
                    {"CONTROL": {"calculation": "bands"}, "SYSTEM": None}
                )
            },
        )
        attach(run, "output_band", make_bands([[0.0, 0.0, 0.0]], [[[-5.0, 5.0]], [[-4.0, 6.0]]]))
        folder = write_run_folder(tmp_path, "si", root)

        found, _ = resolve_band_series([folder])

        assert sorted(item.label for item in found) == ["DFT (down)", "DFT (up)"]

    def test_a_non_dict_magnetization_keeps_the_spin_split(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A malformed, non-dict ``starting_magnetization`` keeps both channels.

        Not a shape QE itself would ever validate through, but the resolver
        should not guess at degeneracy when it cannot make sense of the
        declared inputs.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="WannierizeBlocks")
        run = make_process(
            PW_BASE,
            caller=root,
            link_label="bands",
            inputs={
                "pw__parameters": orm.Dict(  # type: ignore[no-untyped-call]
                    {
                        "CONTROL": {"calculation": "bands"},
                        "SYSTEM": {"nspin": 2, "starting_magnetization": [0.0, 0.0]},
                    }
                )
            },
        )
        attach(run, "output_band", make_bands([[0.0, 0.0, 0.0]], [[[-5.0, 5.0]], [[-4.0, 6.0]]]))
        folder = write_run_folder(tmp_path, "si", root)

        found, _ = resolve_band_series([folder])

        assert sorted(item.label for item in found) == ["DFT (down)", "DFT (up)"]

    def test_the_optimize_scan_yields_one_series(self, aiida_profile: Any, tmp_path: Path) -> None:
        """Only the optimize workchain's own output counts, not its trials.

        The scan reruns the plain wannierization once per trial frozen window.
        Descending into them puts every discarded trial on the axes beside the
        result, with nothing on the figure saying which is which.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="Wannierize")
        optimize = make_process(W90_OPTIMIZE, caller=root, link_label="wannier90")
        attach(
            optimize,
            "wannier90_plot__interpolated_bands",
            make_bands([[0.0, 0.0, 0.0]], [[-4.0]]),
        )
        for trial in ("wannier90_1", "wannier90_2", "wannier90_3"):
            base = make_process(W90_BASE, caller=optimize, link_label=trial)
            attach(base, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        folder = write_run_folder(tmp_path, "si_w90", root)

        found, _ = resolve_band_series([folder])

        assert [item.label for item in found] == ["Wannier interpolation"]
        assert found[0].energies == [[-4.0]]


# ----------------------------------------------------------------------
# Naming the folders
# ----------------------------------------------------------------------


class TestFolderLabels:
    """``--label`` names a folder, and every curve that folder contributes."""

    def test_each_label_names_its_own_folder(self, aiida_profile: Any, tmp_path: Path) -> None:
        """Labels pair with the folders positionally, in the order given.

        Asserting the order of the names alone would pass just as well if the
        two were paired the other way round, so each name is checked against
        the bands of the folder it was given for.
        """
        first = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        second = dft_run(tmp_path, "si_ki", 5.4, [[-6.0, 5.4], [-5.5, 8.0], [-5.0, 8.5]])

        found, _ = resolve_band_series([first, second], ("DFT", "KI@LDA"))

        named = {item.label: item for item in found}
        assert list(named) == ["DFT", "KI@LDA"]
        assert named["DFT"].vbm == pytest.approx(6.0)
        assert named["KI@LDA"].vbm == pytest.approx(5.4)

    def test_no_labels_keeps_the_derived_names(self, aiida_profile: Any, tmp_path: Path) -> None:
        """The control: without labels the folder-qualified names stand.

        Distinguishes naming from the prefixing that happens either way, which
        a test only of the labelled case would leave unpinned.
        """
        first = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        second = dft_run(tmp_path, "si_ki", 5.4, [[-6.0, 5.4], [-5.5, 8.0], [-5.0, 8.5]])

        found, _ = resolve_band_series([first, second])

        assert [item.label for item in found] == ["si_lda: DFT", "si_ki: DFT"]

    def test_a_label_replaces_the_folder_prefix(self, aiida_profile: Any, tmp_path: Path) -> None:
        """A named folder is not also prefixed with its own directory name.

        The prefix exists to tell two folders apart; a name the user chose
        already does that, and 'si_lda: DFT' spends the legend twice over.
        """
        first = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        second = dft_run(tmp_path, "si_ki", 5.4, [[-6.0, 5.4], [-5.5, 8.0], [-5.0, 8.5]])

        found, _ = resolve_band_series([first, second], ("DFT", "KI"))

        assert all("si_" not in item.label for item in found)

    def test_a_label_keeps_the_spin_channel(self, aiida_profile: Any, tmp_path: Path) -> None:
        """A collinear folder becomes two curves, and one name covers both.

        The channel is what tells them apart, so the name is applied to it
        rather than in place of it; dropping it would key the legend twice to
        the same text.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="RunPwBands")
        chain = make_process(PW_BANDS, caller=root, link_label="bands")
        attach(
            chain,
            "band_structure",
            make_spin_bands(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                [[[-5.0, 5.0], [-4.0, 6.0]], [[-5.2, 5.2], [-4.2, 6.2]]],
            ),
        )
        folder = write_run_folder(tmp_path, "fe", root)

        found, _ = resolve_band_series([folder], ("Iron",))

        assert [item.label for item in found] == ["Iron (up)", "Iron (down)"]

    def test_a_label_keeps_the_block_qualifier(self, aiida_profile: Any, tmp_path: Path) -> None:
        """A per-block fan-out keeps the step name each curve ran under."""
        root = make_process("aiida.workflows:workgraph.engine", label="Wannierize")
        for block in ("wannierize_occ", "wannierize_emp"):
            step = make_process(W90_BASE, caller=root, link_label=block)
            attach(step, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        folder = write_run_folder(tmp_path, "zno", root)

        found, _ = resolve_band_series([folder], ("Wannier",))

        assert [item.label for item in found] == [
            "Wannier (wannierize_occ)",
            "Wannier (wannierize_emp)",
        ]

    def test_a_label_tells_apart_curves_of_different_series(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """Curves named by their series alone still get one name each.

        A run holding both a pw.x bands step and a kcw.x one needs no
        qualifier to tell its curves apart, because their series names already
        do; one chosen name over both would key the legend twice to the same
        text.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="MixedRun")
        dft = make_process(PW_BANDS, caller=root, link_label="dft_bands")
        attach(dft, "band_structure", make_bands([[0.0, 0.0, 0.0]], [[-5.0]], cell=CUBIC))
        ham = make_process(
            KCW_HAM, caller=root, link_label="ham", calcjob=True, computer=aiida_localhost
        )
        attach(ham, "bands", make_bands([[0.0, 0.0, 0.0]], [[-3.0]], cell=CUBIC))
        folder = write_run_folder(tmp_path, "zno", root)

        plain, _ = resolve_band_series([folder])
        named, _ = resolve_band_series([folder], ("ZnO",))

        assert [item.label for item in plain] == ["DFT", "KI"]
        assert [item.label for item in named] == ["ZnO (DFT)", "ZnO (KI)"]

    def test_a_label_tells_apart_per_channel_series_names(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """The optimize workchain names its channels in the series, not beside it.

        Its per-spin outputs are separate declared producers, so neither is
        disambiguated against the other and both would take the bare name.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="Wannierize")
        optimize = make_process(W90_OPTIMIZE, caller=root, link_label="wannier90")
        for namespace in ("wannier90_optimal_up", "wannier90_optimal_down"):
            attach(
                optimize,
                f"{namespace}__interpolated_bands",
                make_bands([[0.0, 0.0, 0.0]], [[-5.0]]),
            )
        folder = write_run_folder(tmp_path, "fe", root)

        named, _ = resolve_band_series([folder], ("Iron",))

        assert len({item.label for item in named}) == len(named) > 1
        assert [item.label for item in named] == [
            "Iron (Wannier interpolation (up))",
            "Iron (Wannier interpolation (down))",
        ]

    def test_one_curve_takes_the_bare_label(self, aiida_profile: Any, tmp_path: Path) -> None:
        """The control: a folder drawn as one curve is named and nothing more.

        Qualifying a lone curve would put the derived name back on a figure
        whose whole point was to replace it.
        """
        folder = dft_run(tmp_path, "zno", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        found, _ = resolve_band_series([folder], ("ZnO",))

        assert [item.label for item in found] == ["ZnO"]

    def test_fewer_labels_than_folders_is_refused(self, aiida_profile: Any, tmp_path: Path) -> None:
        """Padding the rest with derived names would hide the typo."""
        first = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        second = dft_run(tmp_path, "si_ki", 5.4, [[-6.0, 5.4], [-5.5, 8.0], [-5.0, 8.5]])

        with pytest.raises(ValueError) as caught:
            resolve_band_series([first, second], ("DFT",))

        assert "1 --label value(s) were given for 2 folder(s)" in str(caught.value)

    def test_more_labels_than_folders_is_refused(self, aiida_profile: Any, tmp_path: Path) -> None:
        """The extra name belongs to a folder that was left off the command."""
        folder = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        with pytest.raises(ValueError) as caught:
            resolve_band_series([folder], ("DFT", "KI"))

        assert "2 --label value(s) were given for 1 folder(s)" in str(caught.value)

    def test_the_count_is_checked_before_the_folders_are_read(self, tmp_path: Path) -> None:
        """A miscount is reported without a profile or a run behind the folders.

        The check is on the command line alone, so it does not depend on the
        folders being readable, let alone on their runs being in this profile.
        """
        missing = tmp_path / "not_a_run"
        missing.mkdir()

        with pytest.raises(ValueError, match="--label"):
            resolve_band_series([missing], ("DFT", "KI"))


class TestFolderStyles:
    """``--style`` draws a folder, and every curve that folder contributes."""

    def test_each_style_draws_its_own_folder(self, aiida_profile: Any, tmp_path: Path) -> None:
        """Styles pair with the folders positionally, in the order given.

        Each style is checked against the bands of the folder it was given
        for, which pairing the two the other way round would fail.
        """
        first = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        second = dft_run(tmp_path, "si_ki", 5.4, [[-6.0, 5.4], [-5.5, 8.0], [-5.0, 8.5]])

        found, _ = resolve_band_series([first, second], styles=("x", "k-"))

        styled = {item.style: item for item in found}
        assert list(styled) == ["x", "k-"]
        assert styled["x"].vbm == pytest.approx(6.0)
        assert styled["k-"].vbm == pytest.approx(5.4)

    def test_no_styles_leaves_the_appearance_to_the_figure(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """The control: unstyled folders carry no format string at all."""
        first = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        second = dft_run(tmp_path, "si_ki", 5.4, [[-6.0, 5.4], [-5.5, 8.0], [-5.0, 8.5]])

        found, _ = resolve_band_series([first, second])

        assert [item.style for item in found] == [None, None]

    def test_a_style_covers_every_curve_of_its_folder(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A collinear folder becomes two curves, and one style draws both.

        A style says how a folder is drawn, not which of its curves is which;
        the spin channel is still what tells them apart, and it survives.
        """
        root = make_process("aiida.workflows:workgraph.engine", label="RunPwBands")
        chain = make_process(PW_BANDS, caller=root, link_label="bands")
        attach(
            chain,
            "band_structure",
            make_spin_bands(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                [[[-5.0, 5.0], [-4.0, 6.0]], [[-5.2, 5.2], [-4.2, 6.2]]],
            ),
        )
        folder = write_run_folder(tmp_path, "fe", root)

        found, _ = resolve_band_series([folder], ("Iron",), ("x",))

        assert [item.style for item in found] == ["x", "x"]
        assert [item.label for item in found] == ["Iron (up)", "Iron (down)"]

    def test_fewer_styles_than_folders_is_refused(self, aiida_profile: Any, tmp_path: Path) -> None:
        """Cycling a short list would draw two folders alike without saying so."""
        first = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        second = dft_run(tmp_path, "si_ki", 5.4, [[-6.0, 5.4], [-5.5, 8.0], [-5.0, 8.5]])

        with pytest.raises(ValueError) as caught:
            resolve_band_series([first, second], styles=("x",))

        assert "1 --style value(s) were given for 2 folder(s)" in str(caught.value)

    def test_more_styles_than_folders_is_refused(self, aiida_profile: Any, tmp_path: Path) -> None:
        """The extra style belongs to a folder that was left off the command."""
        folder = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        with pytest.raises(ValueError) as caught:
            resolve_band_series([folder], styles=("x", "k-"))

        assert "2 --style value(s) were given for 1 folder(s)" in str(caught.value)

    def test_styling_and_naming_are_counted_apart(self, aiida_profile: Any, tmp_path: Path) -> None:
        """One name and no styles is not a miscount, and neither is the reverse.

        The two options are independent, so a folder may be named without
        being styled; counting them together would refuse that.
        """
        folder = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])

        named, _ = resolve_band_series([folder], ("DFT",))
        drawn, _ = resolve_band_series([folder], styles=("x",))

        assert [(item.label, item.style) for item in named] == [("DFT", None)]
        assert [(item.label, item.style) for item in drawn] == [("DFT", "x")]


# ----------------------------------------------------------------------
# A folder that carries nothing
# ----------------------------------------------------------------------


def wannierization_without_bands(tmp_path: Path, name: str, computer: orm.Computer) -> Path:
    """Write a folder naming a wannier90 calculation that interpolated nothing.

    What a wannierize run without a k-point path leaves on disk: wannier90
    is never asked to interpolate, so the calculation finishes and publishes
    no ``interpolated_bands``.
    """
    calculation = make_process(
        W90_CALC, calcjob=True, computer=computer, process_label="Wannier90Calculation"
    )
    attach(calculation, "output_parameters", orm.Dict({"number_wfs": 2}))  # type: ignore[no-untyped-call]
    return write_run_folder(tmp_path, name, calculation)


def wannierize_block_tree(tmp_path: Path, name: str, computer: orm.Computer) -> Path:
    """Write a dumped wannierize block folder, laid out as a real dump is.

    The block folder itself carries no metadata — the dump deletes every
    workflow node's — and holds the preprocessing run, which interpolates
    nothing, beside the minimization, which does.
    """
    block = tmp_path / name / "01-wannier90"
    block.mkdir(parents=True)
    wannierization_without_bands(block, "01-wannier90_pp", computer)
    calculation = make_process(
        W90_CALC, calcjob=True, computer=computer, process_label="Wannier90Calculation"
    )
    attach(calculation, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]], cell=CUBIC))
    write_run_folder(block, "03-wannier90", calculation)
    return tmp_path / name


class TestRejectedFolderSuggestions:
    """A directory holding no run of its own says what under it can be plotted."""

    def test_a_step_folder_names_the_calculation_under_it(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """The block folder is refused, and the wannier90 run under it is offered.

        A step folder is the thing a reader is most likely to type, since it
        is what the block is called; the dump leaves it without metadata, so
        the message has to bridge the gap rather than just refuse.
        """
        folder = wannierize_block_tree(tmp_path, "04-wannierize_occ_1", aiida_localhost)

        with pytest.raises(PlottingError) as caught:
            resolve_band_series([folder])

        message = str(caught.value)
        assert "is not a koopmans run directory" in message
        assert "have band structures to plot" in message
        assert str(folder / "01-wannier90" / "03-wannier90") in message
        assert NODE_METADATA_FILE not in message

    def test_the_preprocessing_run_is_not_offered(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """Only directories that hold a band structure are named.

        The block folder holds two wannier90 calculations; the -pp run
        interpolates nothing, and offering it would send the reader to a
        directory that refuses them in turn.
        """
        folder = wannierize_block_tree(tmp_path, "04-wannierize_occ_1", aiida_localhost)

        with pytest.raises(PlottingError) as caught:
            resolve_band_series([folder])

        assert "01-wannier90_pp" not in str(caught.value)

    def test_the_path_it_prints_can_be_plotted(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """The suggestion is taken from the message and passed straight back in.

        A path that is merely printed is worth nothing; this is what makes it
        a suggestion rather than a guess.
        """
        folder = wannierize_block_tree(tmp_path, "04-wannierize_occ_1", aiida_localhost)

        with pytest.raises(PlottingError) as caught:
            resolve_band_series([folder])
        suggested = [
            line.strip()
            for line in str(caught.value).splitlines()
            if line.startswith("  ") and "more." not in line
        ]

        found, _ = resolve_band_series([Path(path) for path in suggested])

        assert [item.label for item in found] == ["Wannier interpolation"]

    def test_a_folder_with_nothing_under_it_says_so(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A directory that is simply not a run gets no list of paths.

        The control for the suggestion: an empty offer would read as a bug,
        and the reader needs to be told what to pass instead.
        """
        (tmp_path / "elsewhere").mkdir()

        with pytest.raises(PlottingError) as caught:
            resolve_band_series([tmp_path / "elsewhere"])

        message = str(caught.value)
        assert "nothing beneath it has a band structure to plot" in message
        assert "calculation directory inside one" in message
        assert NODE_METADATA_FILE not in message

    def test_a_run_root_outranks_its_own_steps(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """The whole run is listed first, not cut off after its own steps.

        Ordering by path alone sorts a root's metadata after every ``NN-step/``
        beneath it, so the one entry that draws everything falls past the cap
        on a run of a dozen steps.
        """
        holder = tmp_path / "runs"
        root = make_process("aiida.workflows:workgraph.engine", label="Wannierize")
        for index in range(SUGGESTION_LIMIT + 2):
            step = make_process(W90_BASE, caller=root, link_label=f"wannierize_{index}")
            attach(step, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
        run = write_run_folder(holder, "zno", root)
        for index in range(SUGGESTION_LIMIT + 2):
            calculation = make_process(
                W90_CALC,
                calcjob=True,
                computer=aiida_localhost,
                process_label="Wannier90Calculation",
            )
            attach(calculation, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
            write_run_folder(run, f"{index:02d}-wannier90", calculation)

        with pytest.raises(PlottingError) as caught:
            resolve_band_series([holder])

        listed = [line.strip() for line in str(caught.value).splitlines() if line.startswith("  ")]
        assert listed[0] == str(run)

    def test_descendants_from_another_profile_are_not_called_empty(
        self, aiida_profile: Any, tmp_path: Path
    ) -> None:
        """A copied dump says its runs are elsewhere, not that it holds nothing.

        Skipping every descendant because this profile does not hold it, then
        reporting that nothing beneath has a band structure, tells the reader
        the opposite of what is wrong — and this is the case the root folder's
        own message exists for.
        """
        holder = tmp_path / "copied"
        step = write_run_folder(holder, "01-bands", None)
        (step / NODE_METADATA_FILE).write_text(
            yaml.dump({"Node data": {"uuid": "00000000-0000-0000-0000-000000000000"}})
        )

        with pytest.raises(PlottingError) as caught:
            resolve_band_series([holder])

        message = str(caught.value)
        assert "not in this AiiDA profile" in message
        assert "nothing beneath it has a band structure" not in message

    def test_a_long_list_is_capped_and_counted(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """A run of many blocks names the first few and counts the rest.

        Thirty paths are not read; the count is what tells the reader that
        the list is a sample rather than the whole of it.
        """
        root = tmp_path / "si"
        root.mkdir()
        for index in range(SUGGESTION_LIMIT + 3):
            calculation = make_process(
                W90_CALC,
                calcjob=True,
                computer=aiida_localhost,
                process_label="Wannier90Calculation",
            )
            attach(calculation, "interpolated_bands", make_bands([[0.0, 0.0, 0.0]], [[-5.0]]))
            write_run_folder(root, f"{index:02d}-wannier90", calculation)

        with pytest.raises(PlottingError) as caught:
            resolve_band_series([root])

        listed = [line for line in str(caught.value).splitlines() if line.startswith("  ")]
        assert len(listed) == SUGGESTION_LIMIT + 1
        assert listed[-1].strip() == "... and 3 more."


class TestEveryFolderContributes:
    """A folder that yields no series is said so, not quietly dropped."""

    def test_an_empty_folder_beside_a_full_one_is_reported(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """Two folders in, one series out is a figure that misrepresents itself.

        The reader is given a legend of one against a command line of two, and
        nothing on the axes says the second run is missing.
        """
        ki = dft_run(tmp_path, "03-ham", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        empty = wannierization_without_bands(tmp_path, "03-wannier90", aiida_localhost)

        with pytest.raises(PlottingError) as caught:
            resolve_band_series([ki, empty])

        message = str(caught.value)
        assert "03-wannier90" in message
        # The folder that did carry a band structure is not blamed for it.
        assert "03-ham" not in message
        assert "kpoints: {path: ...}" in message
        assert "Leave out the folders above" in message

    def test_two_full_folders_raise_nothing(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """The control: the rule fires on emptiness, not on having two folders."""
        first = dft_run(tmp_path, "si_lda", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        second = dft_run(tmp_path, "si_ki", 5.4, [[-6.0, 5.4], [-5.5, 8.0], [-5.0, 8.5]])

        found, _ = resolve_band_series([first, second])

        assert len(found) == 2

    def test_the_only_folder_being_empty_still_reads_as_before(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, tmp_path: Path
    ) -> None:
        """One folder and nothing in it keeps its message, without the aside.

        There is no rest of the figure to draw, so telling the reader to leave
        the folder out would leave them with no command at all.
        """
        empty = wannierization_without_bands(tmp_path, "03-wannier90", aiida_localhost)

        with pytest.raises(PlottingError) as caught:
            resolve_band_series([empty])

        assert "Leave out the folders above" not in str(caught.value)
        # The advice names the one thing to change: give the input a k-path.
        assert "kpoints: {path: ...}" in str(caught.value)
        assert "k-point path" in str(caught.value)

    def test_a_folder_with_no_bands_is_named(
        self, aiida_profile: Any, aiida_localhost: orm.Computer, runner: Any, tmp_path: Path
    ) -> None:
        """The command fails naming the folder rather than drawing one series.

        Reproduces the ZnO tutorial's invocation: the kcw.x run and the
        wannier90 step it was built from, of which only the first has bands.
        """
        from koopmans.cli import cli

        ham = dft_run(tmp_path, "03-ham", 6.0, [[-5.0, 6.0], [-4.5, 7.0], [-4.0, 7.5]])
        empty = wannierization_without_bands(tmp_path, "03-wannier90", aiida_localhost)

        result = runner.invoke(
            cli,
            ["plot", "bandstructure", str(ham), str(empty), "-o", str(tmp_path / "out.png")],
        )

        assert result.exit_code == 1
        assert "03-wannier90" in result.output
        assert not (tmp_path / "out.png").exists()


# ----------------------------------------------------------------------
# Series that do not belong on one axes
# ----------------------------------------------------------------------


class TestPathAgreement:
    """Two runs share an axis only if they share a path."""

    def test_different_paths_are_refused(self) -> None:
        """A figure that looks right and is not is worse than an error."""
        dft = series(
            "DFT",
            kpoints=[[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]],
            path_labels=[(0, "G"), (2, "X")],
        )
        ki = series(
            "KI",
            kpoints=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
            path_labels=[(0, "G"), (2, "L")],
        )

        with pytest.raises(PathMismatchError) as excinfo:
            check_paths_agree([dft, ki])

        message = str(excinfo.value)
        assert "X (0.5, 0, 0)" in message
        assert "L (0.5, 0.5, 0.5)" in message

    def test_the_same_path_at_a_different_density_is_accepted(self) -> None:
        """A run may sample the path more finely than the one beside it."""
        coarse = series("DFT")
        fine = series(
            "KI",
            kpoints=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.0, 0.0], [0.5, 0.0, 0.0]],
            energies=[[-5.0, 5.0]] * 4,
            path_labels=[(0, "G"), (3, "X")],
        )

        check_paths_agree([coarse, fine])

    def test_the_same_point_spelled_differently_is_accepted(self) -> None:
        """Seekpath writes ``GAMMA`` where an input file writes ``G``."""
        check_paths_agree([series("DFT"), series("KI", path_labels=[(0, "GAMMA"), (2, "X")])])

    def test_a_series_without_special_points_is_refused(self) -> None:
        """Nothing says the unlabelled one ran along the same path."""
        with pytest.raises(PathMismatchError, match="no high-symmetry points"):
            check_paths_agree([series("DFT"), series("KI", path_labels=[])])

    def test_one_series_is_always_agreeable(self) -> None:
        """A single band structure has nothing to disagree with."""
        check_paths_agree([series("KI", path_labels=[])])
