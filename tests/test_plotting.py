"""Tests for ``koopmans plot``: the folder resolver and the renderer.

The resolver tests build process nodes directly rather than running
workflows: what is under test is which producer/socket pairs count as a band
structure, and that is decided by ``process_type`` alone.
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
from plumpy.process_states import ProcessState

from koopmans.aiida.dumping import NODE_METADATA_FILE
from koopmans.plotting import (
    DIVIDER_LABEL,
    BandSeries,
    EnergyZero,
    NoEnergyZeroError,
    PlottingError,
    apply_energy_zero,
    apply_labels,
    describe_energy_zero,
    draw_band_structures,
    path_distances,
    render_band_structures,
    resolve_band_series,
    write_series_json,
)

PW_BANDS = "aiida.workflows:quantumespresso.pw.bands"
PW_BASE = "aiida.workflows:quantumespresso.pw.base"
KCW_HAM = "aiida.calculations:koopmans.kcw_ham"

#: A cubic cell, so that reciprocal-space distances are easy to reason about.
CUBIC = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------


def make_bands(
    kpoints: list[list[float]],
    energies: list[list[float]],
    cell: list[list[float]] | None = None,
    labels: list[tuple[int, str]] | None = None,
    occupations: list[list[float]] | None = None,
) -> orm.BandsData:
    """Return an unstored ``BandsData`` holding the given eigenvalues."""
    bands = orm.BandsData()
    if cell is not None:
        bands.set_cell(cell)  # type: ignore[no-untyped-call]
    bands.set_kpoints(kpoints)  # type: ignore[no-untyped-call]
    bands.set_bands(energies, units="eV", occupations=occupations)  # type: ignore[no-untyped-call]
    if labels is not None:
        bands.labels = labels
    return bands


def make_process(
    process_type: str,
    caller: orm.ProcessNode | None = None,
    link_label: str = "step",
    label: str = "",
    exit_status: int = 0,
    exit_message: str | None = None,
    calcjob: bool = False,
    computer: orm.Computer | None = None,
) -> orm.ProcessNode:
    """Return a stored, finished process node of the given ``process_type``."""
    node: orm.ProcessNode = orm.CalcJobNode() if calcjob else orm.WorkflowNode()
    node.process_type = process_type
    node.label = label
    if calcjob:
        node.computer = computer
        node.set_option("resources", {"num_machines": 1})
    if caller is not None:
        link_type = LinkType.CALL_CALC if calcjob else LinkType.CALL_WORK
        node.base.links.add_incoming(caller, link_type=link_type, link_label=link_label)
    node.store()
    node.set_process_state(ProcessState.FINISHED)
    node.set_exit_status(exit_status)
    if exit_message is not None:
        node.set_exit_message(exit_message)
    return node


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


def blank_axes() -> Any:
    """Return a fresh set of axes on a headless backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots()[1]


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

    def test_labels_are_applied_in_order(self) -> None:
        """``--label`` renames the leading series and leaves the rest alone."""
        first, second = series("DFT"), series("KI")

        apply_labels([first, second], ("LDA",))

        assert (first.label, second.label) == ("LDA", "KI")

    def test_too_many_labels_is_an_error(self) -> None:
        """More labels than series is a typo, not a silent no-op."""
        with pytest.raises(ValueError, match="only 1 band structure"):
            apply_labels([series()], ("one", "two"))

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

    def test_not_a_run_directory(self, aiida_profile: Any, tmp_path: Path) -> None:
        """A folder with no metadata file is named, along with what to pass."""
        folder = tmp_path / "somewhere"
        folder.mkdir()

        with pytest.raises(PlottingError) as excinfo:
            resolve_band_series([folder])

        assert "is not a koopmans run directory" in str(excinfo.value)
        assert "koopmans run" in str(excinfo.value)

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

        render_band_structures([series("DFT")], output_path=target, caption="energies as computed")

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

    def test_not_a_run_directory_is_reported(
        self, aiida_profile: Any, runner: Any, tmp_path: Path
    ) -> None:
        """A wrong folder gets a message, not a traceback."""
        from koopmans.cli import cli

        (tmp_path / "elsewhere").mkdir()

        result = runner.invoke(cli, ["plot", "bandstructure", str(tmp_path / "elsewhere")])

        assert result.exit_code == 1
        assert "is not a koopmans run directory" in result.output
