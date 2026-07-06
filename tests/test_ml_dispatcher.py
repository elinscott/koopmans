"""Tests for the trajectory (machine-learning) stream: snapshots input + ml modes.

Covers the ``SnapshotsInput`` schema arm (the legacy ``atomic_positions:
{snapshots: file.xyz}`` convention from ``read_atoms_dict`` in
``koopmans/workflows/_workflow.py``) and the ``_build_trajectory_workgraph``
dispatcher: multi-snapshot fan-out and the ``ml:predict`` mode. Workgraphs
are built, never run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from koopmans.input_file import (
    AtomicPositionsInput,
    KoopmansInput,
    SnapshotsInput,
)

TWO_FRAME_XYZ = """3
Lattice="14.1738 0.0 0.0 0.0 12.0 0.0 0.0 0.0 12.66" Properties=species:S:1:pos:R:3 pbc="F F F"
O        7.0869   6.0   5.89
O        8.1738   6.0   6.55
O        6.0      6.0   6.55
3
Lattice="14.1738 0.0 0.0 0.0 12.0 0.0 0.0 0.0 12.66" Properties=species:S:1:pos:R:3 pbc="F F F"
O        7.10     6.0   5.90
O        8.20     6.0   6.60
O        6.05     6.0   6.50
"""


@pytest.fixture
def snapshots_xyz(tmp_path: Path) -> Path:
    """Write a two-frame extended-xyz snapshots file."""
    path = tmp_path / "snapshots.xyz"
    path.write_text(TWO_FRAME_XYZ)
    return path


def _trajectory_input_dict(snapshots_file: Path, ml: dict[str, Any] | None = None) -> dict:
    """Return an ozone-like trajectory input dict pointing at ``snapshots_file``."""
    return {
        "workflow": {
            "task": "trajectory",
            "correction": "ki",
            "screening_method": "dscf",
            "init_orbitals": "kohn-sham",
            "alpha_numsteps": 1,
            "pseudo_library": "SG15/1.2/PBE/SR",
        },
        "atoms": {
            "cell_parameters": {
                "vectors": [[14.1738, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.66]],
                "units": "angstrom",
                "periodic": False,
            },
            "atomic_positions": {"snapshots": str(snapshots_file)},
        },
        "calculator_parameters": {
            "ecutwfc": 65.0,
            "nbnd": 10,
            "kcp": {"system": {"ecutrho": 260.0}},
        },
        "ml": ml or {},
    }


class TestSnapshotsInputSchema:
    """Parsing and validation of the ``snapshots`` atomic-positions arm."""

    def test_snapshots_parses_for_trajectory_task(self, snapshots_xyz: Path) -> None:
        """The legacy ``atomic_positions: {snapshots: ...}`` form should parse."""
        inp = KoopmansInput.model_validate(_trajectory_input_dict(snapshots_xyz))
        assert isinstance(inp.atoms.atomic_positions, SnapshotsInput)
        assert inp.atoms.atomic_positions.snapshots == str(snapshots_xyz)

    def test_explicit_positions_still_parse(self, snapshots_xyz: Path) -> None:
        """The plain positions form must keep parsing into ``AtomicPositionsInput``."""
        d = _trajectory_input_dict(snapshots_xyz)
        d["atoms"]["atomic_positions"] = {
            "units": "angstrom",
            "positions": [["O", 7.0869, 6.0, 5.89]],
        }
        inp = KoopmansInput.model_validate(d)
        assert isinstance(inp.atoms.atomic_positions, AtomicPositionsInput)

    def test_snapshots_rejected_for_non_trajectory_task(self, snapshots_xyz: Path) -> None:
        """A snapshots file only makes sense for ``task: trajectory``."""
        d = _trajectory_input_dict(snapshots_xyz)
        d["workflow"]["task"] = "singlepoint"
        with pytest.raises(ValueError, match="only valid for"):
            KoopmansInput.model_validate(d)

    def test_read_frames_returns_one_input_per_frame(self, snapshots_xyz: Path) -> None:
        """Each xyz frame becomes one angstrom-unit ``AtomicPositionsInput``."""
        frames = SnapshotsInput(snapshots=str(snapshots_xyz)).read_frames()
        assert len(frames) == 2
        for frame in frames:
            assert isinstance(frame, AtomicPositionsInput)
            assert frame.units == "ang"
            assert len(frame.positions) == 3
            assert all(entry[0] == "O" for entry in frame.positions)
        assert frames[0].positions[0][1] == pytest.approx(7.0869)
        assert frames[1].positions[0][1] == pytest.approx(7.10)

    def test_read_frames_missing_file_raises(self, tmp_path: Path) -> None:
        """A dangling snapshots path should raise a clear error."""
        missing = tmp_path / "nope.xyz"
        with pytest.raises(ValueError, match="does not exist"):
            SnapshotsInput(snapshots=str(missing)).read_frames()


class TestTrajectoryDispatcher:
    """Build-level checks of ``_build_trajectory_workgraph``."""

    @pytest.fixture
    def kcp_code(self, aiida_profile_clean, aiida_code_installed):
        """Register a ``kcp`` code on a freshly cleaned profile.

        ``aiida_profile_clean`` runs first (fixture-argument order) so a
        leftover ``localhost`` computer from an earlier module (e.g. the
        hyperqueue one registered by ``test_hq_install``) cannot collide
        with the ``aiida_computer_local`` get-or-create underneath
        ``aiida_code_installed``.
        """
        return aiida_code_installed(
            label="kcp",
            default_calc_job_plugin="koopmans.kcp",
            filepath_executable="/bin/true",
        )

    @staticmethod
    def _build(inp: KoopmansInput, kcp_code):
        from koopmans.aiida.workflows import _build_trajectory_workgraph

        return _build_trajectory_workgraph(inp, {"kcp": kcp_code})

    @staticmethod
    def _all_task_names(wg) -> list[str]:
        names: list[str] = []

        def _walk(tasks):
            for t in tasks:
                names.append(t.name)
                children = getattr(t, "children", None)
                if children:
                    _walk(children)

        _walk(wg.tasks)
        return names

    @pytest.fixture
    def model_file(self, tmp_path: Path) -> Path:
        """Write a minimal trained screening model (the ml:train JSON output)."""
        from aiida_koopmans import ml_helpers

        model = ml_helpers.fit_screening_model(
            {
                "descriptors": [[-1.0], [-2.0]],
                "alphas": [0.6, 0.7],
                "filled": [True, False],
                "labels": ["orb_1", "orb_2"],
            },
            "linear_regression",
        )
        path = tmp_path / "model.json"
        path.write_text(json.dumps(model))
        return path

    def test_multi_snapshot_fan_out(
        self, snapshots_xyz: Path, kcp_code, fake_sg15_pseudo_family
    ) -> None:
        """A two-frame snapshots file fans out into two DSCF sub-graphs."""
        inp = KoopmansInput.model_validate(
            _trajectory_input_dict(snapshots_xyz, ml={"descriptor": "self_hartree"})
        )
        wg = self._build(inp, kcp_code)
        names = self._all_task_names(wg)
        assert any("dscf_snapshot_1" in n for n in names), names
        assert any("dscf_snapshot_2" in n for n in names), names
        assert not any("dscf_snapshot_3" in n for n in names), names

    def test_predict_mode_builds(
        self,
        snapshots_xyz: Path,
        model_file: Path,
        kcp_code,
        fake_sg15_pseudo_family,
    ) -> None:
        """ml:predict with a model file builds a per-snapshot predict graph."""
        inp = KoopmansInput.model_validate(
            _trajectory_input_dict(
                snapshots_xyz,
                ml={
                    "predict": True,
                    "model_file": str(model_file),
                    "descriptor": "self_hartree",
                },
            )
        )
        wg = self._build(inp, kcp_code)
        names = self._all_task_names(wg)
        assert any("dscf_snapshot_1" in n for n in names), names
        assert any("dscf_snapshot_2" in n for n in names), names
        # Prediction happens inside each DSCF; no trajectory-level
        # dataset/fit/evaluate layer in predict mode.
        for forbidden in (
            "extract_snapshot_dataset",
            "train_screening_model",
            "evaluate_screening_model",
        ):
            assert not any(forbidden in n for n in names), (forbidden, names)

    def test_predict_without_model_file_raises(
        self, snapshots_xyz: Path, kcp_code, fake_sg15_pseudo_family
    ) -> None:
        """ml:predict without ml:model_file should fail fast."""
        inp = KoopmansInput.model_validate(
            _trajectory_input_dict(
                snapshots_xyz, ml={"predict": True, "descriptor": "self_hartree"}
            )
        )
        with pytest.raises(ValueError, match="ml:predict requires ml:model_file"):
            self._build(inp, kcp_code)

    def test_orbital_density_descriptor_raises(
        self, snapshots_xyz: Path, model_file: Path, kcp_code, fake_sg15_pseudo_family
    ) -> None:
        """The power-spectrum descriptor is still gated behind kcp.x retrieval."""
        inp = KoopmansInput.model_validate(
            _trajectory_input_dict(
                snapshots_xyz,
                ml={"predict": True, "model_file": str(model_file)},
            )
        )
        with pytest.raises(NotImplementedError, match="orbital_density"):
            self._build(inp, kcp_code)
