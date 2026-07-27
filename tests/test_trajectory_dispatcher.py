"""Tests for multi-snapshot trajectory input (schema, conversion, dispatch).

Covers the ``atoms.atomic_positions`` union (explicit positions vs a
multi-frame ``snapshots`` xyz), path resolution relative to the input file,
the per-frame ``StructureData`` conversion (including the cell-override rule),
rejection of a snapshots block by non-trajectory tasks, and a real
``WorkGraph`` build asserting one ``dscf_snapshot_N`` task per frame
(throwaway profile, dummy codes, fake pseudos; nothing runs).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from koopmans.input_file import AtomsInput, KoopmansInput, read_input_file
from koopmans.input_file.atomic_positions import (
    AtomicPositionsInput,
    SnapshotPositionsInput,
)


def _snapshots_atoms_dict(snapshots: str, *, box: float = 6.0) -> dict[str, Any]:
    """Return an ``atoms`` block using a snapshots-style ``atomic_positions``."""
    return {
        "cell_parameters": {
            "periodic": True,
            "units": "angstrom",
            "vectors": [[box, 0.0, 0.0], [0.0, box, 0.0], [0.0, 0.0, box]],
        },
        "atomic_positions": {"snapshots": snapshots},
    }


def _trajectory_input_dict(snapshots: str, **workflow_updates: Any) -> dict[str, Any]:
    """Return a minimal molecular-water DSCF trajectory (ml:train) input dict."""
    d: dict[str, Any] = {
        "workflow": {
            "task": "trajectory",
            "correction": "ki",
            "screening_method": "dscf",
            "init_orbitals": "kohn-sham",
            "alpha_numsteps": 1,
            "pseudo_library": "SG15/1.2/PBE/SR",
        },
        "atoms": _snapshots_atoms_dict(snapshots),
        "calculator_parameters": {
            "ecutwfc": 65.0,
            "nbnd": 6,
            "kcp": {"system": {"ecutrho": 260.0}},
        },
        "ml": {
            "train": True,
            "descriptor": "self_hartree",
            "estimator": "ridge_regression",
        },
    }
    d["workflow"].update(workflow_updates)
    return d


class TestAtomicPositionsUnion:
    """The ``atomic_positions`` union discriminates on the ``snapshots`` key."""

    def test_explicit_positions_parse(self) -> None:
        """An explicit-positions block resolves to ``AtomicPositionsInput``."""
        atoms = AtomsInput.model_validate(
            {
                "cell_parameters": {"ibrav": 2, "celldms": {1: 10.2622}},
                "atomic_positions": {
                    "units": "crystal",
                    "positions": [["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25]],
                },
            }
        )
        assert isinstance(atoms.atomic_positions, AtomicPositionsInput)

    def test_snapshots_block_parses(self) -> None:
        """A ``snapshots`` block resolves to ``SnapshotPositionsInput``."""
        atoms = AtomsInput.model_validate(_snapshots_atoms_dict("frames.xyz"))
        assert isinstance(atoms.atomic_positions, SnapshotPositionsInput)
        assert atoms.atomic_positions.snapshots == "frames.xyz"

    def test_legacy_tutorial_shape_parses_verbatim(self) -> None:
        """The legacy ``atoms.atomic_positions.snapshots`` shape parses as-is."""
        atoms = AtomsInput.model_validate(
            {
                "atomic_positions": {"snapshots": "testing_snapshots.xyz"},
                "cell_parameters": {
                    "periodic": True,
                    "units": "angstrom",
                    "vectors": [
                        [6.8929, 0.0, 0.0],
                        [0.0, 6.8929, 0.0],
                        [0.0, 0.0, 6.8929],
                    ],
                },
            }
        )
        assert isinstance(atoms.atomic_positions, SnapshotPositionsInput)
        assert atoms.atomic_positions.snapshots == "testing_snapshots.xyz"

    def test_snapshots_block_rejects_explicit_keys(self) -> None:
        """A ``snapshots`` block cannot also carry ``positions`` (extra forbidden)."""
        with pytest.raises(ValueError):
            AtomsInput.model_validate(
                {
                    "cell_parameters": {"ibrav": 2, "celldms": {1: 10.2622}},
                    "atomic_positions": {
                        "snapshots": "frames.xyz",
                        "positions": [["Si", 0.0, 0.0, 0.0]],
                    },
                }
            )


class TestPathResolution:
    """``snapshots`` and ``ml.model_file`` resolve against the input file's dir."""

    def _write_input(self, directory: Path, model_file: str, snapshots: str) -> Path:
        d = _trajectory_input_dict(snapshots)
        d["workflow"]["task"] = "trajectory"
        d["ml"] = {
            "test": True,
            "model_file": model_file,
            "descriptor": "self_hartree",
            "estimator": "ridge_regression",
        }
        path = directory / "input.json"
        path.write_text(json.dumps(d))
        return path

    def test_relative_paths_resolve_against_input_dir(self, tmp_path: Path) -> None:
        """Relative ``snapshots`` / ``model_file`` join the input file's directory."""
        sub = tmp_path / "run"
        sub.mkdir()
        input_file = self._write_input(sub, "model.json", "frames.xyz")

        koopmans_input = read_input_file(input_file)

        assert koopmans_input.ml.model_file == str(sub / "model.json")
        positions = koopmans_input.atoms.atomic_positions
        assert isinstance(positions, SnapshotPositionsInput)
        assert positions.snapshots == str(sub / "frames.xyz")

    def test_relative_paths_are_independent_of_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resolution uses the file's directory, not the process working dir."""
        sub = tmp_path / "run"
        sub.mkdir()
        input_file = self._write_input(sub, "model.json", "frames.xyz")

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        koopmans_input = read_input_file(input_file)
        assert koopmans_input.ml.model_file == str(sub / "model.json")

    def test_absolute_paths_are_left_untouched(self, tmp_path: Path) -> None:
        """An absolute ``snapshots`` path is not rewritten."""
        sub = tmp_path / "run"
        sub.mkdir()
        absolute = str(tmp_path / "shared" / "frames.xyz")
        input_file = self._write_input(sub, str(tmp_path / "m.json"), absolute)

        koopmans_input = read_input_file(input_file)
        positions = koopmans_input.atoms.atomic_positions
        assert isinstance(positions, SnapshotPositionsInput)
        assert positions.snapshots == absolute


class TestSnapshotConversion:
    """``atoms_input_to_structures`` yields one node per frame with the JSON cell."""

    def test_one_structure_per_frame(
        self,
        aiida_profile: Any,
        tmp_path: Path,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """Every xyz frame becomes a ``snapshot_N`` StructureData node."""
        from koopmans.aiida.conversion import atoms_input_to_structures

        xyz = write_multiframe_xyz(tmp_path, 4)
        atoms = AtomsInput.model_validate(_snapshots_atoms_dict(str(xyz)))

        structures = atoms_input_to_structures(atoms)

        assert list(structures) == [f"snapshot_{i}" for i in range(1, 5)]
        assert all(len(s.sites) == 3 for s in structures.values())

    def test_input_cell_overrides_xyz_cell(
        self,
        aiida_profile: Any,
        tmp_path: Path,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """The input-file cell (6 A) overrides the xyz-embedded cell (5 A) on every frame."""
        from koopmans.aiida.conversion import atoms_input_to_structures

        xyz = write_multiframe_xyz(tmp_path, 3, xyz_cell=5.0)
        atoms = AtomsInput.model_validate(_snapshots_atoms_dict(str(xyz), box=6.0))

        structures = atoms_input_to_structures(atoms)

        for structure in structures.values():
            assert structure.cell[0][0] == pytest.approx(6.0)
            assert structure.cell[1][1] == pytest.approx(6.0)
            assert structure.cell[2][2] == pytest.approx(6.0)
            assert structure.pbc == (True, True, True)

    def test_keys_are_valid_link_labels(
        self,
        aiida_profile: Any,
        tmp_path: Path,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """Snapshot keys match the AiiDA link-label charset ``[A-Za-z0-9_]+``."""
        import re

        from koopmans.aiida.conversion import atoms_input_to_structures

        xyz = write_multiframe_xyz(tmp_path, 2)
        atoms = AtomsInput.model_validate(_snapshots_atoms_dict(str(xyz)))

        structures = atoms_input_to_structures(atoms)
        assert all(re.fullmatch(r"[A-Za-z0-9_]+", key) for key in structures)

    def test_explicit_block_rejected_by_plural_converter(self, aiida_profile: Any) -> None:
        """The plural converter rejects an explicit-positions block."""
        from koopmans.aiida.conversion import atoms_input_to_structures

        atoms = AtomsInput.model_validate(
            {
                "cell_parameters": {"ibrav": 2, "celldms": {1: 10.2622}},
                "atomic_positions": {
                    "units": "crystal",
                    "positions": [["Si", 0.0, 0.0, 0.0]],
                },
            }
        )
        with pytest.raises(ValueError, match="snapshots"):
            atoms_input_to_structures(atoms)

    def test_snapshots_block_rejected_by_singular_converter(self, aiida_profile: Any) -> None:
        """The singular converter rejects a snapshots block, naming the trajectory gap."""
        from koopmans.aiida.conversion import atoms_input_to_structure

        atoms = AtomsInput.model_validate(_snapshots_atoms_dict("frames.xyz"))
        with pytest.raises(ValueError, match="trajectory"):
            atoms_input_to_structure(atoms)


@pytest.fixture
def trajectory_codes(installed_pw_code: Any, installed_kcp_code: Any) -> dict[str, Any]:
    """Return the codes dict the trajectory dispatcher receives (pw + kcp)."""
    return {"pw": installed_pw_code, "kcp": installed_kcp_code}


class TestTrajectoryDispatcher:
    """``_build_trajectory_workgraph`` fans one DSCF task out per snapshot."""

    def test_builds_one_dscf_task_per_snapshot(
        self,
        aiida_profile_clean: Any,
        tmp_path: Path,
        trajectory_codes: dict[str, Any],
        fake_sg15_pseudo_family: Any,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """A 3-frame xyz produces ``dscf_snapshot_1 .. dscf_snapshot_3``."""
        from koopmans.aiida.workflows import _build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 3)
        koopmans_input = KoopmansInput.model_validate(_trajectory_input_dict(str(xyz)))

        workgraph = _build_trajectory_workgraph(koopmans_input, trajectory_codes)

        names = set(workgraph.get_task_names())
        assert {f"dscf_snapshot_{i}" for i in range(1, 4)} <= names

    def test_non_trajectory_task_rejects_snapshots_block(
        self,
        aiida_profile_clean: Any,
        tmp_path: Path,
        trajectory_codes: dict[str, Any],
        fake_sg15_pseudo_family: Any,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """A singlepoint task fed a snapshots block fails with a clear ValueError."""
        from koopmans.aiida.workflows import _build_singlepoint_workgraph

        xyz = write_multiframe_xyz(tmp_path, 2)
        d = _trajectory_input_dict(str(xyz), task="singlepoint")
        d["ml"] = {}
        koopmans_input = KoopmansInput.model_validate(d)

        with pytest.raises(ValueError, match="trajectory"):
            _build_singlepoint_workgraph(koopmans_input, trajectory_codes)
