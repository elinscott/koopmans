"""Tests for multi-snapshot trajectory input (schema, conversion, dispatch).

Covers the ``atoms.snapshots`` field (mutually exclusive with explicit
``atomic_positions``), path resolution relative to the input file, the
per-frame ``StructureData`` conversion (including the cell-override rule),
rejection of a snapshots input by non-trajectory tasks, and a real
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
from koopmans.input_file.atomic_positions import AtomicPositionsInput


def _snapshots_atoms_dict(snapshots: str, *, box: float = 6.0) -> dict[str, Any]:
    """Return an ``atoms`` block whose positions come from a snapshots xyz."""
    return {
        "cell_parameters": {
            "periodic": True,
            "units": "angstrom",
            "vectors": [[box, 0.0, 0.0], [0.0, box, 0.0], [0.0, 0.0, box]],
        },
        "snapshots": snapshots,
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


class TestAtomsSnapshotsField:
    """``atoms.snapshots`` and ``atoms.atomic_positions`` are mutually exclusive."""

    def test_explicit_positions_parse(self) -> None:
        """An explicit-positions block parses; ``snapshots`` stays unset."""
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
        assert atoms.snapshots is None

    def test_snapshots_parses(self) -> None:
        """A ``snapshots`` path parses; ``atomic_positions`` stays unset."""
        atoms = AtomsInput.model_validate(_snapshots_atoms_dict("frames.xyz"))
        assert atoms.snapshots == "frames.xyz"
        assert atoms.atomic_positions is None

    def test_both_sources_rejected(self) -> None:
        """Supplying both ``atomic_positions`` and ``snapshots`` raises."""
        with pytest.raises(ValueError, match="exactly one"):
            AtomsInput.model_validate(
                {
                    "cell_parameters": {"ibrav": 2, "celldms": {1: 10.2622}},
                    "atomic_positions": {
                        "units": "crystal",
                        "positions": [["Si", 0.0, 0.0, 0.0]],
                    },
                    "snapshots": "frames.xyz",
                }
            )

    def test_neither_source_rejected(self) -> None:
        """An ``atoms`` block with neither positions source raises."""
        with pytest.raises(ValueError, match="exactly one"):
            AtomsInput.model_validate({"cell_parameters": {"ibrav": 2, "celldms": {1: 10.2622}}})

    @pytest.mark.parametrize("path", ["", "   "])
    def test_blank_snapshots_rejected(self, path: str) -> None:
        """An empty or whitespace-only ``snapshots`` path raises at validation."""
        with pytest.raises(ValueError, match="non-empty"):
            AtomsInput.model_validate(_snapshots_atoms_dict(path))

    def test_nested_snapshots_rejected(self) -> None:
        """The retired ``atomic_positions: {"snapshots": ...}`` nesting raises."""
        with pytest.raises(ValueError):
            AtomsInput.model_validate(
                {
                    "cell_parameters": {"ibrav": 2, "celldms": {1: 10.2622}},
                    "atomic_positions": {"snapshots": "frames.xyz"},
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
        assert koopmans_input.atoms.snapshots == str(sub / "frames.xyz")

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
        assert koopmans_input.atoms.snapshots == absolute


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

    def test_explicit_positions_rejected_by_plural_converter(self, aiida_profile: Any) -> None:
        """The plural converter rejects explicit positions."""
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

    def test_snapshots_rejected_by_singular_converter(self, aiida_profile: Any) -> None:
        """The singular converter rejects a snapshots input, naming the trajectory gap."""
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

    def test_external_projectors_rejected(
        self,
        tmp_path: Path,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """``atom_proj_ext`` is rejected: the trajectory route never consults it."""
        from koopmans.aiida.workflows import _build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 2)
        d = _trajectory_input_dict(str(xyz))
        d["calculator_parameters"]["pw2wannier90"] = {"atom_proj_ext": True}
        koopmans_input = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match="not wired into the trajectory route"):
            _build_trajectory_workgraph(koopmans_input, codes={})


def _wannier_trajectory_input_dict(snapshots: str) -> dict[str, Any]:
    """Return a periodic Wannier-initialised water trajectory input dict.

    Water in a cube: the O ``sp3`` block covers the four occupied bands and
    the H ``s`` block the two lowest empty ones, so the projections span the
    six kcp.x bands.
    """
    d = _trajectory_input_dict(snapshots, init_orbitals="mlwfs")
    d["kpoints"] = {"grid": [1, 1, 1], "offset": [0, 0, 0]}
    d["calculator_parameters"]["pw"] = {"system": {"nbnd": 12}}
    d["calculator_parameters"]["wannier90"] = {
        "projections": [
            [{"site": "O", "ang_mtm": "sp3"}],
            [{"site": "H", "ang_mtm": "s"}],
        ],
        "dis_froz_max": 1.0,
    }
    return d


class TestOrbitalDensityDescriptor:
    """The ``power_spectrum`` descriptor reaches the decompose segment."""

    def test_routes_to_decompose_segment(
        self,
        aiida_profile_clean: Any,
        tmp_path: Path,
        trajectory_codes: dict[str, Any],
        installed_wannier_codes: dict[str, Any],
        installed_fold_codes: dict[str, Any],
        installed_decompose_code: Any,
        fake_sg15_pseudo_family: Any,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """Each snapshot gets a descriptor segment instead of the self-Hartree read.

        The discriminator against a dispatcher that accepts the keyword but
        silently keeps building the self-Hartree dataset.
        """
        from koopmans.aiida.workflows import _build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 2)
        d = _wannier_trajectory_input_dict(str(xyz))
        d["ml"]["descriptor"] = "power_spectrum"
        koopmans_input = KoopmansInput.model_validate(d)

        workgraph = _build_trajectory_workgraph(koopmans_input, trajectory_codes)

        names = set(workgraph.get_task_names())
        assert {"descriptors_snapshot_1", "descriptors_snapshot_2"} <= names, names
        assert not any("extract_snapshot_dataset" in name for name in names), names

    def test_self_hartree_keeps_direct_read(
        self,
        aiida_profile_clean: Any,
        tmp_path: Path,
        trajectory_codes: dict[str, Any],
        installed_wannier_codes: dict[str, Any],
        installed_fold_codes: dict[str, Any],
        fake_sg15_pseudo_family: Any,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """The same Wannier-route input on ``self_hartree`` builds no decompose pass."""
        from koopmans.aiida.workflows import _build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 2)
        koopmans_input = KoopmansInput.model_validate(_wannier_trajectory_input_dict(str(xyz)))

        workgraph = _build_trajectory_workgraph(koopmans_input, trajectory_codes)

        names = set(workgraph.get_task_names())
        assert not any(name.startswith("descriptors_") for name in names), names
        assert any("extract_snapshot_dataset" in name for name in names), names

    def test_molecular_route_rejects_power_spectrum(
        self,
        aiida_profile_clean: Any,
        tmp_path: Path,
        trajectory_codes: dict[str, Any],
        installed_decompose_code: Any,
        fake_sg15_pseudo_family: Any,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """A Kohn-Sham-initialised trajectory cannot feed the decompose pass."""
        from koopmans.aiida.workflows import _build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 2)
        d = _trajectory_input_dict(str(xyz))
        d["ml"]["descriptor"] = "power_spectrum"
        koopmans_input = KoopmansInput.model_validate(d)

        with pytest.raises(ValueError, match="init_orbitals"):
            _build_trajectory_workgraph(koopmans_input, trajectory_codes)

    def test_basis_settings_reach_the_namelist(self) -> None:
        """The ``ml`` radial-basis settings become decompose namelist keys.

        Without this mapping the power spectra would silently be built on
        the CalcJob's default basis rather than the requested one.
        """
        from koopmans.aiida.workflows import _decompose_parameters
        from koopmans.input_file.ml import MLConfig

        ml_config = MLConfig(n_max=6, l_max=5, r_min=1.0, r_max=4.5)

        assert _decompose_parameters(ml_config) == {
            "decompose_n_max": 6,
            "decompose_l_max": 5,
            "decompose_r_min": 1.0,
            "decompose_r_max": 4.5,
        }
