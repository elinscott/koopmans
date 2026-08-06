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
    """Return a minimal molecular-water DSCF trajectory (ml mode=train) input dict."""
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
            "mode": "train",
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
            "mode": "test",
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
    """``build_trajectory_workgraph`` fans one DSCF task out per snapshot."""

    def test_builds_one_dscf_task_per_snapshot(
        self,
        aiida_profile_clean: Any,
        tmp_path: Path,
        trajectory_codes: dict[str, Any],
        fake_sg15_pseudo_family: Any,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """A 3-frame xyz produces ``dscf_snapshot_1 .. dscf_snapshot_3``."""
        from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 3)
        koopmans_input = KoopmansInput.model_validate(_trajectory_input_dict(str(xyz)))

        workgraph = build_trajectory_workgraph(koopmans_input, trajectory_codes)

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
        from koopmans.aiida.workflows.dscf import build_singlepoint_workgraph

        xyz = write_multiframe_xyz(tmp_path, 2)
        d = _trajectory_input_dict(str(xyz), task="singlepoint")
        d["ml"] = {}
        koopmans_input = KoopmansInput.model_validate(d)

        with pytest.raises(ValueError, match="trajectory"):
            build_singlepoint_workgraph(koopmans_input, trajectory_codes)

    def test_external_projectors_rejected(
        self,
        tmp_path: Path,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """``atom_proj_ext`` is rejected: the trajectory route never consults it."""
        from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 2)
        d = _trajectory_input_dict(str(xyz))
        d["calculator_parameters"]["pw2wannier90"] = {"atom_proj_ext": True}
        koopmans_input = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match="not wired into the trajectory route"):
            build_trajectory_workgraph(koopmans_input, codes={})


def _wannier_trajectory_input_dict(snapshots: str) -> dict[str, Any]:
    """Return a periodic Wannier-initialised water trajectory input dict.

    Water in a cube: the O ``sp3`` block covers the four occupied bands and
    the H ``s`` block the two lowest empty ones, so the projections span the
    six kcp.x bands.
    """
    d = _trajectory_input_dict(snapshots, init_orbitals="mlwfs")
    d["kpoints"] = {"grid": [1, 1, 1], "offset": [0, 0, 0]}
    d["calculator_parameters"]["pw"] = {"system": {"nbnd": 6}}
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
        from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 2)
        d = _wannier_trajectory_input_dict(str(xyz))
        d["ml"]["descriptor"] = "power_spectrum"
        koopmans_input = KoopmansInput.model_validate(d)

        workgraph = build_trajectory_workgraph(koopmans_input, trajectory_codes)

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
        from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 2)
        koopmans_input = KoopmansInput.model_validate(_wannier_trajectory_input_dict(str(xyz)))

        workgraph = build_trajectory_workgraph(koopmans_input, trajectory_codes)

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
        from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 2)
        d = _trajectory_input_dict(str(xyz))
        d["ml"]["descriptor"] = "power_spectrum"
        koopmans_input = KoopmansInput.model_validate(d)

        with pytest.raises(ValueError, match="init_orbitals"):
            build_trajectory_workgraph(koopmans_input, trajectory_codes)

    def test_basis_settings_reach_the_namelist(self) -> None:
        """The ``ml`` radial-basis settings become decompose namelist keys.

        Without this mapping the power spectra would silently be built on
        the CalcJob's default basis rather than the requested one.
        """
        from koopmans.aiida.workflows.trajectory import _decompose_parameters
        from koopmans.input_file.ml import MLConfig

        ml_config = MLConfig(n_max=6, l_max=5, r_min=1.0, r_max=4.5)

        assert _decompose_parameters(ml_config) == {
            "decompose_n_max": 6,
            "decompose_l_max": 5,
            "decompose_r_min": 1.0,
            "decompose_r_max": 4.5,
        }


def _fitted_model() -> dict[str, Any]:
    """Fit a small, stamped self-Hartree screening model."""
    from aiida_koopmans.workgraphs.ml import helpers as ml_helpers

    return ml_helpers.fit_screening_model(  # type: ignore[no-any-return]
        {
            "descriptors": [[-1.0], [-2.0], [-3.0]],
            "alpha_targets": [0.5, 0.6, 0.7],
            "filled": [True, True, False],
            "labels": ["orb_1", "orb_2", "orb_3"],
        },
        "linear_regression",
        correction="ki",
        init_orbitals="kohn-sham",
    )


class TestPredictMode:
    """``ml: {mode: predict}`` loads the model file and hands the model to every DSCF."""

    @staticmethod
    def _write_model(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
        """Write the fitted model as a model-file JSON."""
        model = _fitted_model()
        path = tmp_path / "model.json"
        path.write_text(json.dumps(model))
        return path, model

    def test_twin_builds_differ_only_in_the_model_injection(
        self,
        aiida_profile_clean: Any,
        tmp_path: Path,
        trajectory_codes: dict[str, Any],
        fake_sg15_pseudo_family: Any,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """Same snapshot twice: ab-initio computes alphas, predict injects the model.

        Build-level twin of the live twin-KI run. The predict build must
        carry the loaded model into the snapshot's DSCF (whose interior
        then swaps the Delta-SCF refinement for the prediction sub-graph —
        that routing is asserted in aiida-koopmans' kcp workgraph tests)
        and must not grow a dataset / fit / score layer.
        """
        from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 1)
        model_path, model = self._write_model(tmp_path)

        d = _trajectory_input_dict(str(xyz))
        d["ml"] = {}
        ab_initio = build_trajectory_workgraph(KoopmansInput.model_validate(d), trajectory_codes)

        d = _trajectory_input_dict(str(xyz))
        d["ml"] = {"mode": "predict", "model_file": str(model_path), "descriptor": "self_hartree"}
        predict = build_trajectory_workgraph(KoopmansInput.model_validate(d), trajectory_codes)

        def _dscf(workgraph: Any) -> Any:
            names = set(workgraph.get_task_names())
            assert "dscf_snapshot_1" in names, names
            return next(t for t in workgraph.tasks if t.name == "dscf_snapshot_1")

        assert _dscf(ab_initio).inputs.ml_model.value is None
        assert _dscf(predict).inputs.ml_model.value == model

        names = set(predict.get_task_names())
        for forbidden in (
            "extract_snapshot_dataset",
            "train_screening_model",
            "evaluate_screening_model",
        ):
            assert not any(forbidden in name for name in names), (forbidden, names)

    def test_predict_without_model_file_raises(
        self,
        tmp_path: Path,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """``mode: predict`` without a model source fails at build."""
        from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 1)
        d = _trajectory_input_dict(str(xyz))
        d["ml"] = {"mode": "predict", "descriptor": "self_hartree"}

        with pytest.raises(ValueError, match="requires a trained model"):
            build_trajectory_workgraph(KoopmansInput.model_validate(d), codes={})

    def test_predict_rejects_power_spectrum(
        self,
        tmp_path: Path,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """``mode: predict`` with the power-spectrum descriptor is an explicit gap."""
        from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 1)
        d = _trajectory_input_dict(str(xyz))
        d["ml"] = {"mode": "predict", "descriptor": "power_spectrum"}

        with pytest.raises(NotImplementedError, match="self_hartree"):
            build_trajectory_workgraph(KoopmansInput.model_validate(d), codes={})

    def test_predict_rejects_alpha_numsteps(
        self,
        tmp_path: Path,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """``alpha_numsteps > 1`` cannot take effect under ``mode: predict``."""
        from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 1)
        d = _trajectory_input_dict(str(xyz), alpha_numsteps=2)
        d["ml"] = {"mode": "predict", "descriptor": "self_hartree"}

        with pytest.raises(ValueError, match="alpha_numsteps cannot take effect"):
            build_trajectory_workgraph(KoopmansInput.model_validate(d), codes={})

    def test_non_trajectory_task_rejects_ml_block(
        self,
        tmp_path: Path,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """A singlepoint carrying an ``ml`` block fails at dispatch.

        Without the guard the full ab-initio graph is built and the model
        silently never consulted (legacy permitted singlepoint
        prediction; that route is not ported).
        """
        from koopmans.aiida.workflows import build_workgraph

        xyz = write_multiframe_xyz(tmp_path, 1)
        d = _trajectory_input_dict(str(xyz), task="singlepoint")
        d["ml"] = {"mode": "predict", "model_file": "model.json", "descriptor": "self_hartree"}

        with pytest.raises(NotImplementedError, match="trajectory task only"):
            build_workgraph(KoopmansInput.model_validate(d))


class TestModelNodeRoute:
    """``ml: {model: <pk-or-uuid>}`` threads the stored Dict node to the graph."""

    @staticmethod
    def _stored_model() -> Any:
        from aiida import orm

        return orm.Dict(dict=_fitted_model()).store()  # type: ignore[no-untyped-call]

    def _build_with_ml(
        self,
        ml_block: dict[str, Any],
        *,
        tmp_path: Path,
        trajectory_codes: dict[str, Any],
        write_multiframe_xyz: Callable[..., Path],
    ) -> Any:
        from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 1)
        d = _trajectory_input_dict(str(xyz))
        d["ml"] = ml_block
        return build_trajectory_workgraph(KoopmansInput.model_validate(d), trajectory_codes)

    @staticmethod
    def _dscf_model_value(workgraph: Any) -> Any:
        return next(t for t in workgraph.tasks if t.name == "dscf_snapshot_1").inputs.ml_model.value

    @pytest.mark.parametrize("identify", ["pk", "uuid"])
    def test_node_route_threads_the_stored_dict(
        self,
        aiida_profile_clean: Any,
        tmp_path: Path,
        trajectory_codes: dict[str, Any],
        fake_sg15_pseudo_family: Any,
        write_multiframe_xyz: Callable[..., Path],
        identify: str,
    ) -> None:
        """The named node itself reaches the DSCF's ml_model input.

        Node identity (not just payload equality) at the graph input is
        the provenance claim: the prediction run consumes the training
        run's stored artifact.
        """
        from aiida import orm

        node = self._stored_model()
        workgraph = self._build_with_ml(
            {
                "mode": "predict",
                "model": node.pk if identify == "pk" else node.uuid,
                "descriptor": "self_hartree",
            },
            tmp_path=tmp_path,
            trajectory_codes=trajectory_codes,
            write_multiframe_xyz=write_multiframe_xyz,
        )
        # The stored node sits on the trajectory graph's own input (a
        # TaggedValue proxy forwards isinstance and attribute access), so
        # the run's provenance links the training artifact; the DSCF
        # sub-graph receives the payload.
        value = workgraph.inputs.ml_model.value
        assert isinstance(value, orm.Dict), type(value)
        assert value.uuid == node.uuid
        assert self._dscf_model_value(workgraph) == node.get_dict()

    def test_routes_deliver_the_same_payload(
        self,
        aiida_profile_clean: Any,
        tmp_path: Path,
        trajectory_codes: dict[str, Any],
        fake_sg15_pseudo_family: Any,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """Node and file routes hand the plugin's stamp guards the same model.

        The guards (``ModelMismatchError``) live plugin-side and read the
        payload; equal payloads across routes is what makes their behavior
        route-independent.
        """
        node = self._stored_model()
        by_node = self._build_with_ml(
            {"mode": "predict", "model": node.pk, "descriptor": "self_hartree"},
            tmp_path=tmp_path,
            trajectory_codes=trajectory_codes,
            write_multiframe_xyz=write_multiframe_xyz,
        )
        model_path = tmp_path / "model.json"
        model_path.write_text(json.dumps(node.get_dict()))
        by_file = self._build_with_ml(
            {"mode": "predict", "model_file": str(model_path), "descriptor": "self_hartree"},
            tmp_path=tmp_path,
            trajectory_codes=trajectory_codes,
            write_multiframe_xyz=write_multiframe_xyz,
        )
        assert self._dscf_model_value(by_node) == self._dscf_model_value(by_file)

    def test_non_dict_node_rejected(
        self,
        aiida_profile_clean: Any,
        tmp_path: Path,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """A node of the wrong type fails naming what ml:model must point at."""
        from aiida import orm

        from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

        wrong = orm.Int(7).store()  # type: ignore[no-untyped-call]
        xyz = write_multiframe_xyz(tmp_path, 1)
        d = _trajectory_input_dict(str(xyz))
        d["ml"] = {"mode": "predict", "model": wrong.pk, "descriptor": "self_hartree"}

        with pytest.raises(ValueError, match="must name the stored trained-model Dict"):
            build_trajectory_workgraph(KoopmansInput.model_validate(d), codes={})

    def test_model_and_model_file_are_exclusive(self) -> None:
        """Naming both model sources fails at schema validation."""
        from koopmans.input_file.ml import MLConfig

        with pytest.raises(ValueError, match="supply exactly one"):
            MLConfig.model_validate({"mode": "predict", "model": 42, "model_file": "model.json"})

    @pytest.mark.parametrize("bad", [True, 42.0], ids=["bool", "float"])
    def test_coercible_model_identifiers_rejected(self, bad: object) -> None:
        """``true`` (PK 1) and ``42.0`` (PK 42) would name a node by accident."""
        from koopmans.input_file.ml import MLConfig

        with pytest.raises(ValueError, match="integer PK or string UUID"):
            MLConfig.model_validate({"mode": "predict", "model": bad})


class TestFrozenWindowThreading:
    """The input file's disentanglement window reaches the Wannier-init route.

    Regression for koopmans#94: the water empty block must build the
    legacy pool-and-freeze shape — bands 5-N read with two Wannier
    functions and ``dis_froz_max`` carried — not an unfrozen
    disentanglement (which moves the folded LUMO ~0.9 eV) and not a
    silent drop of the keyword.
    """

    def _dscf_task(
        self,
        tmp_path: Path,
        trajectory_codes: dict[str, Any],
        write_multiframe_xyz: Callable[..., Path],
        *,
        pw_nbnd: int,
    ) -> Any:
        """Build the water trajectory and return its DSCF task."""
        from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

        xyz = write_multiframe_xyz(tmp_path, 1)
        d = _wannier_trajectory_input_dict(str(xyz))
        d["calculator_parameters"]["pw"] = {"system": {"nbnd": pw_nbnd}}
        wg = build_trajectory_workgraph(KoopmansInput.model_validate(d), trajectory_codes)
        return next(t for t in wg.tasks if t.name == "dscf_snapshot_1")

    def test_water_empty_block_pools_and_freezes(
        self,
        aiida_profile_clean: Any,
        tmp_path: Path,
        trajectory_codes: dict[str, Any],
        installed_wannier_codes: dict[str, Any],
        installed_fold_codes: dict[str, Any],
        fake_sg15_pseudo_family: Any,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """With nscf headroom the empty block reads the pool and keeps the window."""
        dscf = self._dscf_task(tmp_path, trajectory_codes, write_multiframe_xyz, pw_nbnd=12)
        blocks = dscf.inputs.blocks.value
        empty = next(b for b in blocks if not b["filled"])
        assert empty["num_wann"] == 2
        assert empty["num_bands"] == 8
        assert list(empty["exclude_bands"]) == [1, 2, 3, 4]
        overrides = dscf.inputs.wannier_overrides["wannier90"].value
        assert overrides["dis_froz_max"] == 1.0

    def test_water_empty_block_isolated_without_headroom(
        self,
        aiida_profile_clean: Any,
        tmp_path: Path,
        trajectory_codes: dict[str, Any],
        installed_wannier_codes: dict[str, Any],
        installed_fold_codes: dict[str, Any],
        fake_sg15_pseudo_family: Any,
        write_multiframe_xyz: Callable[..., Path],
    ) -> None:
        """Nscf nbnd equal to the Wannier manifold selects the isolated shape."""
        dscf = self._dscf_task(tmp_path, trajectory_codes, write_multiframe_xyz, pw_nbnd=6)
        blocks = dscf.inputs.blocks.value
        empty = next(b for b in blocks if not b["filled"])
        assert empty["num_wann"] == 2
        assert empty["num_bands"] == 2
        assert list(empty["exclude_bands"]) == [1, 2, 3, 4]


class TestPerStepKpointMeshRejected:
    """The trajectory task screens with kcp.x, which runs every step on one mesh."""

    @pytest.mark.parametrize("step", ["scf", "nscf"])
    def test_either_entry_raises_naming_the_grid(self, tmp_path: Path, step: str) -> None:
        """The same rejection as the singlepoint kcp.x route, on the same reasoning.

        The guard runs before any code or pseudopotential is loaded, so it
        needs no profile.
        """
        from koopmans.aiida.workflows.trajectory import build_trajectory_workgraph

        d = _trajectory_input_dict(str(tmp_path / "snapshots.xyz"))
        d["kpoints"] = {"grid": [2, 2, 2], "overrides": {step: {"grid": [4, 4, 4]}}}
        koopmans_input = KoopmansInput.model_validate(d)

        with pytest.raises(ValueError, match=rf"overrides\.{step}.*`kpoints.grid`"):
            build_trajectory_workgraph(koopmans_input, codes={})
