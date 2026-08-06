"""Tests for the input → AiiDA conversion utilities."""

from __future__ import annotations

from typing import Any

import pytest
from qe_tools import CONSTANTS

from koopmans.aiida.conversion import (
    _calculate_kpoints_along_path,
    _parse_kpoints_path_string,
    atoms_input_to_structure,
    input_to_pw_parameters,
)
from koopmans.input_file import AtomsInput

SI_ALAT_BOHR = 10.2622


class TestAlatAtomicPositions:
    """Positions in ``alat`` units (the schema default) must convert correctly."""

    def test_alat_with_ibrav(self, aiida_profile: Any) -> None:
        """``alat`` positions scale by celldm(1) when the cell comes from ibrav."""
        atoms = AtomsInput.model_validate(
            {
                "cell_parameters": {"ibrav": 2, "celldms": {1: SI_ALAT_BOHR}},
                "atomic_positions": {
                    "positions": [["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25]],
                    "units": "alat",
                },
            }
        )
        structure = atoms_input_to_structure(atoms)
        expected = 0.25 * SI_ALAT_BOHR * CONSTANTS.bohr_to_ang
        assert structure.sites[1].position == pytest.approx((expected,) * 3, rel=1e-10)

    def test_alat_with_explicit_vectors(self, aiida_profile: Any) -> None:
        """Without celldm(1), ``alat`` falls back to |a1| (QE's convention)."""
        a = 2.715
        atoms = AtomsInput.model_validate(
            {
                "cell_parameters": {
                    "vectors": [[0.0, a, a], [a, 0.0, a], [a, a, 0.0]],
                    "units": "ang",
                },
                "atomic_positions": {
                    "positions": [["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25]],
                    "units": "alat",
                },
            }
        )
        structure = atoms_input_to_structure(atoms)
        alat = (2 * a**2) ** 0.5
        assert structure.sites[1].position == pytest.approx((0.25 * alat,) * 3, rel=1e-10)


POINT_COORDS = {
    "GAMMA": [0.0, 0.0, 0.0],
    "X": [0.5, 0.0, 0.5],
    "M": [0.5, 0.5, 0.5],
    "K": [0.375, 0.375, 0.75],
}


class TestKpointsPath:
    """Tests for explicit k-path parsing and sampling."""

    def test_continuous_path_shares_vertices(self) -> None:
        """Adjacent segments of a continuous path share their common vertex."""
        path = _parse_kpoints_path_string("GXG", POINT_COORDS)
        assert path == [("GAMMA", "X"), ("X", "GAMMA")]

        kpoints, labels = _calculate_kpoints_along_path(path, POINT_COORDS, density=10.0)
        label_names = [name for _, name in labels]
        assert label_names == ["GAMMA", "X", "GAMMA"]
        # X appears exactly once in the sampled points
        assert sum(kpt == POINT_COORDS["X"] for kpt in kpoints) == 1

    def test_discontinuous_path_keeps_break_vertex(self) -> None:
        """A comma in the path string is a break: both of its vertices must survive."""
        path = _parse_kpoints_path_string("GX,MK", POINT_COORDS)
        assert path == [("GAMMA", "X"), ("M", "K")]

        kpoints, labels = _calculate_kpoints_along_path(path, POINT_COORDS, density=10.0)
        label_names = [name for _, name in labels]
        assert label_names == ["GAMMA", "X", "M", "K"]
        # M is present as a sampled point, adjacent to X
        m_index = next(i for i, name in labels if name == "M")
        x_index = next(i for i, name in labels if name == "X")
        assert kpoints[m_index] == POINT_COORDS["M"]
        assert m_index == x_index + 1


class TestSeekpathBasisGuard:
    """Special points are re-expressed in the input frame when seekpath re-vectors the cell."""

    def test_revectored_primitive_cell_transforms_points(self, aiida_profile: Any) -> None:
        """A QE ibrav=2 fcc cell gets seekpath's automatic path mapped into its own basis."""
        import numpy as np
        from aiida import orm

        from koopmans.aiida.conversion import kpoints_input_to_kpoints_path
        from koopmans.input_file import GridKpointsInput

        a = 5.43
        cell = np.array([[-1, 0, 1], [0, 1, 1], [-1, 1, 0]]) * a / 2
        structure = orm.StructureData(cell=cell.tolist())
        structure.append_atom(position=(0, 0, 0), symbols="Si")  # type: ignore[no-untyped-call]
        structure.append_atom(  # type: ignore[no-untyped-call]
            position=(-a / 4, a / 4, a / 4), symbols="Si"
        )

        kpoints = GridKpointsInput(grid=(2, 2, 2))
        kpts = kpoints_input_to_kpoints_path(kpoints, structure)
        labels = dict(kpts.labels)
        coords = kpts.get_kpoints()  # type: ignore[no-untyped-call]
        label_names = list(labels.values())
        assert "GAMMA" in label_names
        assert "X" in label_names
        x_index = next(i for i, name in labels.items() if name == "X")
        # X sits on the fcc BZ boundary at Cartesian distance 1/a (in 2*pi
        # units), whichever primitive basis expresses it.
        recip_input = np.linalg.inv(cell).T
        assert np.isclose(np.linalg.norm(coords[x_index] @ recip_input), 1 / a, atol=1e-8)

    def test_explicit_path_uses_cell_bravais_points(self, aiida_profile: Any) -> None:
        """An explicit path resolves against the cell's own Bravais-lattice points."""
        import numpy as np
        from aiida import orm

        from koopmans.aiida.conversion import kpoints_input_to_kpoints_path
        from koopmans.input_file import GridKpointsInput

        a = 5.43
        cell = np.array([[-1, 0, 1], [0, 1, 1], [-1, 1, 0]]) * a / 2
        structure = orm.StructureData(cell=cell.tolist())
        structure.append_atom(position=(0, 0, 0), symbols="Si")  # type: ignore[no-untyped-call]
        structure.append_atom(  # type: ignore[no-untyped-call]
            position=(-a / 4, a / 4, a / 4), symbols="Si"
        )

        kpoints = GridKpointsInput(grid=(2, 2, 2), path="GX")
        kpts = kpoints_input_to_kpoints_path(kpoints, structure)
        labels = dict(kpts.labels)
        coords = kpts.get_kpoints()  # type: ignore[no-untyped-call]
        assert labels[0] == "GAMMA"
        assert np.allclose(coords[0], [0.0, 0.0, 0.0])
        last = max(labels)
        assert labels[last] == "X"
        recip_input = np.linalg.inv(cell).T
        assert np.isclose(np.linalg.norm(coords[last] @ recip_input), 1 / a, atol=1e-8)

    def test_explicit_path_survives_near_symmetric_positions(self, aiida_profile: Any) -> None:
        """Legacy hexagonal labels parse even when positions are only nearly symmetric.

        The ZnO tutorial's ``0.33330`` (vs exactly 1/3) demotes the detected
        symmetry below hexagonal for seekpath, which renames every special
        point; the explicit-path vocabulary must come from the cell shape
        alone so ``"ALMGAHK"`` keeps resolving.
        """
        import numpy as np
        from aiida import orm

        from koopmans.aiida.conversion import kpoints_input_to_kpoints_path
        from koopmans.input_file import GridKpointsInput

        a, c = 3.25, 5.21
        cell = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]
        structure = orm.StructureData(cell=cell)
        for symbol, scaled in (
            ("Zn", (0.33330, 0.66670, 0.5)),
            ("Zn", (0.66670, 0.33330, 0.0)),
            ("O", (0.33330, 0.66670, 0.11725)),
            ("O", (0.66670, 0.33330, 0.61725)),
        ):
            structure.append_atom(  # type: ignore[no-untyped-call]
                position=tuple(np.array(scaled) @ np.array(cell)), symbols=symbol
            )

        kpoints = GridKpointsInput(grid=(4, 4, 4), path="ALMGAHK")
        kpts = kpoints_input_to_kpoints_path(kpoints, structure)
        label_names = [name for _, name in kpts.labels]
        assert label_names == ["A", "L", "M", "GAMMA", "A", "H", "K"]

    def test_supercell_is_rejected(self, aiida_profile: Any) -> None:
        """A conventional (non-primitive) fcc cell cannot host the primitive path."""
        import pytest
        from aiida import orm

        from koopmans.aiida.conversion import kpoints_input_to_kpoints_path
        from koopmans.input_file import GridKpointsInput

        a = 5.43
        structure = orm.StructureData(cell=[[a, 0, 0], [0, a, 0], [0, 0, a]])
        fcc = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
        for tx, ty, tz in fcc:
            for bx, by, bz in [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]:
                structure.append_atom(  # type: ignore[no-untyped-call]
                    position=((tx + bx) * a, (ty + by) * a, (tz + bz) * a), symbols="Si"
                )

        kpoints = GridKpointsInput(grid=(2, 2, 2))
        with pytest.raises(NotImplementedError, match="not a primitive cell"):
            kpoints_input_to_kpoints_path(kpoints, structure)


class TestInputToPwParameters:
    """The shared pw parameter dict carries no calculation type of its own."""

    def test_no_calculation_key(self, aiida_profile: Any) -> None:
        """No ``calculation`` entry: each step owner supplies its own type."""
        from koopmans.input_file import KoopmansInput

        inp = KoopmansInput.model_validate(_pw_input())
        parameters = input_to_pw_parameters(inp)
        assert "calculation" not in parameters.get("CONTROL", {})


class TestCodeParallelizationHelper:
    """``code_parallelization`` maps a per-code config to (options, settings)."""

    def test_ntasks_npool_and_pd(self) -> None:
        """Ntasks → resources; npool → -npool then pd → -pd true on the cmdline."""
        from koopmans.aiida.conversion import code_parallelization
        from koopmans.input_file.parallelization import CodeParallelization

        options, settings = code_parallelization(CodeParallelization(ntasks=8, npool=4, pd=True))
        assert options == {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 8}}
        assert settings == {"cmdline": ["-npool", "4", "-pd", "true"]}

    def test_partial_and_none(self) -> None:
        """Unset fields yield empty halves; ``None`` config yields two empties."""
        from koopmans.aiida.conversion import code_parallelization
        from koopmans.input_file.parallelization import CodeParallelization

        options, settings = code_parallelization(CodeParallelization(npool=2))
        assert options == {}
        assert settings == {"cmdline": ["-npool", "2"]}
        # pd False must not emit a flag (only pd True does).
        assert code_parallelization(CodeParallelization(pd=False)) == ({}, {})
        assert code_parallelization(None) == ({}, {})


class TestParallelizationWiring:
    """The pw parallelization directive threads into the shared pw overrides."""

    def test_npool_lands_in_pw_settings(self, aiida_profile: Any) -> None:
        """With pw.npool set, every override key carries settings.cmdline."""
        from koopmans.aiida.workflows import prepare_common_inputs
        from koopmans.input_file import KoopmansInput

        inp = KoopmansInput.model_validate(_pw_input(parallelization={"pw": {"npool": 4}}))
        _, _, overrides = prepare_common_inputs(inp, ["scf", "bands"])
        for key in ("scf", "bands"):
            assert overrides[key]["pw"]["settings"]["cmdline"] == ["-npool", "4"]

    def test_ntasks_lands_in_pw_metadata_options(self, aiida_profile: Any) -> None:
        """An explicit pw.ntasks entry rides metadata.options.resources."""
        from koopmans.aiida.workflows import prepare_common_inputs
        from koopmans.input_file import KoopmansInput

        inp = KoopmansInput.model_validate(_pw_input(parallelization={"pw": {"ntasks": 8}}))
        _, _, overrides = prepare_common_inputs(inp, ["scf"])
        resources = overrides["scf"]["pw"]["metadata"]["options"]["resources"]
        assert resources == {"num_machines": 1, "num_mpiprocs_per_machine": 8}

    def test_no_parallelization_leaves_pw_clean(self, aiida_profile: Any) -> None:
        """With nothing configured, neither settings nor metadata is injected."""
        from koopmans.aiida.workflows import prepare_common_inputs
        from koopmans.input_file import KoopmansInput

        inp = KoopmansInput.model_validate(_pw_input())
        _, _, overrides = prepare_common_inputs(inp, ["scf"])
        assert "settings" not in overrides["scf"]["pw"]
        assert "metadata" not in overrides["scf"]["pw"]

    def test_survives_get_builder_from_protocol(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """Eager build: the pw overrides reach the CalcJob builder intact.

        Exercises the exact machinery ``RunPwBands`` uses
        (``PwBandsWorkChain.get_builder_from_protocol``), without building a
        WorkGraph — so it runs locally despite the aiida-workgraph skew.
        """
        from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

        from koopmans.aiida.workflows import prepare_common_inputs
        from koopmans.input_file import KoopmansInput

        inp = KoopmansInput.model_validate(
            _pw_input(
                pseudo_library="SG15/1.0/PBE/SR",
                parallelization={"pw": {"ntasks": 8, "npool": 4}},
            )
        )
        structure, _, overrides = prepare_common_inputs(inp, ["scf", "bands"])
        builder = PwBandsWorkChain.get_builder_from_protocol(
            code=installed_pw_code, structure=structure, overrides=overrides
        )
        assert builder.scf.pw.settings.get_dict()["cmdline"] == ["-npool", "4"]
        assert builder.scf.pw.metadata.options["resources"]["num_mpiprocs_per_machine"] == 8
        # The bands step's own calculation type survives the shared pw overrides.
        assert builder.bands.pw.parameters.get_dict()["CONTROL"]["calculation"] == "bands"


class TestDispatcherThreadsParallelization:
    """The dispatcher forwards the per-code mapping to the workgraph builder."""

    def test_mapping_reaches_the_builder(self, aiida_profile: Any, monkeypatch: Any) -> None:
        """A configured block is passed as the graph's ``parallelization`` kwarg."""
        import aiida_koopmans.workgraphs.pw as pw_module

        from koopmans.aiida.workflows import dft as dft_module
        from koopmans.aiida.workflows.dft import build_dft_bands_workgraph
        from koopmans.input_file import KoopmansInput

        captured: dict[str, Any] = {}

        def fake_build(**kwargs: Any) -> str:
            """Capture the builder call's kwargs."""
            captured.update(kwargs)
            return "workgraph"

        # Stub the profile-dependent structure/pseudo setup and the graph build
        # so the test isolates the dispatcher's threading logic.
        monkeypatch.setattr(
            dft_module, "prepare_common_inputs", lambda inp, keys: (None, "fam", {})
        )
        monkeypatch.setattr(pw_module.RunPwBands, "build", staticmethod(fake_build))

        inp = KoopmansInput.model_validate(
            _pw_input(parallelization={"pw": {"npool": 4}, "kcw": {"ntasks": 8}})
        )
        build_dft_bands_workgraph(inp, {"pw": object()})
        assert captured["parallelization"] == {"pw": {"npool": 4}, "kcw": {"ntasks": 8}}

    def test_no_config_passes_none(self, aiida_profile: Any, monkeypatch: Any) -> None:
        """With nothing configured the builder receives ``parallelization=None``."""
        import aiida_koopmans.workgraphs.pw as pw_module

        from koopmans.aiida.workflows import dft as dft_module
        from koopmans.aiida.workflows.dft import build_dft_bands_workgraph
        from koopmans.input_file import KoopmansInput

        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            dft_module, "prepare_common_inputs", lambda inp, keys: (None, "fam", {})
        )
        monkeypatch.setattr(
            pw_module.RunPwBands, "build", staticmethod(lambda **kw: captured.update(kw))
        )

        build_dft_bands_workgraph(KoopmansInput.model_validate(_pw_input()), {"pw": object()})
        assert captured["parallelization"] is None


def _pw_input(
    *,
    pseudo_library: str = "SG15/1.2/PBE/SR",
    parallelization: dict[str, Any] | None = None,
    kpoints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a minimal silicon dft_bands input dict for the wiring tests."""
    d: dict[str, Any] = {
        "workflow": {"task": "dft_bands", "pseudo_library": pseudo_library},
        "atoms": {
            "cell_parameters": {"periodic": True, "ibrav": 2, "celldms": {"1": 10.2622}},
            "atomic_positions": {
                "units": "crystal",
                "positions": [["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25]],
            },
        },
        "kpoints": kpoints or {"grid": [2, 2, 2], "offset": [0, 0, 0]},
        "calculator_parameters": {"ecutwfc": 20.0},
    }
    if parallelization is not None:
        d["parallelization"] = parallelization
    return d


class TestDftBandsScfMesh:
    """The dft_bands scf samples the input file's grid, not a protocol distance."""

    def test_scf_samples_the_input_mesh(
        self, aiida_profile: Any, installed_pw_code: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Left to the protocol the scf would pick its own mesh from a distance.

        The bands step is checked alongside it because it must stay on a
        path: a mesh reaching it would replace the band structure with a
        second ground state.
        """
        from koopmans.aiida.workflows import build_workgraph
        from koopmans.input_file import KoopmansInput

        inp = KoopmansInput.model_validate(_pw_input(pseudo_library="SG15/1.0/PBE/SR"))
        wg = build_workgraph(inp)
        scf = wg.tasks["PwBandsWorkChain"].inputs["scf"]
        assert list(scf["kpoints"].value.get_kpoints_mesh()[0]) == [2, 2, 2]
        assert scf["kpoints_distance"].value is None
        assert wg.tasks["PwBandsWorkChain"].inputs["bands"]["kpoints"].value is None

    def test_the_scf_entry_moves_the_scf_mesh_alone(
        self, aiida_profile: Any, installed_pw_code: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """The scf converges the density on its own mesh, denser than the rest."""
        from koopmans.aiida.workflows import build_workgraph
        from koopmans.input_file import KoopmansInput

        inp = KoopmansInput.model_validate(
            _pw_input(
                pseudo_library="SG15/1.0/PBE/SR",
                kpoints={"grid": [2, 2, 2], "overrides": {"scf": {"grid": [4, 4, 4]}}},
            )
        )
        wg = build_workgraph(inp)
        scf = wg.tasks["PwBandsWorkChain"].inputs["scf"]
        assert list(scf["kpoints"].value.get_kpoints_mesh()[0]) == [4, 4, 4]

    def test_a_grid_spacing_reaches_the_scf_as_a_distance(
        self, aiida_profile: Any, installed_pw_code: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """``grid_spacing`` leaves the dimensions to the cell, not to a protocol.

        The value has to arrive as the workchain's own ``kpoints_distance``,
        with no mesh alongside it: the two inputs exclude each other, and a
        mesh would win.
        """
        from koopmans.aiida.workflows import build_workgraph
        from koopmans.input_file import KoopmansInput

        inp = KoopmansInput.model_validate(
            _pw_input(
                pseudo_library="SG15/1.0/PBE/SR",
                kpoints={"grid": [2, 2, 2], "overrides": {"scf": {"grid_spacing": 0.11}}},
            )
        )
        wg = build_workgraph(inp)
        scf = wg.tasks["PwBandsWorkChain"].inputs["scf"]
        assert scf["kpoints"].value is None
        assert float(scf["kpoints_distance"].value) == pytest.approx(0.11)

    def test_an_nscf_entry_is_rejected(
        self, aiida_profile: Any, installed_pw_code: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """There is no nscf step here for the mesh to reach."""
        from koopmans.aiida.workflows import build_workgraph
        from koopmans.input_file import KoopmansInput

        inp = KoopmansInput.model_validate(
            _pw_input(
                pseudo_library="SG15/1.0/PBE/SR",
                kpoints={"grid": [2, 2, 2], "overrides": {"nscf": {"grid": [4, 4, 4]}}},
            )
        )
        with pytest.raises(ValueError, match=r"overrides\.nscf.*dft_bands"):
            build_workgraph(inp)


class TestStepKpointsMesh:
    """A step's mesh is its own attributes laid over the top-level ones."""

    @pytest.mark.parametrize(
        ("override", "grid", "offset"),
        [
            (None, [2, 2, 2], [0.0, 0.5, 0.0]),
            ({"grid": [4, 4, 4]}, [4, 4, 4], [0.0, 0.5, 0.0]),
            ({"offset": [0.5, 0.5, 0.5]}, [2, 2, 2], [0.5, 0.5, 0.5]),
            ({"grid": [4, 4, 4], "offset": [0.0, 0.0, 0.0]}, [4, 4, 4], [0.0, 0.0, 0.0]),
        ],
    )
    def test_unset_attributes_come_from_the_top_level(
        self,
        aiida_profile: Any,
        override: dict[str, Any] | None,
        grid: list[int],
        offset: list[float],
    ) -> None:
        """An entry states only what differs; the rest is the top-level mesh."""
        from koopmans.aiida.conversion import step_kpoints_mesh
        from koopmans.input_file import GridKpointsInput

        kpoints = GridKpointsInput.model_validate(
            {"grid": [2, 2, 2], "offset": [0.0, 0.5, 0.0], "overrides": {"scf": override}}
        )
        mesh, mesh_offset = step_kpoints_mesh(kpoints, "scf").get_kpoints_mesh()  # type: ignore[no-untyped-call]
        assert [int(x) for x in mesh] == grid
        assert list(mesh_offset) == offset


class TestKpointsOffsetConversion:
    """Every offset the schema admits reaches Quantum ESPRESSO as written."""

    @pytest.mark.parametrize(
        ("offset", "card"),
        [((0.0, 0.0, 0.0), "2 2 2 0 0 0"), ((0.5, 0.5, 0.5), "2 2 2 1 1 1")],
    )
    def test_quantum_espresso_accepts_the_mesh(
        self,
        aiida_profile: Any,
        fake_sg15_pseudo_family: Any,
        offset: tuple[float, float, float],
        card: str,
    ) -> None:
        """Both shifts survive the trip into a ``K_POINTS automatic`` card.

        Driven through aiida-quantumespresso's own card writer rather than
        by asserting our numbers back at ourselves: it rejects any shift
        but 0 or 0.5 outright, and what it accepts it converts into the
        integer flags Quantum ESPRESSO actually reads.
        """
        from aiida import orm
        from aiida_quantumespresso.calculations.pw import PwCalculation

        from koopmans.aiida.conversion import kpoints_input_to_kpoints_mesh
        from koopmans.input_file import GridKpointsInput

        structure = orm.StructureData(
            cell=[[2.7, 2.7, 0.0], [2.7, 0.0, 2.7], [0.0, 2.7, 2.7]], pbc=True
        )
        structure.append_atom(  # type: ignore[no-untyped-call]
            position=(0.0, 0.0, 0.0), symbols="Si", name="Si"
        )

        parameters = orm.Dict(  # type: ignore[no-untyped-call]
            {"CONTROL": {"calculation": "scf"}, "SYSTEM": {"ecutwfc": 20.0}}
        )
        content, _ = PwCalculation._generate_pwcp_inputdata(
            parameters=parameters,
            settings={},
            pseudos=fake_sg15_pseudo_family.get_pseudos(structure=structure),
            structure=structure,
            kpoints=kpoints_input_to_kpoints_mesh(GridKpointsInput(grid=(2, 2, 2), offset=offset)),
        )
        lines = content.splitlines()
        assert lines[lines.index("K_POINTS automatic") + 1].split() == card.split()
