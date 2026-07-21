"""Tests for the input → AiiDA conversion utilities."""

from __future__ import annotations

from typing import Any

import pytest
from qe_tools import CONSTANTS

from koopmans.aiida.conversion import (
    _calculate_kpoints_along_path,
    _parse_kpoints_path_string,
    atoms_input_to_structure,
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


class TestNpoolCmdline:
    """workflow.npool lands as -npool on the pw.x command line."""

    def test_npool_lands_in_pw_settings(self, aiida_profile: Any) -> None:
        """With npool set, every override key carries settings.cmdline."""
        from koopmans.aiida.workflows import _prepare_common_inputs
        from koopmans.input_file import KoopmansInput

        inp = KoopmansInput.model_validate(_NPOOL_INPUT)
        _, _, overrides = _prepare_common_inputs(inp, ["scf", "bands"])
        for key in ("scf", "bands"):
            assert overrides[key]["pw"]["settings"]["cmdline"] == ["-npool", "4"]

    def test_no_npool_leaves_settings_absent(self, aiida_profile: Any) -> None:
        """With npool unset, no settings key is injected."""
        from koopmans.aiida.workflows import _prepare_common_inputs
        from koopmans.input_file import KoopmansInput

        d = dict(_NPOOL_INPUT)
        d["workflow"] = {k: v for k, v in _NPOOL_INPUT["workflow"].items() if k != "npool"}
        inp = KoopmansInput.model_validate(d)
        _, _, overrides = _prepare_common_inputs(inp, ["scf"])
        assert "settings" not in overrides["scf"]["pw"]


_NPOOL_INPUT: dict[str, Any] = {
    "workflow": {
        "task": "dft_bands",
        "pseudo_library": "SG15/1.2/PBE/SR",
        "npool": 4,
    },
    "atoms": {
        "cell_parameters": {
            "periodic": True,
            "ibrav": 2,
            "celldms": {"1": 10.2622},
        },
        "atomic_positions": {
            "units": "crystal",
            "positions": [["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25]],
        },
    },
    "kpoints": {"grid": [2, 2, 2], "offset": [0, 0, 0]},
    "calculator_parameters": {"ecutwfc": 20.0},
}
