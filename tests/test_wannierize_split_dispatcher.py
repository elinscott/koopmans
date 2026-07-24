"""Dispatcher tests for the automated block-splitting Wannierize route.

Builds real ``WorkGraph`` objects through ``_build_wannierize_split_workgraph``
against a throwaway profile (dummy codes, fake pseudos; nothing runs) and
checks the routing guards, the lenient block derivation, and the built graph
topology.
"""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.aiida.workflows import (
    _build_wannierize_split_workgraph,
    _derive_wannierize_blocks,
)
from koopmans.input_file import KoopmansInput


def _si_split_dict(**workflow_updates: Any) -> dict[str, Any]:
    """Return a silicon wannierize input with block splitting enabled.

    One projection block of sp3 hybrids on both Si sites: 8 Wannier
    functions spanning the 4 occupied and 4 lowest empty bands (fake Si
    z_valence 4, nelec 8) — a block that straddles the occupied/empty
    boundary, i.e. one the detection must split.
    """
    d: dict[str, Any] = {
        "workflow": {
            "task": "wannierize",
            # The cutoffs family fixture: the split builder calls
            # get_builder_from_protocol eagerly at build time, which only
            # accepts SSSP / PseudoDojo / cutoffs families.
            "pseudo_library": "SG15/1.0/PBE/SR",
            "block_wannierization_threshold": 1.5,
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
        "kpoints": {"grid": [2, 2, 2], "offset": [0, 0, 0], "path": "GX"},
        "calculator_parameters": {
            "ecutwfc": 20.0,
            "wannier90": {
                "projections": [
                    [
                        {"site": "Si", "ang_mtm": "sp3"},
                    ]
                ],
            },
        },
    }
    d["workflow"].update(workflow_updates)
    return d


@pytest.fixture
def split_codes(
    installed_pw_code: Any, installed_wannier_codes: Any, localhost_code: Any
) -> dict[str, Any]:
    """Assemble the code dict for the split flow (incl. the julia code)."""
    return {
        "pw": installed_pw_code,
        "wannierjl": localhost_code("wannierjl", "wannierjl.check_neighbors"),
        **installed_wannier_codes,
    }


def _build(d: dict[str, Any], codes: dict[str, Any]) -> Any:
    inp = KoopmansInput.model_validate(d)
    return _build_wannierize_split_workgraph(inp, codes)


class TestBlockDerivation:
    """Unit tests for the lenient block derivation."""

    @pytest.fixture
    def silicon_structure(self, aiida_profile: Any) -> Any:
        """Return a 2-atom periodic silicon ``StructureData``."""
        from aiida.orm import StructureData

        cell = [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
        struct = StructureData(cell=cell, pbc=True)
        struct.append_atom(position=(0.0, 0.0, 0.0), symbols="Si", name="Si")  # type: ignore[no-untyped-call]
        struct.append_atom(position=(1.3575, 1.3575, 1.3575), symbols="Si", name="Si")  # type: ignore[no-untyped-call]
        return struct

    @staticmethod
    def _sp3_block() -> list[Any]:
        from koopmans.input_file import KoopmansInput

        inp = KoopmansInput.model_validate(_si_split_dict())
        return inp.calculator_parameters.wannier90.projections

    def test_straddling_block_is_allowed(self, silicon_structure: Any) -> None:
        """A block spanning occupied and empty bands is not an error here."""
        blocks = _derive_wannierize_blocks(silicon_structure, self._sp3_block(), nbnd=8)
        assert len(blocks) == 1
        assert blocks[0]["num_wann"] == 8
        assert blocks[0]["num_bands"] == 8
        assert blocks[0]["include_bands"] == list(range(1, 9))
        assert blocks[0].get("exclude_bands") is None

    def test_last_block_absorbs_extra_bands(self, silicon_structure: Any) -> None:
        """An nbnd beyond the Wannier count becomes the disentanglement pool."""
        blocks = _derive_wannierize_blocks(silicon_structure, self._sp3_block(), nbnd=12)
        assert blocks[0]["num_wann"] == 8
        assert blocks[0]["num_bands"] == 12
        assert blocks[0]["include_bands"] == list(range(1, 13))
        assert blocks[0].get("exclude_bands") is None

    def test_too_few_bands_raises(self, silicon_structure: Any) -> None:
        """Projections needing more bands than nbnd are an input error."""
        with pytest.raises(ValueError, match="span 8 bands but nbnd = 6"):
            _derive_wannierize_blocks(silicon_structure, self._sp3_block(), nbnd=6)


class TestGuards:
    """Routing guards for the unsupported configurations."""

    def test_collinear_not_implemented(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Collinear spin is not wired into the split flow yet."""
        with pytest.raises(NotImplementedError, match="spin='none'"):
            _build(_si_split_dict(spin="collinear"), split_codes)

    def test_missing_projections_not_implemented(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Automatic projections are a follow-up; explicit ones are required."""
        d = _si_split_dict()
        d["calculator_parameters"]["wannier90"] = {}
        with pytest.raises(NotImplementedError, match="explicit Wannier90 projections"):
            _build(d, split_codes)

    def test_missing_kpath_raises(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """The detection needs a bands run, hence a k-path."""
        d = _si_split_dict()
        d["kpoints"].pop("path")
        with pytest.raises(ValueError, match="k-point path"):
            _build(d, split_codes)


class TestGraphBuild:
    """Built-graph topology and input wiring."""

    def test_topology_and_detection_inputs(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """scf+nscf, bands, detection and one nested per-block graph."""
        wg = _build(_si_split_dict(), split_codes)
        names = [t.name for t in wg.tasks]
        assert names.count("scf_nscf") == 1
        assert names.count("bands") == 1
        assert names.count("detect_band_groups") == 1
        assert "wannierize_split_block_1" in names

        detect_task = wg.tasks["detect_band_groups"]
        # 8 Wannier functions; nelec 8 -> 4 occupied bands; threshold 1.5 eV.
        assert detect_task.inputs["num_bands_total"].value == 8
        assert detect_task.inputs["num_occ_bands"].value == 4
        assert detect_task.inputs["threshold"].value == 1.5

        bands_task = wg.tasks["bands"]
        params = bands_task.inputs["pw"]["parameters"].value.get_dict()
        assert params["CONTROL"]["calculation"] == "bands"
        # The nscf-derived overrides carry the resolved nbnd into the bands
        # run so the detection sees every Wannierised band.
        assert params["SYSTEM"]["nbnd"] == 8

    def test_scf_drops_nbnd(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """The scf override drops nbnd; only the nscf override carries it."""
        wg = _build(_si_split_dict(), split_codes)
        overrides = wg.tasks["scf_nscf"].inputs["overrides"].value
        assert "nbnd" not in overrides["scf"]["pw"]["parameters"].get("SYSTEM", {})
        assert overrides["nscf"]["pw"]["parameters"]["SYSTEM"]["nbnd"] == 8

    def test_parallelization_reaches_the_pw_steps(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """The pw parallelization block threads into the split graph's pw steps."""
        d = _si_split_dict()
        d["parallelization"] = {"pw": {"ntasks": 3, "npool": 2}}
        wg = _build(d, split_codes)

        bands_pw = wg.tasks["bands"].inputs["pw"]
        assert bands_pw["metadata"]["options"]["resources"].value["num_mpiprocs_per_machine"] == 3
        assert bands_pw["settings"].value["cmdline"] == ["-npool", "2"]
        assert wg.tasks["scf_nscf"].inputs["parallelization"].value == {
            "pw": {"ntasks": 3, "npool": 2}
        }
