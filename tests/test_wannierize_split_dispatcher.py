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


@pytest.fixture
def silicon_structure(aiida_profile: Any) -> Any:
    """Return a 2-atom periodic silicon ``StructureData``."""
    from aiida.orm import StructureData

    cell = [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
    struct = StructureData(cell=cell, pbc=True)
    struct.append_atom(position=(0.0, 0.0, 0.0), symbols="Si", name="Si")  # type: ignore[no-untyped-call]
    struct.append_atom(position=(1.3575, 1.3575, 1.3575), symbols="Si", name="Si")  # type: ignore[no-untyped-call]
    return struct


class TestBlockDerivation:
    """Unit tests for the lenient block derivation."""

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

    def test_missing_kpath_raises(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """The detection needs a bands run, hence a k-path."""
        d = _si_split_dict()
        d["kpoints"].pop("path")
        with pytest.raises(ValueError, match="k-point path"):
            _build(d, split_codes)


def _si_auto_dict(**workflow_updates: Any) -> dict[str, Any]:
    """Return the silicon split input without explicit projections.

    The fake Si pseudo carries an s+p valence, so the two-atom cell has 8
    atomic projectors — the automatic block spans bands 1-8 with the
    occupied/empty boundary at band 4 (nelec 8).
    """
    d = _si_split_dict(**workflow_updates)
    d["calculator_parameters"]["wannier90"] = {}
    return d


class TestAutomaticProjections:
    """The atomic-projector route taken when no projections are given."""

    def test_automatic_route_builds(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """One atomic-projector block spans the manifold and splits at runtime.

        ``nbnd`` is optional: this input supplies none anywhere, and the
        projector count alone sizes every step.
        """
        d = _si_auto_dict()
        assert "nbnd" not in d["calculator_parameters"]
        assert "pw" not in d["calculator_parameters"]
        wg = _build(d, split_codes)
        names = [t.name for t in wg.tasks]
        assert names.count("scf_nscf") == 1
        assert names.count("bands") == 1
        assert names.count("detect_band_groups") == 1
        assert "wannierize_split_block_1" in names

        detect_task = wg.tasks["detect_band_groups"]
        # 8 atomic projectors; nelec 8 -> 4 occupied bands; threshold 1.5 eV.
        assert detect_task.inputs["num_bands_total"].value == 8
        assert detect_task.inputs["num_occ_bands"].value == 4
        assert detect_task.inputs["threshold"].value == 1.5

        # The nscf (and the bands run seeded from it) covers every band of
        # the projector manifold.
        overrides = wg.tasks["scf_nscf"].inputs["overrides"].value
        assert overrides["nscf"]["pw"]["parameters"]["SYSTEM"]["nbnd"] == 8

    def test_derived_block_covers_the_projector_manifold_exactly(
        self, aiida_profile_clean: Any, fake_sg15_cutoffs_family: Any, silicon_structure: Any
    ) -> None:
        """The single derived block is pool-free and covers every projector band.

        ``include_bands`` must run over exactly ``1..num_wann`` — a shorter
        list would silently drop Wannier functions from the runtime split —
        and ``num_bands == num_wann`` is the no-pool invariant behind the
        nbnd guards.
        """
        from aiida_wannier90_workflows.common.types import WannierProjectionType

        from koopmans.aiida.conversion import get_pseudos_from_family
        from koopmans.aiida.workflows import _derive_automatic_wannierize_blocks

        pseudos = get_pseudos_from_family(fake_sg15_cutoffs_family.label, silicon_structure)
        blocks, nbnd = _derive_automatic_wannierize_blocks(silicon_structure, pseudos, None, 4)
        [block] = blocks
        assert block["num_wann"] == 8
        assert block["num_bands"] == 8
        assert block["include_bands"] == list(range(1, 9))
        assert block["projection_type"] == WannierProjectionType.ATOMIC_PROJECTORS_QE
        assert block.get("exclude_bands") is None
        assert nbnd == 8

    def test_projectors_short_of_occupied_manifold_raise(
        self, aiida_profile_clean: Any, fake_sg15_cutoffs_family: Any, silicon_structure: Any
    ) -> None:
        """Projectors that cannot span the occupied manifold are rejected."""
        from koopmans.aiida.conversion import get_pseudos_from_family
        from koopmans.aiida.workflows import _derive_automatic_wannierize_blocks

        pseudos = get_pseudos_from_family(fake_sg15_cutoffs_family.label, silicon_structure)
        with pytest.raises(ValueError, match="cannot span the occupied manifold"):
            _derive_automatic_wannierize_blocks(silicon_structure, pseudos, None, 10)

    def test_nbnd_above_projector_count_not_implemented(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """A disentanglement pool above the projector manifold cannot split."""
        d = _si_auto_dict()
        d["calculator_parameters"]["nbnd"] = 12
        with pytest.raises(NotImplementedError, match="disentangle"):
            _build(d, split_codes)

    def test_nbnd_below_projector_count_raises(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Fewer bands than atomic projectors is an input error."""
        d = _si_auto_dict()
        d["calculator_parameters"]["nbnd"] = 6
        with pytest.raises(ValueError, match="smaller than the 8 atomic projectors"):
            _build(d, split_codes)

    def test_fully_relativistic_family_not_implemented(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_fr_cutoffs_family: Any
    ) -> None:
        """A fully relativistic family is rejected before any projector counting."""
        d = _si_auto_dict()
        d["workflow"]["pseudo_library"] = fake_sg15_fr_cutoffs_family.label
        with pytest.raises(NotImplementedError, match="fully relativistic"):
            _build(d, split_codes)


def _si_external_dict(projector_dir: Any, **workflow_updates: Any) -> dict[str, Any]:
    """Return the silicon split input using external projectors.

    No explicit projections; the projector directory's ``Si.dat`` (s + p
    per atom, 8 projectors for the two-atom cell) sizes the manifold.
    """
    d = _si_auto_dict(**workflow_updates)
    d["calculator_parameters"]["pw2wannier90"] = {
        "atom_proj_ext": True,
        "atom_proj_dir": str(projector_dir),
    }
    return d


class TestProjectorFileParsing:
    """The pw2wannier90-format reader behind the external projector sizing."""

    @staticmethod
    def _read(tmp_path: Any, content: str) -> list[int]:
        from koopmans.aiida.workflows import _read_projector_angular_momenta

        projector_file = tmp_path / "X.dat"
        projector_file.write_text(content)
        return _read_projector_angular_momenta(projector_file)

    def test_leading_comments_are_skipped(self, tmp_path: Any) -> None:
        """Lines whose first non-blank character is `#` are skipped, as in QE."""
        assert self._read(tmp_path, "# a\n  # b\n100 3\n0 1 2\n0.0 0.1\n") == [0, 1, 2]

    def test_momenta_may_continue_over_lines(self, tmp_path: Any) -> None:
        """The l values are a list-directed read: they may span several lines."""
        assert self._read(tmp_path, "100 4\n0 1\n1 2\n") == [0, 1, 1, 2]

    def test_surplus_tokens_after_the_momenta_are_ignored(self, tmp_path: Any) -> None:
        """Radial data following the declared l count is not misread as momenta."""
        assert self._read(tmp_path, "100 2\n0 1 99 98\n") == [0, 1]

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            ("# only comments\n", "only comments"),
            ("100\n0 1\n", r"must start with `<ngrid> <nproj>`"),
            ("100 x\n0 1\n", r"must start with `<ngrid> <nproj>`"),
            ("100 0\n\n", "at least one projector"),
            ("100 3\n0 1\n", "lists only 2 angular momenta"),
            ("100 2\n0 q\n", "'q' is not an integer"),
            ("100 2\n0 -1\n", r"negative angular momenta: \[-1\]"),
        ],
    )
    def test_malformed_files_raise(self, tmp_path: Any, content: str, match: str) -> None:
        """Every malformation fails naming the file and the reason."""
        with pytest.raises(ValueError, match=match):
            self._read(tmp_path, content)


class TestExternalProjectors:
    """The external-projector route (`pw2wannier90.atom_proj_ext`)."""

    def test_external_route_builds_and_forwards_projector_inputs(
        self,
        aiida_profile_clean: Any,
        split_codes: Any,
        fake_sg15_cutoffs_family: Any,
        si_external_projector_dir: Any,
    ) -> None:
        """The auto block is sized from the `.dat` files and the inputs thread through.

        Topology matches the pseudo-projector route; additionally the
        nested per-block graph carries the projector directory and the
        synthesized tables (only the whole-block wannierisation consumes
        them).
        """
        from tests.fixtures import si_external_projector_tables

        wg = _build(_si_external_dict(si_external_projector_dir), split_codes)
        names = [t.name for t in wg.tasks]
        assert names.count("scf_nscf") == 1
        assert names.count("detect_band_groups") == 1
        assert "wannierize_split_block_1" in names

        detect_task = wg.tasks["detect_band_groups"]
        # 8 external projectors; nelec 8 -> 4 occupied bands.
        assert detect_task.inputs["num_bands_total"].value == 8
        assert detect_task.inputs["num_occ_bands"].value == 4

        split_task = wg.tasks["wannierize_split_block_1"]
        assert split_task.inputs["external_projectors_path"].value == str(si_external_projector_dir)
        assert split_task.inputs["external_projectors"].value == si_external_projector_tables()

        # The nscf covers exactly the projector manifold.
        overrides = wg.tasks["scf_nscf"].inputs["overrides"].value
        assert overrides["nscf"]["pw"]["parameters"]["SYSTEM"]["nbnd"] == 8

    def test_derived_block_is_external_and_pool_free(
        self,
        aiida_profile_clean: Any,
        fake_sg15_cutoffs_family: Any,
        silicon_structure: Any,
    ) -> None:
        """The derived block carries the external type and the no-pool shape."""
        from aiida_wannier90_workflows.common.types import WannierProjectionType

        from koopmans.aiida.workflows import _derive_external_wannierize_blocks
        from tests.fixtures import si_external_projector_tables

        blocks, nbnd = _derive_external_wannierize_blocks(
            silicon_structure, si_external_projector_tables(), None, 4
        )
        [block] = blocks
        assert block["num_wann"] == 8
        assert block["num_bands"] == 8
        assert block["include_bands"] == list(range(1, 9))
        assert block["projection_type"] == WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL
        assert block.get("exclude_bands") is None
        assert nbnd == 8

    def test_explicit_projections_and_external_projectors_conflict(
        self,
        aiida_profile_clean: Any,
        split_codes: Any,
        fake_sg15_cutoffs_family: Any,
        si_external_projector_dir: Any,
    ) -> None:
        """Explicit projections would silently shadow the external projectors."""
        d = _si_split_dict()
        d["calculator_parameters"]["pw2wannier90"] = {
            "atom_proj_ext": True,
            "atom_proj_dir": str(si_external_projector_dir),
        }
        with pytest.raises(ValueError, match="Drop one of the two"):
            _build(d, split_codes)

    def test_missing_dat_file_raises(
        self,
        aiida_profile_clean: Any,
        split_codes: Any,
        fake_sg15_cutoffs_family: Any,
        si_external_projector_dir: Any,
    ) -> None:
        """A directory without the element's `.dat` file is rejected naming it."""
        (si_external_projector_dir / "Si.dat").unlink()
        with pytest.raises(ValueError, match=r"missing the projector files \['Si.dat'\]"):
            _build(_si_external_dict(si_external_projector_dir), split_codes)

    def test_missing_atom_proj_dir_raises(
        self,
        aiida_profile_clean: Any,
        split_codes: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """`atom_proj_ext` without `atom_proj_dir` is an input error."""
        d = _si_auto_dict()
        d["calculator_parameters"]["pw2wannier90"] = {"atom_proj_ext": True}
        with pytest.raises(ValueError, match="atom_proj_dir"):
            _build(d, split_codes)

    def test_nbnd_above_external_projector_count_not_implemented(
        self,
        aiida_profile_clean: Any,
        split_codes: Any,
        fake_sg15_cutoffs_family: Any,
        si_external_projector_dir: Any,
    ) -> None:
        """The no-pool constraint applies to the external source too."""
        d = _si_external_dict(si_external_projector_dir)
        d["calculator_parameters"]["nbnd"] = 12
        with pytest.raises(NotImplementedError, match="external projector files"):
            _build(d, split_codes)

    def test_plain_route_stages_the_projector_inputs(
        self,
        aiida_profile_clean: Any,
        split_codes: Any,
        fake_sg15_cutoffs_family: Any,
        si_external_projector_dir: Any,
    ) -> None:
        """Without the threshold the plain Wannierize route consumes them too.

        The upstream builder demands the orbital tables alongside the
        directory path, so the plain route must synthesize both; its eager
        build exercises that translation end-to-end down to the
        pw2wannier90 step's staged inputs. No frozen list is ever emitted:
        every external projector is Lowdin-orthonormalized.
        """
        from koopmans.aiida.workflows import _build_wannierize_workgraph
        from koopmans.input_file import KoopmansInput

        d = _si_external_dict(si_external_projector_dir)
        del d["workflow"]["block_wannierization_threshold"]
        inp = KoopmansInput.model_validate(d)
        wg = _build_wannierize_workgraph(inp, split_codes)
        [w90_task] = [t for t in wg.tasks if "annier90WorkChain" in t.name]
        p2w = w90_task.inputs["pw2wannier90"]["pw2wannier90"]
        inputpp = p2w["parameters"].value.get_dict()["INPUTPP"]
        assert inputpp["atom_proj"] is True
        assert inputpp["atom_proj_ext"] is True
        assert inputpp["atom_proj_dir"] == "external_projectors/"
        assert "atom_proj_frozen" not in inputpp
        assert p2w["external_projectors_path"].value.get_remote_path() == str(
            si_external_projector_dir
        )
        assert p2w["external_projectors_list"].value.get_dict() == {"Si": "Si"}


class TestPseudoSocSniffing:
    """The ``has_so`` sniffing that gates automatic projections."""

    @staticmethod
    def _upf(has_so: bool | None) -> Any:
        import io

        from aiida_pseudo.data.pseudo.upf import UpfData

        from tests.fixtures import fake_upf_content

        content = fake_upf_content("Si", 4.0, has_so=has_so)
        return UpfData(io.BytesIO(content.encode("utf-8")), filename="Si.upf").store()

    def test_flag_values_are_read(self, aiida_profile: Any) -> None:
        """``has_so="F"`` reads scalar-relativistic; ``has_so="T"`` fully relativistic."""
        from koopmans.aiida.workflows import _pseudo_is_fully_relativistic

        assert _pseudo_is_fully_relativistic("Si", self._upf(False)) is False
        assert _pseudo_is_fully_relativistic("Si", self._upf(True)) is True

    def test_missing_flag_raises_a_named_error(self, aiida_profile: Any) -> None:
        """A header without ``has_so`` fails naming the pseudo, not with a bare TypeError."""
        from koopmans.aiida.workflows import _pseudo_is_fully_relativistic

        with pytest.raises(ValueError, match=r"Si does not declare `has_so`"):
            _pseudo_is_fully_relativistic("Si", self._upf(None))


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
