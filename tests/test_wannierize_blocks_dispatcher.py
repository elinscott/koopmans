"""Dispatcher tests for the block-by-block Wannierize route.

Builds real ``WorkGraph`` objects through ``_build_wannierize_blocks_workgraph``
against a throwaway profile (dummy codes, fake pseudos; nothing runs) and
checks the routing guards, the lenient block derivation, and the built graph
topology — with ``block_wannierization_threshold`` set (automated splitting)
and without it (one Wannierization per block).
"""

from __future__ import annotations

from typing import Any

import pytest
from aiida_koopmans.types import get_included_bands

from koopmans.aiida.workflows import (
    _build_wannierize_blocks_workgraph,
    _build_wannierize_workgraph,
    _create_explicit_blocks,
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
    return _build_wannierize_blocks_workgraph(inp, codes)


def _build_via_route(d: dict[str, Any], codes: dict[str, Any]) -> Any:
    """Build through the route selection, which is where the guards live."""
    inp = KoopmansInput.model_validate(d)
    return _build_wannierize_workgraph(inp, codes)


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
    """Unit tests for the block derivation as the split route uses it.

    Silicon has four occupied bands here, so a block reaching band 5 or
    above spans the occupied/empty boundary. The split route accepts such
    a block — cutting it at the boundary is its whole job — and the
    derivation marks it as provisional by leaving ``filled`` unset.
    """

    @staticmethod
    def _sp3_block() -> list[Any]:
        from koopmans.input_file import KoopmansInput

        inp = KoopmansInput.model_validate(_si_split_dict())
        return inp.calculator_parameters.wannier90.projections

    @staticmethod
    def _s_block() -> list[Any]:
        """Return a single block of s projections: one per Si site, 2 Wannier functions."""
        d = _si_split_dict()
        d["calculator_parameters"]["wannier90"]["projections"] = [[{"site": "Si", "ang_mtm": "s"}]]
        return KoopmansInput.model_validate(d).calculator_parameters.wannier90.projections

    @staticmethod
    def _blocks(structure: Any, projections: list[Any], nbnd: int) -> list[Any]:
        from aiida_koopmans.types import SpinChannel

        return _create_explicit_blocks(structure, projections, nbnd, 4, SpinChannel.NONE)

    def test_straddling_block_is_provisional(self, silicon_structure: Any) -> None:
        """A block spanning occupied and empty bands is unstamped, not an error.

        It is the split that will decide where this block's Wannier
        functions fall, so stating an occupancy now would be a guess.
        """
        blocks = self._blocks(silicon_structure, self._sp3_block(), nbnd=8)
        assert len(blocks) == 1
        assert blocks[0]["num_wann"] == 8
        assert blocks[0]["num_bands"] == 8
        assert get_included_bands(blocks[0]) == list(range(1, 9))
        assert blocks[0].get("exclude_bands") is None
        assert "filled" not in blocks[0]

    def test_blocks_within_one_manifold_are_stamped(self, silicon_structure: Any) -> None:
        """Blocks that fall on one side of the boundary state their occupancy.

        Nothing about the split route makes such a block provisional: it
        already lies where a Koopmans calculation needs it, and the runtime
        split can only cut it into pieces of the same occupancy.
        """
        blocks = self._blocks(silicon_structure, self._s_block() * 4, nbnd=8)
        assert [b["label"] for b in blocks] == ["occ_1", "occ_2", "emp_1", "emp_2"]
        assert [b["filled"] for b in blocks] == [True, True, False, False]

    def test_one_provisional_block_unstamps_the_set(self, silicon_structure: Any) -> None:
        """A provisional block leaves its stampable siblings unstamped too.

        Occupancies are consumed as a partition of every Wannier function,
        so a set covering only the blocks that happen to be final is worse
        than none at all.
        """
        blocks = self._blocks(silicon_structure, self._s_block() + self._sp3_block(), nbnd=10)
        assert [b["label"] for b in blocks] == ["occ_1", "block_2"]
        assert all("filled" not in block for block in blocks)

    def test_a_pool_crossing_the_boundary_is_provisional(self, silicon_structure: Any) -> None:
        """An occupied block that disentangles against empty bands is unstamped.

        The two bands it occupies are occupied bands, but wannier90
        optimizes its Wannier functions out of all ten bands it reads, so
        they are not the occupied manifold's and the two bands cannot say
        otherwise.
        """
        blocks = self._blocks(silicon_structure, self._s_block() * 2, nbnd=12)
        assert blocks[-1]["num_bands"] > blocks[-1]["num_wann"]
        assert all("filled" not in block for block in blocks)

    def test_last_block_absorbs_extra_bands(self, silicon_structure: Any) -> None:
        """An nbnd beyond the Wannier count becomes the disentanglement pool.

        The pool shows up as ``num_bands`` and the absent upper exclusion,
        and the bands derived from those two keep naming exactly the
        eight Wannier bands, so the runtime group detection and the
        band-to-Wannier map stay addressed to the manifold rather than the
        pool.
        """
        blocks = self._blocks(silicon_structure, self._sp3_block(), nbnd=12)
        assert blocks[0]["num_wann"] == 8
        assert blocks[0]["num_bands"] == 12
        assert get_included_bands(blocks[0]) == list(range(1, 9))
        assert blocks[0].get("exclude_bands") is None

    def test_too_few_bands_raises(self, silicon_structure: Any) -> None:
        """Projections needing more bands than nbnd are an input error."""
        with pytest.raises(ValueError, match="span 8 bands but nbnd = 6"):
            self._blocks(silicon_structure, self._sp3_block(), nbnd=6)


class TestGuards:
    """Routing guards for the unsupported configurations."""

    def test_collinear_not_implemented(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Collinear spin is not wired into any Wannierization route yet."""
        with pytest.raises(NotImplementedError, match="spin='none'"):
            _build_via_route(_si_split_dict(spin="collinear"), split_codes)

    def test_collinear_not_implemented_without_the_threshold(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Dropping the threshold does not make a route read ``workflow.spin``.

        No route sets ``nspin``, so a collinear input must fail rather than
        be Wannierized as if it were unpolarized.
        """
        d = _si_split_dict(spin="collinear")
        del d["workflow"]["block_wannierization_threshold"]
        with pytest.raises(NotImplementedError, match="spin='none'"):
            _build_via_route(d, split_codes)

    def test_missing_kpath_raises(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """The detection needs a bands run, hence a k-path."""
        d = _si_split_dict()
        d["kpoints"].pop("path")
        with pytest.raises(ValueError, match="k-point path"):
            _build(d, split_codes)

    def test_missing_kpath_is_fine_without_the_threshold(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """No threshold means no bands step, so no k-path is needed."""
        d = _si_split_dict()
        del d["workflow"]["block_wannierization_threshold"]
        d["kpoints"].pop("path")
        wg = _build(d, split_codes)
        assert "bands" not in [t.name for t in wg.tasks]

    @pytest.mark.parametrize("keep_top_level", [False, True])
    def test_spin_channel_projections_not_wired(
        self,
        aiida_profile_clean: Any,
        split_codes: Any,
        fake_sg15_cutoffs_family: Any,
        keep_top_level: bool,
    ) -> None:
        """This route reads the top-level block only, so channel blocks must not pass.

        Whether or not a top-level block accompanies them: on their own
        they used to be reported as no projection source at all, and
        alongside one they would have gone unread.
        """
        d = _si_split_dict()
        projections = d["calculator_parameters"]["wannier90"]["projections"]
        if not keep_top_level:
            del d["calculator_parameters"]["wannier90"]["projections"]
        d["calculator_parameters"]["wannier90"]["up"] = {"projections": projections}
        d["calculator_parameters"]["wannier90"]["down"] = {"projections": projections}
        with pytest.raises(NotImplementedError, match=r"w90.up.projections.*block-by-block"):
            _build(d, split_codes)


def _si_auto_dict(**workflow_updates: Any) -> dict[str, Any]:
    """Return the silicon split input with automatic projections.

    ``auto_projections`` replaces the explicit projections. The fake Si
    pseudo carries an s+p valence, so the two-atom cell has 8 atomic
    projectors — the automatic block spans bands 1-8 with the
    occupied/empty boundary at band 4 (nelec 8).
    """
    d = _si_split_dict(**workflow_updates)
    d["calculator_parameters"]["wannier90"] = {}
    d["workflow"].setdefault("auto_projections", True)
    return d


class TestAutomaticProjections:
    """The atomic-projector route behind ``workflow.auto_projections``."""

    def test_flag_with_explicit_projections_conflicts(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """The flag and explicit projections each define the full projection set."""
        d = _si_split_dict(auto_projections=True)
        with pytest.raises(ValueError, match=r"auto_projections.*were both given"):
            _build(d, split_codes)

    def test_no_projection_source_raises(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Dropping the projections without opting into the flag is an error."""
        d = _si_auto_dict(auto_projections=False)
        with pytest.raises(ValueError, match="Nothing defines the Wannier projections"):
            _build(d, split_codes)

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

        The bands it occupies must run over exactly ``1..num_wann`` — a
        shorter list would silently drop Wannier functions from the runtime
        split — and ``num_bands == num_wann`` is the no-pool invariant
        behind the nbnd guards. The block states no occupancy: it spans the
        whole manifold and exists only to be cut up by the runtime
        detection.
        """
        from aiida_wannier90_workflows.common.types import WannierProjectionType

        from koopmans.aiida.conversion import get_pseudos_from_family
        from koopmans.aiida.workflows import _create_automatic_blocks

        pseudos = get_pseudos_from_family(fake_sg15_cutoffs_family.label, silicon_structure)
        blocks, nbnd = _create_automatic_blocks(silicon_structure, pseudos, None, None, 4)
        [block] = blocks
        assert block["num_wann"] == 8
        assert block["num_bands"] == 8
        assert get_included_bands(block) == list(range(1, 9))
        assert block["projection_type"] == WannierProjectionType.ATOMIC_PROJECTORS_QE
        assert block.get("exclude_bands") is None
        assert "filled" not in block
        assert nbnd == 8

    def test_projectors_short_of_occupied_manifold_raise(
        self, aiida_profile_clean: Any, fake_sg15_cutoffs_family: Any, silicon_structure: Any
    ) -> None:
        """Projectors that cannot span the occupied manifold are rejected."""
        from koopmans.aiida.conversion import get_pseudos_from_family
        from koopmans.aiida.workflows import _create_automatic_blocks

        pseudos = get_pseudos_from_family(fake_sg15_cutoffs_family.label, silicon_structure)
        with pytest.raises(ValueError, match="cannot span the occupied manifold"):
            _create_automatic_blocks(silicon_structure, pseudos, None, None, 10)

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

    ``auto_projections`` asks for the block to be derived automatically and
    ``atom_proj_ext`` points the derivation at the projector directory,
    whose ``Si.dat`` (s + p per atom, 8 projectors for the two-atom cell)
    sizes the manifold.
    """
    d = _si_auto_dict(**workflow_updates)
    d["calculator_parameters"]["pw2wannier90"] = {
        "atom_proj_ext": True,
        "atom_proj_dir": str(projector_dir),
    }
    return d


class TestProjectorFileParsing:
    """The pw2wannier90-format reader behind the external projector sizing.

    Differentially validated against a transcription of QE's
    ``read_atomproj`` on a 30-file corpus: the reader agrees with it on
    every case except three deliberately stricter rejections — a zero
    projector count, an early ``/`` terminator and negative angular
    momenta — where the Fortran reader proceeds with undefined or
    unusable values.
    """

    @staticmethod
    def _read(tmp_path: Any, content: str) -> list[int]:
        from koopmans.aiida.workflows import _read_projector_angular_momenta

        projector_file = tmp_path / "X.dat"
        projector_file.write_text(content)
        return _read_projector_angular_momenta(projector_file)

    @pytest.mark.parametrize(
        ("content", "momenta"),
        [
            # Space-indented comments are skipped, as QE's ADJUSTL check does.
            ("# a\n  # b\n100 3\n0 1 2\n0.0 0.1\n", [0, 1, 2]),
            # Blank records are skipped anywhere a list-directed read runs:
            # after the comment block (the QE example07 shape), before any
            # data at all, and between the header and the momenta.
            ("# c\n\n100 2\n0 1\n", [0, 1]),
            ("\n100 2\n0 1\n", [0, 1]),
            ("100 2\n\n0 1\n", [0, 1]),
            # The header itself is a list-directed read: it may span records.
            ("100\n2\n0 1\n", [0, 1]),
            # Blanks and/or commas separate values.
            ("100, 2\n0, 1\n", [0, 1]),
            ("100,2\n0,1\n", [0, 1]),
            # r*v repeat counts expand.
            ("100 4\n4*0\n", [0, 0, 0, 0]),
            # The momenta may continue over records...
            ("100 4\n0 1\n1 2\n", [0, 1, 1, 2]),
            # ...and the read stops mid-record once the count is met, so
            # surplus tokens (the radial tables) are never inspected.
            ("100 2\n0 1 99 98\n", [0, 1]),
            ("100 3\n0 1\n5 0.1 0.2\n", [0, 1, 5]),
            # Explicit plus signs are plain integers.
            ("100 2\n+0 +1\n", [0, 1]),
        ],
    )
    def test_list_directed_acceptance(
        self, tmp_path: Any, content: str, momenta: list[int]
    ) -> None:
        """Files QE's reader accepts parse to the same angular momenta."""
        assert self._read(tmp_path, content) == momenta

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            # A tab-indented `#` is not a comment to QE (ADJUSTL shifts
            # spaces only); it becomes the header record and fails there.
            ("\t# tab comment\n100 2\n0 1\n", "'#' is not an integer"),
            # Each read starts on a fresh record, so momenta on the header
            # record are discarded — exactly as QE then fails at EOF.
            ("100 2 0 1\n", "ends before the angular-momentum list"),
            ("# only comments\n", r"ends before the `<ngrid> <nproj>` header"),
            ("", r"ends before the `<ngrid> <nproj>` header"),
            ("100 x\n0 1\n", "'x' is not an integer"),
            # A one-token header pulls nproj from the next record: here 0.
            ("100\n0 1\n", "at least one projector"),
            ("100 0\n\n", "at least one projector"),
            ("100 -2\n0 1\n", "at least one projector"),
            ("100 3\n0 1\n", "ends before the angular-momentum list"),
            ("100 2\n0 q\n", "'q' is not an integer"),
            ("100 2\n0 1.0\n", "'1.0' is not an integer"),
            ("100 2\n0 -1\n", r"negative angular momenta: \[-1\]"),
            ("100 2\n0 /\n", "leaving the rest undefined"),
            ("100 2\n0,,1\n", "adjacent commas"),
        ],
    )
    def test_rejected_files_raise(self, tmp_path: Any, content: str, match: str) -> None:
        """Every rejection fails naming the file and the reason."""
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
        """The derived block carries the external type, the no-pool shape and no occupancy."""
        from aiida_wannier90_workflows.common.types import WannierProjectionType

        from koopmans.aiida.conversion import get_pseudos_from_family
        from koopmans.aiida.workflows import _create_automatic_blocks
        from tests.fixtures import si_external_projector_tables

        pseudos = get_pseudos_from_family(fake_sg15_cutoffs_family.label, silicon_structure)
        blocks, nbnd = _create_automatic_blocks(
            silicon_structure, pseudos, si_external_projector_tables(), None, 4
        )
        [block] = blocks
        assert block["num_wann"] == 8
        assert block["num_bands"] == 8
        assert get_included_bands(block) == list(range(1, 9))
        assert block["projection_type"] == WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL
        assert block.get("exclude_bands") is None
        assert "filled" not in block
        assert nbnd == 8

    @pytest.mark.parametrize("channels", [False, True])
    def test_explicit_projections_and_external_projectors_conflict(
        self,
        aiida_profile_clean: Any,
        split_codes: Any,
        fake_sg15_cutoffs_family: Any,
        si_external_projector_dir: Any,
        channels: bool,
    ) -> None:
        """Explicit projections would silently shadow the external projectors.

        Both where they live at the top level and where they live only in
        the spin channels: the channel blocks are just as explicit, so the
        conflict must name them rather than let the build proceed with the
        channel projections quietly dropped.
        """
        d = _si_split_dict()
        if channels:
            projections = d["calculator_parameters"]["wannier90"].pop("projections")
            d["calculator_parameters"]["wannier90"]["up"] = {"projections": projections}
            d["calculator_parameters"]["wannier90"]["down"] = {"projections": projections}
        d["calculator_parameters"]["pw2wannier90"] = {
            "atom_proj_ext": True,
            "atom_proj_dir": str(si_external_projector_dir),
        }
        expected = "w90.up.projections" if channels else "w90.projections"
        with pytest.raises(ValueError, match=rf"{expected}.*Drop one of the two"):
            _build(d, split_codes)

    def test_external_projectors_require_the_flag(
        self,
        aiida_profile_clean: Any,
        split_codes: Any,
        fake_sg15_cutoffs_family: Any,
        si_external_projector_dir: Any,
    ) -> None:
        """``atom_proj_ext`` alone does not ask for automatic blocks.

        The external files supply projector functions; the single block
        spanning the manifold comes from the automatic derivation, so
        without the flag nothing has asked for it.
        """
        d = _si_external_dict(si_external_projector_dir, auto_projections=False)
        with pytest.raises(ValueError, match=r"atom_proj_ext.*without `workflow.auto_projections`"):
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


def _build_plain(d: dict[str, Any], codes: dict[str, Any]) -> Any:
    """Build through the route selection with the threshold dropped."""
    del d["workflow"]["block_wannierization_threshold"]
    inp = KoopmansInput.model_validate(d)
    return _build_wannierize_workgraph(inp, codes)


class TestPlainRoute:
    """Routing and gating without ``block_wannierization_threshold``."""

    def test_explicit_projections_need_no_wannierjl_code(
        self,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        installed_wannier_codes: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """Nothing splits, so the julia code the splitting needs is not required.

        ``load_codes_for_task`` only loads it behind the threshold, so a
        build that demanded it here would fail for every user with
        explicit projections.
        """
        codes = {"pw": installed_pw_code, **installed_wannier_codes}
        assert "wannierjl" not in codes
        wg = _build_plain(_si_split_dict(), codes)
        assert "wannierize_block_1" in [t.name for t in wg.tasks]

    def test_flag_builds_the_qe_projector_route(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """The flag routes through upstream's pseudo-atomic-projector mechanism.

        The eager build reaches the staged wannier90 / pw2wannier90 inputs:
        ``auto_projections`` in the wannier90 parameters and ``atom_proj``
        in pw2wannier90 are upstream's own automatic-projection switches.
        """
        wg = _build_plain(_si_auto_dict(), split_codes)
        [w90_task] = [t for t in wg.tasks if "annier90WorkChain" in t.name]
        w90_params = w90_task.inputs["wannier90"]["wannier90"]["parameters"].value.get_dict()
        assert w90_params["auto_projections"] is True
        inputpp = w90_task.inputs["pw2wannier90"]["pw2wannier90"]["parameters"].value.get_dict()[
            "INPUTPP"
        ]
        assert inputpp["atom_proj"] is True

    def test_flag_with_explicit_projections_conflicts(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """The flag-vs-explicit conflict applies on this route too."""
        with pytest.raises(ValueError, match=r"auto_projections.*were both given"):
            _build_plain(_si_split_dict(auto_projections=True), split_codes)

    def test_explicit_projections_wannierize_block_by_block(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Explicit projections route to one Wannierization per block, splitting nothing.

        Two blocks in, two Wannierizations out, off a single shared scf +
        nscf. None of the split machinery is built: no bands step, no group
        detection. Each task is named after its block: the two s-type
        Wannier functions sit wholly in the occupied manifold, while the
        six p-type ones straddle the boundary and stay provisional.
        """
        d = _si_split_dict()
        d["calculator_parameters"]["wannier90"]["projections"] = [
            [{"site": "Si", "ang_mtm": "s"}],
            [{"site": "Si", "ang_mtm": "p"}],
        ]
        wg = _build_plain(d, split_codes)
        names = [t.name for t in wg.tasks]
        assert names.count("scf_nscf") == 1
        assert sorted(n for n in names if n.startswith("wannierize_")) == [
            "wannierize_block_2",
            "wannierize_occ_1",
        ]
        assert "bands" not in names
        assert "detect_band_groups" not in names

        # Blocks cover consecutive bands in input order: 2 s-type Wannier
        # functions then 6 p-type ones over the 8-band manifold.
        blocks = [
            wg.tasks[name].inputs["block"].value
            for name in ("wannierize_occ_1", "wannierize_block_2")
        ]
        assert [b["num_wann"] for b in blocks] == [2, 6]
        assert get_included_bands(blocks[0]) == [1, 2]
        assert get_included_bands(blocks[1]) == list(range(3, 9))

    def test_spin_channel_projections_not_wired(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Per-channel projections stay unwired: the route reads the top-level block."""
        d = _si_split_dict()
        projections = d["calculator_parameters"]["wannier90"].pop("projections")
        d["calculator_parameters"]["wannier90"]["up"] = {"projections": projections}
        d["calculator_parameters"]["wannier90"]["down"] = {"projections": projections}
        with pytest.raises(NotImplementedError, match=r"w90.up.projections.*block-by-block"):
            _build_plain(d, split_codes)

    def test_external_projectors_require_the_flag(
        self,
        aiida_profile_clean: Any,
        split_codes: Any,
        fake_sg15_cutoffs_family: Any,
        si_external_projector_dir: Any,
    ) -> None:
        """The flag requirement holds on this route too, before the route's own gates."""
        d = _si_external_dict(si_external_projector_dir, auto_projections=False)
        with pytest.raises(ValueError, match=r"atom_proj_ext.*without `workflow.auto_projections`"):
            _build_plain(d, split_codes)

    def test_no_projection_source_raises(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Without any projection source the route fails naming the options."""
        with pytest.raises(ValueError, match="Nothing defines the Wannier projections"):
            _build_plain(_si_auto_dict(auto_projections=False), split_codes)


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

    def test_scf_samples_the_input_mesh(
        self, aiida_profile_clean: Any, split_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """The input file's grid reaches the scf, not just the nscf.

        Left to the protocol the scf would pick its own mesh from a
        k-point distance, so the calculation would not be the one the
        input file describes.
        """
        wg = _build(_si_split_dict(), split_codes)
        scf_kpoints = wg.tasks["scf_nscf"].inputs["scf_kpoints"].value
        assert list(scf_kpoints.get_kpoints_mesh()[0]) == [2, 2, 2]
        # The nscf keeps the unreduced expansion of the same grid.
        assert len(wg.tasks["scf_nscf"].inputs["nscf_kpoints"].value.get_kpoints()) == 8
