"""Dispatcher tests for the Wannier-initialised (periodic mlwfs) DSCF route.

Exercises the block derivation together with the checks this route adds to
it (pure bookkeeping), and builds real ``WorkGraph`` objects through
``_build_singlepoint_workgraph`` for a periodic silicon input (throwaway
profile, dummy codes; nothing runs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from koopmans.aiida.workflows import (
    _build_singlepoint_workgraph,
    _dscf_wannier_init_inputs,
)
from koopmans.input_file import KoopmansInput

if TYPE_CHECKING:
    from wannier90_input.models.parameters import Projection


def _si_dscf_dict(**workflow_updates: Any) -> dict[str, Any]:
    """Return a minimal periodic-silicon DSCF+mlwfs input dict."""
    d: dict[str, Any] = {
        "workflow": {
            "task": "singlepoint",
            "correction": "ki",
            "screening_method": "dscf",
            "init_orbitals": "mlwfs",
            "calculate_alpha": True,
            "pseudo_library": "SG15/1.2/PBE/SR",
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
        "calculator_parameters": {
            "ecutwfc": 20.0,
            "nbnd": 8,
            # occ: sp on both Si sites (4 wann = nocc for nelec 8);
            # emp: another sp block covering bands 5-8.
            "wannier90": {
                "projections": [
                    [{"site": "Si", "ang_mtm": "sp"}],
                    [{"site": "Si", "ang_mtm": "sp"}],
                ],
            },
        },
    }
    d["workflow"].update(workflow_updates)
    return d


def _si_collinear_dscf_dict() -> dict[str, Any]:
    """Return the same input with both spin channels projected separately.

    Silicon is unmagnetized, so both channels carry four occupied bands
    and the same sp projections: an occupied block over bands 1-4 and an
    empty one over 5-8.
    """
    sp = [{"site": "Si", "ang_mtm": "sp"}]
    d = _si_dscf_dict(spin="collinear")
    d["calculator_parameters"]["tot_magnetization"] = 0
    d["calculator_parameters"]["wannier90"] = {
        "up": {"projections": [sp, sp]},
        "down": {"projections": [sp, sp]},
    }
    return d


def _build(d: dict[str, Any], codes: dict[str, Any]) -> Any:
    inp = KoopmansInput.model_validate(d)
    return _build_singlepoint_workgraph(inp, codes=codes)


def _dscf_blocks(
    structure: Any,
    projection_blocks: list[Any],
    nocc: int,
    nbnd: int,
    spin_channel: Any,
) -> list[Any]:
    """Derive and check blocks the way the Wannier-initialised route does.

    The route derives blocks with the same helper the split route uses and
    then applies the checks its supercell fold needs, so the two steps only
    mean anything together.
    """
    from koopmans.aiida.workflows import (
        _create_explicit_blocks,
        _validate_blocks_cover_all_occ_bands,
        _validate_blocks_separate_occ_and_emp,
    )

    blocks = _create_explicit_blocks(structure, projection_blocks, nbnd, nocc, spin_channel)
    _validate_blocks_separate_occ_and_emp(blocks, nocc)
    _validate_blocks_cover_all_occ_bands(blocks, nocc)
    return blocks


@pytest.fixture
def dscf_codes(
    installed_pw_code: Any,
    installed_kcp_code: Any,
    installed_wannier_codes: Any,
    installed_fold_codes: Any,
) -> dict[str, Any]:
    """Assemble the codes dict the dispatcher receives, plus fold-path dummies.

    Only ``pw`` and ``kcp`` are passed in (mirroring ``load_codes_for_task``);
    the wannier / fold codes are looked up by label inside the builder, so the
    fixtures merely register them.
    """
    return {"pw": installed_pw_code, "kcp": installed_kcp_code}


class TestDscfBlocks:
    """Unit tests for the projection-block bookkeeping."""

    @pytest.fixture
    def si_structure(self, aiida_profile: Any) -> Any:
        """Return a bare silicon StructureData for projection counting."""
        from aiida.orm import StructureData

        cell = [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
        struct = StructureData(cell=cell, pbc=True)
        struct.append_atom(  # type: ignore[no-untyped-call]
            position=(0.0, 0.0, 0.0), symbols="Si", name="Si"
        )
        struct.append_atom(  # type: ignore[no-untyped-call]
            position=(1.3575, 1.3575, 1.3575), symbols="Si", name="Si"
        )
        return struct

    class _FakeQuantumNumbers:
        def __init__(self, l_value: int) -> None:
            self.angular = type("A", (), {"value": l_value})()
            self.m_r = None

        def __str__(self) -> str:
            return f"l={self.angular.value}"

    class _FakeProjection:
        def __init__(
            self,
            site: str | None,
            l_value: int,
            fractional_site: list[float] | None = None,
        ) -> None:
            self.site = site
            self.fractional_site = fractional_site
            self.cartesian_site = None
            self.ang_mtm = TestDscfBlocks._FakeQuantumNumbers(l_value)

    def test_fractional_site_projections(self, si_structure: Any) -> None:
        """Point-hosted (bond-centred) projections derive and format.

        One fractional site hosts exactly one orbital set (sp3 -> 4 WFs) and
        renders as wannier90's ``f=x,y,z:<ang_mtm>`` form.
        """
        from aiida_koopmans.types import SpinChannel

        sp3 = [self._FakeProjection(None, -3, fractional_site=[0.25, 0.25, 0.25])]
        blocks = _dscf_blocks(si_structure, [sp3, sp3], 4, 8, SpinChannel.NONE)
        occ, emp = blocks
        assert occ["num_wann"] == 4
        assert occ["projections"] == ["f=0.25,0.25,0.25:l=-3"]
        assert emp["num_wann"] == 4
        assert emp["num_bands"] == 4  # sized to its own manifold, no pool

    def test_every_block_is_sized_to_its_own_manifold(self, si_structure: Any) -> None:
        """With nbnd exactly spanned, no block carries a disentanglement pool.

        Each block spans exactly the bands its projections name
        (``num_bands == num_wann``) and excludes everything below *and*
        above, so U_dis is the identity.
        """
        from aiida_koopmans.types import SpinChannel

        sp = [self._FakeProjection("Si", -1)]  # 2 orbitals x 2 sites = 4
        blocks = _dscf_blocks(si_structure, [sp, sp], 4, 8, SpinChannel.NONE)
        occ, emp = blocks
        assert occ["num_bands"] == 4
        assert occ["exclude_bands"] == [5, 6, 7, 8]
        assert emp["num_wann"] == 4
        assert emp["num_bands"] == 4
        assert emp["include_bands"] == [5, 6, 7, 8]
        assert emp["exclude_bands"] == [1, 2, 3, 4]

    def test_leftover_nscf_bands_become_the_pool(self, si_structure: Any) -> None:
        """Absorb nscf headroom above the blocks into the uppermost block's pool.

        The pool widens only ``num_bands`` and drops the upper exclusion —
        ``include_bands`` still names the four Wannier bands, since it is the
        band-to-Wannier-function map every downstream consumer reads.
        """
        from aiida_koopmans.types import SpinChannel

        sp = [self._FakeProjection("Si", -1)]  # 4
        occ, emp = _dscf_blocks(si_structure, [sp, sp], 4, 20, SpinChannel.NONE)
        assert occ["num_bands"] == 4
        assert occ["exclude_bands"] == list(range(5, 21))
        assert emp["num_wann"] == 4
        assert emp["num_bands"] == 16  # 4 Wannier bands + 12 pool bands
        assert emp["include_bands"] == [5, 6, 7, 8]
        assert emp["exclude_bands"] == [1, 2, 3, 4]

    def test_pool_block_preserves_the_wann2kcp_band_identity(self, si_structure: Any) -> None:
        """Every block satisfies ``len(exclude_bands) + num_bands == nbnd``.

        wann2kcp.x reads the ``.chk`` against the pw.x band count and rejects
        any block whose excluded and read bands do not add back up to it.
        """
        from aiida_koopmans.types import SpinChannel

        sp = [self._FakeProjection("Si", -1)]  # 4
        for nbnd in (8, 12, 20):
            for block in _dscf_blocks(si_structure, [sp, sp], 4, nbnd, SpinChannel.NONE):
                excluded = block["exclude_bands"] or []
                assert len(excluded) + block["num_bands"] == nbnd

    def test_occ_emp_split_and_exclusions(self, si_structure: Any) -> None:
        """Two sp blocks split into occ_1 (bands 1-4) and emp_1 (5-8)."""
        from aiida_koopmans.types import SpinChannel

        sp = [self._FakeProjection("Si", -1)]  # 2 orbitals x 2 sites = 4
        blocks = _dscf_blocks(si_structure, [sp, sp], 4, 8, SpinChannel.NONE)
        assert [b["label"] for b in blocks] == ["occ_1", "emp_1"]
        assert blocks[0]["include_bands"] == [1, 2, 3, 4]
        assert blocks[0]["exclude_bands"] == [5, 6, 7, 8]
        assert blocks[1]["include_bands"] == [5, 6, 7, 8]
        assert blocks[1]["exclude_bands"] == [1, 2, 3, 4]

    def test_middle_block_gets_two_sided_exclusion(self, si_structure: Any) -> None:
        """A block sandwiched between others excludes bands on both sides."""
        from aiida_koopmans.types import SpinChannel

        s = [self._FakeProjection("Si", 0)]  # 1 x 2 sites = 2
        sp = [self._FakeProjection("Si", -1)]  # 4
        blocks = _dscf_blocks(si_structure, [s, s, sp], 4, 8, SpinChannel.NONE)
        assert [b["label"] for b in blocks] == ["occ_1", "occ_2", "emp_1"]
        assert blocks[1]["exclude_bands"] == [1, 2, 5, 6, 7, 8]

    def test_straddling_block_raises(self, si_structure: Any) -> None:
        """A block crossing the occupied/empty boundary is an input error."""
        from aiida_koopmans.types import SpinChannel

        sp = [self._FakeProjection("Si", -1)]  # 4
        with pytest.raises(ValueError, match="straddles"):
            _dscf_blocks(si_structure, [sp, sp], 6, 8, SpinChannel.NONE)

    def test_band_count_is_reported_before_the_straddle(self, si_structure: Any) -> None:
        """A block that both overruns nbnd and straddles reports the band count.

        Both conditions hold for the second block here (bands 5-8, boundary
        at 6, nbnd 6). The band count is the error every route agrees on —
        eight Wannier functions cannot come out of six bands — so the shared
        derivation raises it and the straddle, which only the Wannier-seeded
        route objects to, is never reached.
        """
        from aiida_koopmans.types import SpinChannel

        sp = [self._FakeProjection("Si", -1)]  # 4
        with pytest.raises(ValueError, match="span 8 bands but nbnd = 6"):
            _dscf_blocks(si_structure, [sp, sp], 6, 6, SpinChannel.NONE)

    def test_pool_above_an_occupied_block_raises(self, si_structure: Any) -> None:
        """Occupied-only projections must not disentangle against empty bands.

        The pool would land on the topmost *occupied* block, whose Wannier
        functions seed the occupied manifold of the supercell kcp.x run;
        letting them mix in empty character corrupts that seed with nothing
        downstream to catch it.
        """
        from aiida_koopmans.types import SpinChannel

        sp = [self._FakeProjection("Si", -1)]  # 4 = nocc
        with pytest.raises(ValueError, match="disentanglement pool"):
            _dscf_blocks(si_structure, [sp], 4, 8, SpinChannel.NONE)

    def test_boundary_check_alone_rejects_a_pool_that_crosses(self, si_structure: Any) -> None:
        """The boundary check rejects a pool crossing it, reading no band slots.

        This block's own bands are the four occupied ones — a check that
        compares band slots against the boundary sees nothing wrong with it,
        and only its disentanglement pool reaches into the empty manifold.
        Asking the plugin whether the block is occupied is what catches it,
        so run that check on its own: paired with the coverage check it
        would pass for the wrong reason.
        """
        from aiida_koopmans.types import SpinChannel

        from koopmans.aiida.workflows import (
            _create_explicit_blocks,
            _validate_blocks_separate_occ_and_emp,
        )

        # The stand-in projections duck-type the pydantic model the
        # derivation reads (``.site`` / ``.ang_mtm``), which is all it touches.
        sp = cast("list[Projection]", [self._FakeProjection("Si", -1)])  # 4 = nocc
        blocks = _create_explicit_blocks(si_structure, [sp], 8, 4, SpinChannel.NONE)
        assert blocks[0]["include_bands"] == [1, 2, 3, 4]  # entirely occupied slots
        with pytest.raises(ValueError, match="disentanglement pool"):
            _validate_blocks_separate_occ_and_emp(blocks, 4)

    def test_uncovered_occupied_bands_raise(self, si_structure: Any) -> None:
        """Occupied blocks must cover every occupied band.

        The nscf stops at the occupied manifold, so the single block is
        occupied outright and the boundary check has nothing to say; what is
        wrong is that two Wannier functions cannot seed four occupied bands.
        """
        from aiida_koopmans.types import SpinChannel

        s = [self._FakeProjection("Si", 0)]  # 2 wann < nocc 4
        with pytest.raises(ValueError, match="every occupied band"):
            _dscf_blocks(si_structure, [s], 4, 4, SpinChannel.NONE)

    def test_blocks_beyond_nbnd_raise(self, si_structure: Any) -> None:
        """Blocks spanning more bands than nbnd are an input error."""
        from aiida_koopmans.types import SpinChannel

        sp = [self._FakeProjection("Si", -1)]
        with pytest.raises(ValueError, match="nbnd"):
            _dscf_blocks(si_structure, [sp, sp], 4, 6, SpinChannel.NONE)

    def test_blocks_carry_their_occupancy(self, si_structure: Any) -> None:
        """Each block states which manifold it belongs to.

        Downstream the merge places a block by this stamp, so a derivation
        that only named the block ``occ_1`` would leave the occupancy to be
        re-read off the band indices — which is exactly what the plugin
        stopped doing.
        """
        from aiida_koopmans.types import SpinChannel

        sp = [self._FakeProjection("Si", -1)]  # 4
        blocks = _dscf_blocks(si_structure, [sp, sp], 4, 8, SpinChannel.NONE)
        assert [b["filled"] for b in blocks] == [True, False]


class TestPeriodicMlwfsBuild:
    """Graph-construction tests for the Wannier-initialised DSCF route."""

    def test_wannier_route_builds(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The periodic mlwfs input builds the Wannier-seeded workgraph."""
        wg = _build(_si_dscf_dict(), dscf_codes)
        names = wg.get_task_names()
        assert "wannier_initialization" in names, names
        assert "make_supercell" in names, names
        # The molecular KS-init chain must NOT be present.
        assert "dft_init_nspin1" not in names

    def test_pool_carrying_input_builds_and_validates(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """An nscf with headroom above the Wannier manifold is a buildable input.

        ``nbnd`` sets the kcp.x orbital count and ``pw.system.nbnd`` the nscf
        band count; the eight bands between them are the uppermost block's
        disentanglement pool. Building is not enough to call this wired —
        ``check_before_run`` is what proves every task of the assembled graph
        has the inputs it declares as required.
        """
        d = _si_dscf_dict()
        d["calculator_parameters"]["pw"] = {"system": {"nbnd": 16}}
        wg = _build(d, dscf_codes)
        assert "wannier_initialization" in wg.get_task_names()
        wg.check_before_run()

    def test_wannier_initialization_gets_the_input_mesh(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The mesh handed over carries the input file's grid.

        This route reaches its scf through the Wannier initialization, which
        samples exactly this node; nothing downstream can recover a grid the
        dispatcher does not hand over.
        """
        d = _si_dscf_dict()
        d["kpoints"] = {"grid": [4, 4, 4], "offset": [0, 0, 0]}
        wg = _build(d, dscf_codes)
        kpoints = wg.tasks["wannier_initialization"].inputs["kpoints"].value
        assert list(kpoints.get_kpoints_mesh()[0]) == [4, 4, 4]

    def test_self_hartree_grouping_defaulted(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Wannier-initialised runs resolve to self-Hartree grouping at 1e-4 eV."""
        wg = _build(_si_dscf_dict(), dscf_codes)
        tol = wg.tasks["ComputeScreeningParameters"].inputs["self_hartree_tol"].value
        assert tol == pytest.approx(1.0e-4)

    def test_user_grouping_tol_wins(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """An explicit group_orbitals_tol overrides the criterion default."""
        d = _si_dscf_dict(group_orbitals_tol=0.05)
        wg = _build(d, dscf_codes)
        tol = wg.tasks["ComputeScreeningParameters"].inputs["self_hartree_tol"].value
        assert tol == pytest.approx(0.05)

    def test_grouping_none_disables(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """group_orbitals_by='none' disables grouping even on the Wannier route."""
        d = _si_dscf_dict(group_orbitals_by="none")
        wg = _build(d, dscf_codes)
        tol = wg.tasks["ComputeScreeningParameters"].inputs["self_hartree_tol"].value
        assert tol is None

    def test_tol_without_criterion_rejected(self, aiida_profile: Any) -> None:
        """A tolerance alongside group_orbitals_by='none' fails validation."""
        import pytest as _pytest

        from koopmans.input_file import KoopmansInput

        d = _si_dscf_dict(group_orbitals_by="none", group_orbitals_tol=0.05)
        with _pytest.raises(ValueError, match="group_orbitals_tol"):
            KoopmansInput(**d)

    def test_eps_inf_auto_not_wired(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """eps_inf='auto' is still NotImplemented for the DSCF stream."""
        d = _si_dscf_dict(eps_inf="auto")
        with pytest.raises(NotImplementedError, match="eps_inf"):
            _build(d, dscf_codes)

    def test_w90_keywords_land_flat_in_overrides(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """User wannier90 keywords ride into ``wannier_overrides['wannier90']`` flat.

        The dispatcher assembles the flat ``WannierizeOverrides`` shape (the
        namespace nesting is added later, inside the block wannierization
        builder), so the keyword dict appears verbatim under ``wannier90``.
        """
        from koopmans.aiida.conversion import atoms_input_to_structure

        d = _si_dscf_dict()
        d["calculator_parameters"]["wannier90"]["num_iter"] = 17
        inp = KoopmansInput.model_validate(d)
        structure = atoms_input_to_structure(inp.atoms)
        nbnd = inp.calculator_parameters.nbnd
        assert nbnd is not None
        extra = _dscf_wannier_init_inputs(inp, structure, dscf_codes, nbnd)
        assert extra["wannier_overrides"]["wannier90"] == {"num_iter": 17}
        # Projections are consumed by the block derivation, never leaked into
        # the flat keyword override.
        assert "projections" not in extra["wannier_overrides"]["wannier90"]

    def test_route_stamps_the_occupancies_it_hands_over(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The blocks reaching the workgraph say which manifold they are in.

        The merge grouping is a deferred graph body, so a route that handed
        over unstamped blocks would build cleanly and fail on the daemon;
        this reads the stamps at the hand-over point instead.
        """
        from koopmans.aiida.conversion import atoms_input_to_structure

        inp = KoopmansInput.model_validate(_si_dscf_dict())
        structure = atoms_input_to_structure(inp.atoms)
        nbnd = inp.calculator_parameters.nbnd
        assert nbnd is not None
        extra = _dscf_wannier_init_inputs(inp, structure, dscf_codes, nbnd)
        assert [(b["label"], b["filled"]) for b in extra["blocks"]] == [
            ("occ_1", True),
            ("emp_1", False),
        ]

    def test_collinear_route_stamps_both_channels(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Each spin channel's blocks are stamped against its own band count.

        The two channels are derived separately, so a stamp reaching only the
        first would leave the down channel to fail on the daemon alone.
        """
        from koopmans.aiida.conversion import atoms_input_to_structure

        inp = KoopmansInput.model_validate(_si_collinear_dscf_dict())
        structure = atoms_input_to_structure(inp.atoms)
        nbnd = inp.calculator_parameters.nbnd
        assert nbnd is not None
        extra = _dscf_wannier_init_inputs(inp, structure, dscf_codes, nbnd)
        assert [(b["label"], b["filled"]) for b in extra["blocks"]] == [
            ("occ_up_1", True),
            ("emp_up_1", False),
            ("occ_down_1", True),
            ("emp_down_1", False),
        ]

    def test_collinear_route_builds(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Two spin channels of blocks assemble into a runnable graph.

        The occupancies the route now states are consumed at build time by
        the Wannierization, which orders the orbitals by channel and then by
        filling; ``check_before_run`` is what proves the assembled graph has
        every input it declares as required.
        """
        wg = _build(_si_collinear_dscf_dict(), dscf_codes)
        assert "wannier_initialization" in wg.get_task_names()
        wg.check_before_run()

    def test_no_w90_keywords_leaves_overrides_flat(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """With only projections set, no ``wannier90`` override key is added."""
        from koopmans.aiida.conversion import atoms_input_to_structure

        inp = KoopmansInput.model_validate(_si_dscf_dict())
        structure = atoms_input_to_structure(inp.atoms)
        nbnd = inp.calculator_parameters.nbnd
        assert nbnd is not None
        extra = _dscf_wannier_init_inputs(inp, structure, dscf_codes, nbnd)
        assert set(extra["wannier_overrides"]) == {"scf", "nscf"}
