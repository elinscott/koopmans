"""Dispatcher tests for the Wannier-initialised (periodic mlwfs) DSCF route.

Exercises the block derivation together with the checks this route adds to
it (pure bookkeeping), and builds real ``WorkGraph`` objects through
``build_singlepoint_workgraph`` for a periodic silicon input (throwaway
profile, dummy codes; nothing runs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from aiida_koopmans.projections import get_wannier_indices
from wannier90_input.models.parameters import Projection

from koopmans.aiida.workflows.dscf import (
    build_singlepoint_workgraph,
    dscf_wannier_init_inputs,
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


def _build(d: dict[str, Any]) -> Any:
    inp = KoopmansInput.model_validate(d)
    return build_singlepoint_workgraph(inp)


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
    from koopmans.aiida.workflows.blocks import (
        create_explicit_blocks,
        validate_blocks_cover_all_occ_bands,
        validate_blocks_separate_occ_and_emp,
    )

    blocks = create_explicit_blocks(structure, projection_blocks, nbnd, nocc, spin_channel)
    validate_blocks_separate_occ_and_emp(blocks, nocc)
    validate_blocks_cover_all_occ_bands(blocks, nocc)
    return blocks


@pytest.fixture
def dscf_codes(
    installed_pw_code: Any,
    installed_kcp_code: Any,
    installed_wannier_codes: Any,
    installed_fold_codes: Any,
) -> dict[str, Any]:
    """Register every dummy code the Wannier-initialised DSCF route resolves.

    The route loads each ``DscfCodes`` member as ``<name>@localhost``, so
    the fixtures merely register them; the returned mapping only serves
    tests that inspect a code directly.
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

    def test_fractional_site_projections(self, si_structure: Any) -> None:
        """Point-hosted (bond-centred) projections derive and format.

        One fractional site hosts exactly one orbital set (sp3 -> 4 WFs) and
        renders as wannier90's ``f=x,y,z:<ang_mtm>`` form.
        """
        from aiida_koopmans.spin import SpinChannel

        sp3 = [Projection(fractional_site=[0.25, 0.25, 0.25], ang_mtm="l=-3")]
        blocks = _dscf_blocks(si_structure, [sp3, sp3], 4, 8, SpinChannel.NONE)
        occ, emp = blocks
        assert occ["num_wann"] == 4
        assert occ["projections"] == ["f=0.25,0.25,0.25:l=-3:0,0,1:1,0,0:1:1.0"]
        assert emp["num_wann"] == 4
        assert emp["num_bands"] == 4  # sized to its own manifold, no extra bands

    def test_every_block_is_sized_to_its_own_manifold(self, si_structure: Any) -> None:
        """With nbnd exactly spanned, no block requires disentanglement.

        Each block spans exactly the bands its projections name
        (``num_bands == num_wann``) and excludes everything below *and*
        above, so U_dis is the identity.
        """
        from aiida_koopmans.spin import SpinChannel

        sp = [Projection(site="Si", ang_mtm="l=-1")]  # 2 orbitals x 2 sites = 4
        blocks = _dscf_blocks(si_structure, [sp, sp], 4, 8, SpinChannel.NONE)
        occ, emp = blocks
        assert occ["num_bands"] == 4
        assert occ["exclude_bands"] == [5, 6, 7, 8]
        assert emp["num_wann"] == 4
        assert emp["num_bands"] == 4
        assert get_wannier_indices(emp) == [5, 6, 7, 8]
        assert emp["exclude_bands"] == [1, 2, 3, 4]

    def test_leftover_nscf_bands_become_the_extra_bands(self, si_structure: Any) -> None:
        """Absorb nscf headroom above the blocks into the uppermost block.

        The extra disentanglement bands widen only ``num_bands`` and drop
        the upper exclusion — the derived Wannier-function indices still
        name the block's four functions, since they are the map every
        downstream consumer reads.
        """
        from aiida_koopmans.spin import SpinChannel

        sp = [Projection(site="Si", ang_mtm="l=-1")]  # 4
        occ, emp = _dscf_blocks(si_structure, [sp, sp], 4, 20, SpinChannel.NONE)
        assert occ["num_bands"] == 4
        assert occ["exclude_bands"] == list(range(5, 21))
        assert emp["num_wann"] == 4
        assert emp["num_bands"] == 16  # 4 Wannier bands + 12 extra bands
        assert get_wannier_indices(emp) == [5, 6, 7, 8]
        assert emp["exclude_bands"] == [1, 2, 3, 4]

    def test_disentangling_block_preserves_the_wann2kcp_band_identity(
        self, si_structure: Any
    ) -> None:
        """Every block satisfies ``len(exclude_bands) + num_bands == nbnd``.

        wann2kcp.x reads the ``.chk`` against the pw.x band count and rejects
        any block whose excluded and read bands do not add back up to it.
        """
        from aiida_koopmans.spin import SpinChannel

        sp = [Projection(site="Si", ang_mtm="l=-1")]  # 4
        for nbnd in (8, 12, 20):
            for block in _dscf_blocks(si_structure, [sp, sp], 4, nbnd, SpinChannel.NONE):
                excluded = block["exclude_bands"] or []
                assert len(excluded) + block["num_bands"] == nbnd

    def test_occ_emp_split_and_exclusions(self, si_structure: Any) -> None:
        """Two sp blocks split into occ_1 (bands 1-4) and emp_1 (5-8)."""
        from aiida_koopmans.spin import SpinChannel

        sp = [Projection(site="Si", ang_mtm="l=-1")]  # 2 orbitals x 2 sites = 4
        blocks = _dscf_blocks(si_structure, [sp, sp], 4, 8, SpinChannel.NONE)
        assert [b["label"] for b in blocks] == ["occ_1", "emp_1"]
        assert get_wannier_indices(blocks[0]) == [1, 2, 3, 4]
        assert blocks[0]["exclude_bands"] == [5, 6, 7, 8]
        assert get_wannier_indices(blocks[1]) == [5, 6, 7, 8]
        assert blocks[1]["exclude_bands"] == [1, 2, 3, 4]

    def test_middle_block_gets_two_sided_exclusion(self, si_structure: Any) -> None:
        """A block sandwiched between others excludes bands on both sides."""
        from aiida_koopmans.spin import SpinChannel

        s = [Projection(site="Si", ang_mtm="l=0")]  # 1 x 2 sites = 2
        sp = [Projection(site="Si", ang_mtm="l=-1")]  # 4
        blocks = _dscf_blocks(si_structure, [s, s, sp], 4, 8, SpinChannel.NONE)
        assert [b["label"] for b in blocks] == ["occ_1", "occ_2", "emp_1"]
        assert blocks[1]["exclude_bands"] == [1, 2, 5, 6, 7, 8]

    def test_straddling_block_raises(self, si_structure: Any) -> None:
        """A block crossing the occupied/empty boundary is an input error."""
        from aiida_koopmans.spin import SpinChannel

        sp = [Projection(site="Si", ang_mtm="l=-1")]  # 4
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
        from aiida_koopmans.spin import SpinChannel

        sp = [Projection(site="Si", ang_mtm="l=-1")]  # 4
        with pytest.raises(ValueError, match="span 8 bands but nbnd = 6"):
            _dscf_blocks(si_structure, [sp, sp], 6, 6, SpinChannel.NONE)

    def test_disentanglement_above_an_occupied_block_raises(self, si_structure: Any) -> None:
        """Occupied-only projections must not disentangle against empty bands.

        The extra bands would land on the topmost *occupied* block, whose
        Wannier functions seed the occupied manifold of the supercell kcp.x
        run; letting them mix in empty character corrupts that seed with
        nothing downstream to catch it.
        """
        from aiida_koopmans.spin import SpinChannel

        sp = [Projection(site="Si", ang_mtm="l=-1")]  # 4 = nocc
        with pytest.raises(ValueError, match="for disentanglement"):
            _dscf_blocks(si_structure, [sp], 4, 8, SpinChannel.NONE)

    def test_boundary_check_alone_rejects_disentanglement_that_crosses(
        self, si_structure: Any
    ) -> None:
        """The boundary check rejects extra bands crossing it, reading no indices.

        This block's own bands are the four occupied ones — a check that
        compares its Wannier-function indices against the boundary sees
        nothing wrong with it, and only the extra bands it reads for
        disentanglement reach into the empty manifold.
        Asking the plugin whether the block is occupied is what catches it,
        so run that check on its own: paired with the coverage check it
        would pass for the wrong reason.
        """
        from aiida_koopmans.spin import SpinChannel

        from koopmans.aiida.workflows.blocks import (
            create_explicit_blocks,
            validate_blocks_separate_occ_and_emp,
        )

        # The stand-in projections duck-type the pydantic model the
        # derivation reads (``.site`` / ``.ang_mtm``), which is all it touches.
        sp = [Projection(site="Si", ang_mtm="l=-1")]  # 4 = nocc
        blocks = create_explicit_blocks(si_structure, [sp], 8, 4, SpinChannel.NONE)
        assert get_wannier_indices(blocks[0]) == [1, 2, 3, 4]  # entirely occupied
        with pytest.raises(ValueError, match="for disentanglement"):
            validate_blocks_separate_occ_and_emp(blocks, 4)

    def test_uncovered_occupied_bands_raise(self, si_structure: Any) -> None:
        """Occupied blocks must cover every occupied band.

        The nscf stops at the occupied manifold, so the single block is
        occupied outright and the boundary check has nothing to say; what is
        wrong is that two Wannier functions cannot seed four occupied bands.
        """
        from aiida_koopmans.spin import SpinChannel

        s = [Projection(site="Si", ang_mtm="l=0")]  # 2 wann < nocc 4
        with pytest.raises(ValueError, match="every occupied band"):
            _dscf_blocks(si_structure, [s], 4, 4, SpinChannel.NONE)

    def test_blocks_beyond_nbnd_raise(self, si_structure: Any) -> None:
        """Blocks spanning more bands than nbnd are an input error."""
        from aiida_koopmans.spin import SpinChannel

        sp = [Projection(site="Si", ang_mtm="l=-1")]
        with pytest.raises(ValueError, match="nbnd"):
            _dscf_blocks(si_structure, [sp, sp], 4, 6, SpinChannel.NONE)

    def test_blocks_carry_their_occupancy(self, si_structure: Any) -> None:
        """Each block states which manifold it belongs to.

        Downstream the merge places a block by this stamp, so a derivation
        that only named the block ``occ_1`` would leave the occupancy to be
        re-read off the band indices — which is exactly what the plugin
        stopped doing.
        """
        from aiida_koopmans.spin import SpinChannel

        sp = [Projection(site="Si", ang_mtm="l=-1")]  # 4
        blocks = _dscf_blocks(si_structure, [sp, sp], 4, 8, SpinChannel.NONE)
        assert [b["filled"] for b in blocks] == [True, False]


class TestPeriodicMlwfsBuild:
    """Graph-construction tests for the Wannier-initialised DSCF route."""

    def test_wannier_route_builds(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The periodic mlwfs input builds the Wannier-seeded workgraph."""
        wg = _build(_si_dscf_dict())
        names = wg.get_task_names()
        assert "wannier_initialization" in names, names
        assert "make_supercell" in names, names
        # The molecular KS-init steps must NOT be present.
        assert "dft_init_nspin1" not in names

    def test_disentangling_input_builds_and_validates(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """An nscf with headroom above the Wannier manifold is a buildable input.

        ``nbnd`` sets the kcp.x orbital count and ``pw.system.nbnd`` the nscf
        band count; the eight bands between them are the uppermost block's
        extra disentanglement bands. Building is not enough to call this wired —
        ``check_before_run`` is what proves every task of the assembled graph
        has the inputs it declares as required.
        """
        d = _si_dscf_dict()
        d["calculator_parameters"]["pw"] = {"system": {"nbnd": 16}}
        wg = _build(d)
        assert "wannier_initialization" in wg.get_task_names()
        wg.check_before_run()

    def test_out_of_order_blocks_rejected_at_build_time(
        self,
        aiida_profile: Any,
        dscf_codes: Any,
        fake_sg15_pseudo_family: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A block set violating the sequence invariants dies before submission.

        The block builder emits ascending blocks by construction, so the
        derivation cannot produce this set today; the plugin validator backs
        that structural guarantee with a check, and this pins that the route
        runs it. The coverage check alone passes a reversed set — it counts
        occupied Wannier functions wherever the blocks sit in the list.
        """
        import koopmans.aiida.workflows.dscf as dscf_module
        from koopmans.aiida.workflows.blocks import create_explicit_blocks as derive

        def reversed_blocks(*args: Any, **kwargs: Any) -> Any:
            """Derive the real blocks, reversed — the layout only the validator rejects."""
            return list(reversed(derive(*args, **kwargs)))

        monkeypatch.setattr(dscf_module, "create_explicit_blocks", reversed_blocks)
        with pytest.raises(ValueError, match="ascending band order"):
            _build(_si_dscf_dict())

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
        wg = _build(d)
        kpoints = wg.tasks["wannier_initialization"].inputs["kpoints"].value
        assert list(kpoints.get_kpoints_mesh()[0]) == [4, 4, 4]

    def test_self_hartree_grouping_defaulted(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Wannier-initialised runs resolve to self-Hartree grouping at 1e-4 eV."""
        wg = _build(_si_dscf_dict())
        tol = wg.tasks["ComputeScreeningParameters"].inputs["self_hartree_tol"].value
        assert tol == pytest.approx(1.0e-4)

    def test_user_grouping_tol_wins(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """An explicit group_orbitals_tol overrides the criterion default."""
        d = _si_dscf_dict(group_orbitals_tol=0.05)
        wg = _build(d)
        tol = wg.tasks["ComputeScreeningParameters"].inputs["self_hartree_tol"].value
        assert tol == pytest.approx(0.05)

    def test_grouping_none_disables(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """group_orbitals_by='none' disables grouping even on the Wannier route."""
        d = _si_dscf_dict(group_orbitals_by="none")
        wg = _build(d)
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
            _build(d)

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
        extra = dscf_wannier_init_inputs(inp, structure, nbnd)
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
        extra = dscf_wannier_init_inputs(inp, structure, nbnd)
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
        extra = dscf_wannier_init_inputs(inp, structure, nbnd)
        assert [(b["label"], b["filled"]) for b in extra["blocks"]] == [
            ("occ_up_1", True),
            ("emp_up_1", False),
            ("occ_down_1", True),
            ("emp_down_1", False),
        ]

    def test_collinear_magnetization_reaches_the_pw_steps(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The shared magnetization lands in the scf and nscf SYSTEM namelists.

        Those two run at nspin = 2 with fixed occupations, so pw.x needs the
        moment; before it was threaded here the key was absent altogether and
        only the retired ``pw.system`` spelling could supply one.
        """
        from koopmans.aiida.conversion import atoms_input_to_structure

        inp = KoopmansInput.model_validate(_si_collinear_dscf_dict())
        structure = atoms_input_to_structure(inp.atoms)
        nbnd = inp.calculator_parameters.nbnd
        assert nbnd is not None
        overrides = dscf_wannier_init_inputs(inp, structure, nbnd)["wannier_overrides"]

        for step in ("scf", "nscf"):
            system = overrides[step]["pw"]["parameters"]["SYSTEM"]
            assert system["tot_magnetization"] == inp.calculator_parameters.tot_magnetization

    def test_unpolarized_pw_steps_state_no_magnetization(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """A ``spin: none`` run leaves the keyword out rather than stamping a moment.

        Pairs with the collinear case: together they show the value is read
        from the input in the collinear branch, not written unconditionally.
        """
        from koopmans.aiida.conversion import atoms_input_to_structure

        inp = KoopmansInput.model_validate(_si_dscf_dict())
        structure = atoms_input_to_structure(inp.atoms)
        nbnd = inp.calculator_parameters.nbnd
        assert nbnd is not None
        overrides = dscf_wannier_init_inputs(inp, structure, nbnd)["wannier_overrides"]

        for step in ("scf", "nscf"):
            assert "tot_magnetization" not in overrides[step]["pw"]["parameters"]["SYSTEM"]

    @pytest.mark.parametrize(
        ("input_dict", "expected"),
        [(_si_collinear_dscf_dict(), True), (_si_dscf_dict(), False)],
        ids=["collinear", "none"],
    )
    def test_the_stated_regime_reaches_the_wannier_initialization(
        self,
        aiida_profile: Any,
        dscf_codes: Any,
        fake_sg15_pseudo_family: Any,
        input_dict: dict[str, Any],
        expected: bool,
    ) -> None:
        """The regime reaches the sub-workflow that runs the ground state.

        Downstream of this socket the plugin turns it into the ``spin_type``
        its scf and nscf are built from, so a run whose regime stops here
        Wannierizes spin-resolved blocks off an unpolarized density — and
        the magnetization above lands in a namelist with no ``nspin``, which
        pw.x refuses outright. Both cases together show the flag is read
        from the input rather than stamped.
        """
        task = _build(input_dict).tasks["wannier_initialization"]
        # A graph input is a proxy, for which ``is`` against a bool is false.
        assert task.inputs["spin_polarized"].value == expected

    def test_collinear_route_builds(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Two spin channels of blocks assemble into a runnable graph.

        The occupancies the route now states are consumed at build time by
        the Wannierization, which orders the orbitals by channel and then by
        filling; ``check_before_run`` is what proves the assembled graph has
        every input it declares as required.
        """
        wg = _build(_si_collinear_dscf_dict())
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
        extra = dscf_wannier_init_inputs(inp, structure, nbnd)
        assert set(extra["wannier_overrides"]) == {"scf", "nscf"}


class TestKcpNbndComesFromTheProjections:
    """On the Wannier route the input file's ``nbnd`` sizes pw.x, not kcp.x.

    kcp.x runs one variational orbital per projected Wannier function: the
    merged evc_occupied/evc_empty files carry exactly those. The top-level
    ``nbnd`` is the pw.x band count the blocks disentangle among, which is
    normally larger.
    """

    def test_the_kcp_step_takes_the_wannier_count_and_the_nscf_the_input_one(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """20 bands for pw.x, 8 variational orbitals for kcp.x, from one input.

        The two numbers must differ here: an implementation that fed one
        of them to both steps would agree with only half of this.
        """
        d = _si_dscf_dict()
        d["calculator_parameters"]["nbnd"] = 20
        wg = _build(d)
        assert wg.inputs["nbnd"].value == 8
        # The kcp.x runs happen on the diag(2,2,2) supercell, where the
        # same 8 Wannier functions become 8 x 8 orbitals.
        assert wg.tasks["wannier_initialization"].inputs["nbnd"].value == 64
        nscf = wg.tasks["wannier_initialization"].inputs["wannier_overrides"]["nscf"].value
        assert nscf["pw"]["parameters"]["SYSTEM"]["nbnd"] == 20

    def test_the_disentanglement_window_is_sized_from_the_input_nbnd(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The blocks index the nscf bands, so 20 bands means an empty block of 20.

        Sizing the blocks by the kcp.x orbital count instead would leave
        the bands above it neither included nor excluded, and wannier90
        rejects the mmn it is then handed.
        """
        d = _si_dscf_dict()
        d["calculator_parameters"]["nbnd"] = 20
        wg = _build(d)
        blocks = wg.tasks["wannier_initialization"].inputs["blocks"].value
        assert [b["num_bands"] for b in blocks] == [4, 16]
        assert [b["num_wann"] for b in blocks] == [4, 4]

    def test_the_shipped_silicon_tutorial_builds_as_a_singlepoint(
        self,
        aiida_profile: Any,
        dscf_codes: Any,
        fake_sg15_cutoffs_family: Any,
        tutorials_dir: Any,
    ) -> None:
        """The tutorial's own projections and band count reach the two steps that need them.

        The shipped input Wannierizes 20 pw.x bands and projects two sp3
        blocks (4 Wannier functions each). Only the singlepoint keywords
        the tutorial names as the KI variant are overlaid; the
        projections, windows and band count are the tutorial's.
        """
        import json

        si_json = tutorials_dir / "band_structures/silicon_finite_differences/si.json"
        d = json.loads(si_json.read_text())
        d["workflow"].update(
            {
                "task": "singlepoint",
                "correction": "ki",
                "screening_method": "dscf",
                "init_orbitals": "mlwfs",
                "alpha_guess": 0.077,
                "mp_correction": False,
                "pseudo_library": fake_sg15_cutoffs_family.label,
            }
        )
        d["calculator_parameters"]["nbnd"] = 20

        wg = build_singlepoint_workgraph(KoopmansInput.model_validate(d))

        assert wg.inputs["nbnd"].value == 8
        nscf = wg.tasks["wannier_initialization"].inputs["wannier_overrides"]["nscf"].value
        assert nscf["pw"]["parameters"]["SYSTEM"]["nbnd"] == 20

    def test_a_supercell_sized_kcp_nbnd_is_rejected(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Regression for koopmans#163, at the keyword that now states the count.

        The primitive projections describe 8 Wannier functions
        (occ_1 + emp_1, 4 each); writing the supercell count
        (8 x prod(kgrid) = 64) into ``kcp.system.nbnd`` is the koopmans#163
        mistake, and must be named directly rather than surfacing as an
        nscf shortfall.
        """
        d = _si_dscf_dict()
        d["calculator_parameters"]["kcp"] = {"system": {"nbnd": 64}}
        with pytest.raises(ValueError, match="inconsistent with the") as excinfo:
            _build(d)
        assert "nscf runs" not in str(excinfo.value)

    def test_a_matching_kcp_nbnd_is_accepted(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Negative control: it is the disagreement that raises, not the keyword."""
        d = _si_dscf_dict()
        d["calculator_parameters"]["kcp"] = {"system": {"nbnd": 8}}
        assert _build(d).inputs["nbnd"].value == 8

    def test_an_input_nbnd_below_the_wannier_count_is_an_nscf_shortfall(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Too few pw.x bands to span the Wannier functions is named as such."""
        d = _si_dscf_dict()
        d["calculator_parameters"]["nbnd"] = 2
        with pytest.raises(ValueError, match=r"The nscf runs 2 bands but the kcp\.x steps need 8"):
            _build(d)

    def test_spin_channels_describing_different_counts_are_refused(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """One kcp.x run, one orbital count: the channels cannot disagree."""
        d = _si_collinear_dscf_dict()
        d["calculator_parameters"]["wannier90"]["down"]["projections"] = [
            [{"site": "Si", "ang_mtm": "sp"}],
            [{"site": "Si", "ang_mtm": "sp"}],
            [{"site": "Si", "ang_mtm": "l=0"}],
        ]
        with pytest.raises(ValueError, match="spin up projections describe 8"):
            _build(d)

    def test_genuine_nscf_shortfall_still_raises_pw_guard(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """A correct nbnd with too few nscf bands still hits the existing pw guard.

        ``nbnd`` = 8 matches the projections exactly, so the new check
        passes; only ``pw.system.nbnd`` is too small, and the nscf-sizing
        guard is what must catch that.
        """
        d = _si_dscf_dict()
        d["calculator_parameters"]["pw"] = {"system": {"nbnd": 6}}
        with pytest.raises(ValueError, match=r"The nscf runs 6 bands but the kcp\.x steps need 8"):
            _build(d)

    def test_a_collinear_shortfall_names_the_spin_channel(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Projections are checked, and reported, per spin channel."""
        d = _si_collinear_dscf_dict()
        d["calculator_parameters"]["wannier90"]["up"]["projections"] = [
            [{"site": "Si", "ang_mtm": "l=0"}]
        ]
        with pytest.raises(ValueError, match="spin up projections"):
            _build(d)

    def test_collinear_genuine_nscf_shortfall_still_raises_pw_guard(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The nscf guard still fires for a genuine shortfall on the collinear branch.

        Both channels' projections match ``nbnd`` = 8 exactly, so the new
        per-channel checks pass; only ``pw.system.nbnd`` is too small.
        """
        d = _si_collinear_dscf_dict()
        d["calculator_parameters"]["pw"] = {"system": {"nbnd": 6}}
        with pytest.raises(ValueError, match=r"The nscf runs 6 bands but the kcp\.x steps need 8"):
            _build(d)

    def test_projections_short_of_occupied_bands_names_the_shortfall(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Projections that undercount the occupied manifold report that directly.

        ``nocc`` = 4, but this input's single occupied-only block covers
        only 2 bands, leaving `nbnd` = 8 to compare against 2 Wannier
        functions. The message must name the shortfall, not print the
        negative "empty Wannier functions" count that a plain nbnd/nwann
        comparison would produce here.
        """
        d = _si_dscf_dict()
        d["calculator_parameters"]["wannier90"]["projections"] = [
            [{"site": "Si", "ang_mtm": "l=0"}]
        ]
        with pytest.raises(ValueError, match=r"describe only 2 Wannier functions") as excinfo:
            _build(d)
        assert "-" not in str(excinfo.value)


class TestFrozenWindowThreading:
    """The disentanglement window reaches the wannier-initialization inputs."""

    def test_frozen_window_reaches_the_wannier_initialization(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """``dis_froz_max`` lands on the route's wannier_overrides socket.

        Regression for koopmans#94: the seam between the input file and
        the plugin's per-block builders — a keyword lost here disentangles
        unfrozen and silently shifts the folded empty manifold.
        """
        d = _si_dscf_dict()
        d["calculator_parameters"]["pw"] = {"system": {"nbnd": 16}}
        d["calculator_parameters"]["wannier90"]["dis_froz_max"] = 1.0
        wg = _build(d)

        init = next(t for t in wg.tasks if t.name == "wannier_initialization")
        overrides = init.inputs.wannier_overrides["wannier90"].value
        assert overrides["dis_froz_max"] == 1.0


class TestPerStepKpointMeshRejected:
    """The kcp.x route runs every step on one mesh, so it takes no per-step entry.

    Its kcp.x steps recompute the ground state in the supercell that
    ``kpoints.grid`` describes. An scf on another mesh would be a ground
    state the Wannier functions did not come from, and the mismatch would
    show up nowhere in the run.
    """

    @pytest.mark.parametrize("step", ["scf", "nscf"])
    def test_either_entry_raises_naming_the_grid(self, step: str) -> None:
        """The message sends the reader to the one keyword this route reads.

        The guard runs before any code or pseudopotential is loaded, so it
        needs no profile.
        """
        d = _si_dscf_dict()
        d["kpoints"]["overrides"] = {step: {"grid": [4, 4, 4]}}
        with pytest.raises(ValueError, match=rf"overrides\.{step}.*`kpoints.grid`"):
            _build(d)

    def test_wannier90_density_raises(self) -> None:
        """No interpolated band structure exists here for a density to describe.

        The Wannier initialisation folds Wannier functions to a supercell,
        not an interpolated band structure along a path.
        """
        d = _si_dscf_dict()
        d["kpoints"]["overrides"] = {"wannier90": {"path_density": 25.0}}
        with pytest.raises(ValueError, match=r"overrides\.wannier90\.path_density.*kcp\.x"):
            _build(d)


class TestBandPathRejected:
    """Where a ΔSCF band path still has nowhere to go."""

    def test_a_molecular_kohn_sham_path_is_rejected(self, read_input_dict: Any) -> None:
        """A molecule has no band structure, whatever the interpolation can do.

        Refused while the input file is read, so the reader gets the error
        report rather than a traceback out of the graph build.
        """
        d = _si_dscf_dict(init_orbitals="kohn-sham")
        d["atoms"]["cell_parameters"] = {
            "periodic": False,
            "units": "angstrom",
            "vectors": [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        }
        d["kpoints"]["path"] = "GX"

        with pytest.raises(ValueError) as excinfo:
            read_input_dict(d)

        message = str(excinfo.value)
        assert "Errors found in the input file" in message
        assert "`kpoints.path`" in message
        assert "periodic along no direction" in message

    def test_the_same_input_periodic_and_wannier_initialized_parses(
        self, read_input_dict: Any
    ) -> None:
        """Discriminates the refusal from a blanket one on the kcp.x route.

        The periodic Wannier-initialised route interpolates the same path,
        so only the molecular input may be refused.
        """
        d = _si_dscf_dict()
        d["kpoints"]["path"] = "GX"

        read_input_dict(d)

    def test_a_periodic_kohn_sham_input_hears_its_own_blocker_first(
        self, aiida_profile_clean: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Pins the refusal behind the initialisation-route check.

        kcp.x runs no periodic ``kohn-sham`` route at all, and refusing the
        band path first would send the reader to ``screening_method =
        'dfpt'`` — which refuses the same input again, for wanting Wannier
        orbitals. So this input must parse, and the blocker must arrive on
        the first hop when it is built.
        """
        d = _si_dscf_dict(init_orbitals="kohn-sham")
        d["kpoints"]["path"] = "GX"

        with pytest.raises(NotImplementedError) as excinfo:
            _build(d)

        message = str(excinfo.value)
        assert "init_orbitals='kohn-sham' on a periodic structure" in message
        assert "`kpoints.path`" not in message


class TestCutoffLessPseudoFamily:
    """A family recommending no cutoffs drives this route's pw steps from the input.

    The Wannier initialisation builds its own scf and nscf overrides rather
    than going through ``prepare_common_inputs``, so it carries the cutoff
    check of its own. The checks drive the built graph's override entries
    into the pw protocol builder, which is what the steps do when they run.
    """

    @staticmethod
    def _si_structure(d: dict[str, Any]) -> Any:
        from koopmans.aiida.conversion import atoms_input_to_structure

        return atoms_input_to_structure(KoopmansInput.model_validate(d).atoms)

    @pytest.mark.parametrize("step", ["scf", "nscf"])
    def test_the_step_takes_the_input_cutoffs_and_the_family_pseudos(
        self,
        aiida_profile_clean: Any,
        dscf_codes: Any,
        fake_sg15_family_without_cutoffs: Any,
        step: str,
    ) -> None:
        """Both cutoffs and the family's own pseudos reach the pw.x calculation.

        The family recommends nothing, so 20 Ry and its derived 80 Ry can only
        have come from the input; the pseudo uuids say the builder resolved
        them against the named family rather than some other one.
        """
        from tests.fixtures import pw_step_from_overrides

        d = _si_dscf_dict(pseudo_library=fake_sg15_family_without_cutoffs.label)
        wg = _build(d)
        structure = self._si_structure(d)

        entry = wg.tasks["wannier_initialization"].inputs["wannier_overrides"][step].value
        pw = pw_step_from_overrides(dscf_codes["pw"], structure, entry)

        assert pw.parameters["SYSTEM"]["ecutwfc"] == pytest.approx(20.0)
        assert pw.parameters["SYSTEM"]["ecutrho"] == pytest.approx(80.0)
        expected = fake_sg15_family_without_cutoffs.get_pseudos(structure=structure)
        assert pw.pseudos["Si"].uuid == expected["Si"].uuid


class TestBandPathBuildsTheInterpolation:
    """``kpoints.path`` gates the unfold-and-interpolate stage.

    Without a path the ΔSCF route finishes on a supercell and there is
    nothing to plot, which is the state these tests discriminate against.
    """

    @staticmethod
    def _with_bands(**workflow_updates: Any) -> dict[str, Any]:
        d = _si_dscf_dict(**workflow_updates)
        d["kpoints"]["path"] = "GXG"
        return d

    def test_the_stage_is_absent_without_a_path(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Negative control: the same input naming no path builds no interpolation."""
        wg = _build(_si_dscf_dict())
        assert "interpolate_band_structure" not in wg.get_task_names()
        assert wg.tasks["RunFinalKI"].inputs["write_hr"].value == False  # noqa: E712

    def test_a_path_builds_the_interpolation(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The path adds the interpolation stage and its one required input."""
        wg = _build(self._with_bands())
        names = wg.get_task_names()
        assert "interpolate_band_structure" in names, names
        # The Hamiltonians the stage reads only exist because the final KI
        # is asked to print them.
        assert wg.tasks["RunFinalKI"].inputs["write_hr"].value == True  # noqa: E712

    def test_the_input_path_reaches_the_interpolation(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Nothing downstream can recover a path the dispatcher does not hand over."""
        d = self._with_bands()
        d["kpoints"]["path"] = "GX"
        wg = _build(d)
        kpath = wg.tasks["interpolate_band_structure"].inputs["kpath"].value
        assert [label for _, label in kpath.labels] == ["GAMMA", "X"]

    def test_the_knobs_reach_the_interpolation(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The `unfold_and_interpolate` block shapes the stage that runs."""
        d = self._with_bands()
        d["calculator_parameters"]["unfold_and_interpolate"] = {
            "use_ws_distance": False,
            "do_dos": False,
        }
        wg = _build(d)
        stage = wg.tasks["interpolate_band_structure"]
        assert stage.inputs["use_ws_distance"].value == False  # noqa: E712
        assert stage.inputs["do_dos"].value == False  # noqa: E712

    def test_the_knobs_alone_are_refused(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Settings that shape a stage no path asks for cannot take effect."""
        d = _si_dscf_dict()
        d["calculator_parameters"]["unfold_and_interpolate"] = {"do_dos": False}
        with pytest.raises(ValueError, match=r"kpoints: \{path"):
            _build(d)

    def test_smooth_interpolation_is_refused_by_name(
        self, aiida_profile: Any, dscf_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """An input that cannot take effect raises rather than being dropped."""
        d = self._with_bands()
        d["calculator_parameters"]["unfold_and_interpolate"] = {"smooth_int_factor": 2}
        with pytest.raises(NotImplementedError, match="smooth_int_factor"):
            _build(d)
