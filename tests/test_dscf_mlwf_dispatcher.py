"""Dispatcher tests for the Wannier-initialised (periodic mlwfs) DSCF route.

Exercises ``_derive_dscf_blocks`` (pure bookkeeping) and builds real
``WorkGraph`` objects through ``_build_singlepoint_workgraph`` for a periodic
silicon input (throwaway profile, dummy codes; nothing runs).
"""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.aiida.workflows import _build_singlepoint_workgraph
from koopmans.input_file import KoopmansInput


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


def _build(d: dict[str, Any], codes: dict[str, Any]) -> Any:
    inp = KoopmansInput.model_validate(d)
    return _build_singlepoint_workgraph(inp, codes=codes)


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


class TestDeriveDscfBlocks:
    """Unit tests for the projection-block bookkeeping."""

    @pytest.fixture
    def si_structure(self, aiida_profile: Any) -> Any:
        """Return a bare silicon StructureData for projection counting."""
        from aiida.orm import StructureData

        cell = [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
        struct = StructureData(cell=cell, pbc=True)  # type: ignore[no-untyped-call]
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
            self.ang_mtm = TestDeriveDscfBlocks._FakeQuantumNumbers(l_value)

    def test_fractional_site_projections(self, si_structure: Any) -> None:
        """Point-hosted (bond-centred) projections derive and format.

        One fractional site hosts exactly one orbital set (sp3 -> 4 WFs) and
        renders as wannier90's ``f=x,y,z:<ang_mtm>`` form.
        """
        from aiida_koopmans.types import SpinChannel

        from koopmans.aiida.workflows import _derive_dscf_blocks

        sp3 = [self._FakeProjection(None, -3, fractional_site=[0.25, 0.25, 0.25])]
        blocks = _derive_dscf_blocks(si_structure, [sp3, sp3], 4, 20, SpinChannel.NONE)
        occ, emp = blocks
        assert occ["num_wann"] == 4
        assert occ["projections"] == ["f=0.25,0.25,0.25:l=-3"]
        assert emp["num_wann"] == 4
        assert emp["num_bands"] == 16  # disentanglement pool

    def test_last_block_absorbs_disentanglement_pool(self, si_structure: Any) -> None:
        """Extra bands above the last block become its disentanglement window.

        The uppermost block gets ``num_bands = num_wann + (nbnd - covered)``
        and excludes nothing above itself, so an entangled empty manifold can
        disentangle.
        """
        from aiida_koopmans.types import SpinChannel

        from koopmans.aiida.workflows import _derive_dscf_blocks

        sp = [self._FakeProjection("Si", -1)]  # 2 orbitals x 2 sites = 4
        blocks = _derive_dscf_blocks(si_structure, [sp, sp], 4, 20, SpinChannel.NONE)
        occ, emp = blocks
        # occupied block untouched: two-sided exclusion against all 20 bands
        assert occ["num_bands"] == 4
        assert occ["exclude_bands"] == list(range(5, 21))
        # last block: 4 target WFs + 12 pool bands, exclusion only below
        assert emp["num_wann"] == 4
        assert emp["num_bands"] == 16
        assert emp["include_bands"] == [5, 6, 7, 8]
        assert emp["exclude_bands"] == [1, 2, 3, 4]

    def test_occ_emp_split_and_exclusions(self, si_structure: Any) -> None:
        """Two sp blocks split into occ_1 (bands 1-4) and emp_1 (5-8)."""
        from aiida_koopmans.types import SpinChannel

        from koopmans.aiida.workflows import _derive_dscf_blocks

        sp = [self._FakeProjection("Si", -1)]  # 2 orbitals x 2 sites = 4
        blocks = _derive_dscf_blocks(si_structure, [sp, sp], 4, 8, SpinChannel.NONE)
        assert [b["label"] for b in blocks] == ["occ_1", "emp_1"]
        assert blocks[0]["include_bands"] == [1, 2, 3, 4]
        assert blocks[0]["exclude_bands"] == [5, 6, 7, 8]
        assert blocks[1]["include_bands"] == [5, 6, 7, 8]
        assert blocks[1]["exclude_bands"] == [1, 2, 3, 4]

    def test_middle_block_gets_two_sided_exclusion(self, si_structure: Any) -> None:
        """A block sandwiched between others excludes bands on both sides."""
        from aiida_koopmans.types import SpinChannel

        from koopmans.aiida.workflows import _derive_dscf_blocks

        s = [self._FakeProjection("Si", 0)]  # 1 x 2 sites = 2
        sp = [self._FakeProjection("Si", -1)]  # 4
        blocks = _derive_dscf_blocks(si_structure, [s, s, sp], 4, 8, SpinChannel.NONE)
        assert [b["label"] for b in blocks] == ["occ_1", "occ_2", "emp_1"]
        assert blocks[1]["exclude_bands"] == [1, 2, 5, 6, 7, 8]

    def test_straddling_block_raises(self, si_structure: Any) -> None:
        """A block crossing the occupied/empty boundary is an input error."""
        from aiida_koopmans.types import SpinChannel

        from koopmans.aiida.workflows import _derive_dscf_blocks

        sp = [self._FakeProjection("Si", -1)]  # 4
        with pytest.raises(ValueError, match="straddles"):
            _derive_dscf_blocks(si_structure, [sp, sp], 6, 8, SpinChannel.NONE)

    def test_uncovered_occupied_bands_raise(self, si_structure: Any) -> None:
        """Occupied blocks must cover every occupied band."""
        from aiida_koopmans.types import SpinChannel

        from koopmans.aiida.workflows import _derive_dscf_blocks

        s = [self._FakeProjection("Si", 0)]  # 2 wann < nocc 4
        with pytest.raises(ValueError, match="every occupied band"):
            _derive_dscf_blocks(si_structure, [s], 4, 8, SpinChannel.NONE)

    def test_blocks_beyond_nbnd_raise(self, si_structure: Any) -> None:
        """Blocks spanning more bands than nbnd are an input error."""
        from aiida_koopmans.types import SpinChannel

        from koopmans.aiida.workflows import _derive_dscf_blocks

        sp = [self._FakeProjection("Si", -1)]
        with pytest.raises(ValueError, match="nbnd"):
            _derive_dscf_blocks(si_structure, [sp, sp], 4, 6, SpinChannel.NONE)


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
