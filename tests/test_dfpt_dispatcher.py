"""Dispatcher tests for the DFPT (kcw.x) singlepoint stream.

Builds real ``WorkGraph`` objects through ``_build_singlepoint_dfpt_workgraph``
against a throwaway profile (dummy codes, fake pseudos; nothing runs) and
checks the spin routing: unpolarized, collinear (per-channel fan-out), and
spinor (noncollinear / spin-orbit).
"""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.aiida.workflows import _build_singlepoint_dfpt_workgraph
from koopmans.input_file import KoopmansInput


def _si_dfpt_dict(**workflow_updates: Any) -> dict[str, Any]:
    """Return a minimal silicon DFPT input dict (fake SG15 pseudos: Si z=4)."""
    d: dict[str, Any] = {
        "workflow": {
            "task": "singlepoint",
            "correction": "ki",
            "screening_method": "dfpt",
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
            "wannier90": {
                # One occupied block: sp hybrids (2 orbitals) on each of the
                # 2 Si sites = 4 Wannier functions = nocc (nelec 8, fake
                # z_valence 4).
                "projections": [[{"site": "Si", "ang_mtm": "sp"}]],
            },
        },
    }
    d["workflow"].update(workflow_updates)
    return d


def _build(d: dict[str, Any], codes: dict[str, Any]) -> Any:
    inp = KoopmansInput.model_validate(d)
    return _build_singlepoint_dfpt_workgraph(inp, codes=codes)


@pytest.fixture
def dfpt_codes(
    installed_pw_code: Any, installed_kcw_code: Any, installed_wannier_codes: Any
) -> dict[str, Any]:
    """Assemble the full DFPT code dict from the dummy-code fixtures."""
    return {
        "pw": installed_pw_code,
        "kcw": installed_kcw_code,
        **installed_wannier_codes,
    }


@pytest.fixture
def fake_pseudodojo_lda_family(aiida_profile: Any) -> Any:
    """Install a minimal fake ``PseudoDojo/0.4/LDA/SR/standard/upf`` family.

    Zn (z=20) and O (z=6) synthetic UPF streams — enough for the dispatcher's
    electron counting, not physically meaningful pseudos. Local so the ZnO
    test does not fight over the shared fixtures module.
    """
    import io

    from aiida.common.exceptions import NotExistent
    from aiida_pseudo.data.pseudo.upf import UpfData
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    label = "PseudoDojo/0.4/LDA/SR/standard/upf"
    try:
        return PseudoPotentialFamily.collection.get(label=label)
    except NotExistent:
        pass

    family = PseudoPotentialFamily(label=label, description="fake PseudoDojo family for tests")
    family.store()
    for element, z_valence in (("Zn", 20.0), ("O", 6.0)):
        content = (
            f'<UPF version="2.0.1"><PP_HEADER\nelement="{element}"\n'
            f'z_valence="{z_valence}"\n/></UPF>\n'
        )
        upf = UpfData(io.BytesIO(content.encode("utf-8")), filename=f"{element}.upf")
        family.add_nodes([upf.store()])
    return family


class TestUnpolarized:
    """spin='none' builds the closed-shell single chain."""

    def test_single_chain(
        self, aiida_profile: Any, dfpt_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """One kcw chain, no per-channel task suffixes."""
        wg = _build(_si_dfpt_dict(), dfpt_codes)
        names = wg.get_task_names()
        assert "wannierize_occ" in names
        assert "dfpt" in names
        assert "dfpt_up" not in names
        assert "dfpt_down" not in names


class TestCollinear:
    """spin='collinear' fans out per spin channel and validates its inputs."""

    def _collinear_dict(self) -> dict[str, Any]:
        d = _si_dfpt_dict(spin="collinear")
        per_spin = {"projections": [[{"site": "Si", "ang_mtm": "sp"}]]}
        d["calculator_parameters"]["wannier90"]["up"] = per_spin
        d["calculator_parameters"]["wannier90"]["down"] = per_spin
        d["calculator_parameters"]["tot_magnetization"] = 0
        return d

    def test_fans_out_per_channel(
        self, aiida_profile: Any, dfpt_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """One shared scf+nscf; wannierize + kcw chain per channel."""
        wg = _build(self._collinear_dict(), dfpt_codes)
        names = wg.get_task_names()
        assert names.count("scf_nscf") == 1
        for expected in ("wannierize_occ_up", "dfpt_up", "wannierize_occ_down", "dfpt_down"):
            assert expected in names, names

        # The magnetization reaches the PW SYSTEM namelist alongside the
        # forced nspin=2.
        pw_overrides = wg.tasks["scf_nscf"].inputs["overrides"].value
        scf_system = pw_overrides["scf"]["pw"]["parameters"]["SYSTEM"]
        assert scf_system["nspin"] == 2
        assert scf_system["tot_magnetization"] == 0

    def test_missing_per_spin_projections_raises(self, dfpt_codes: Any) -> None:
        """Collinear DFPT requires w90.up / w90.down projections."""
        d = _si_dfpt_dict(spin="collinear")
        d["calculator_parameters"]["tot_magnetization"] = 0
        with pytest.raises(ValueError, match="per-spin projections"):
            _build(d, dfpt_codes)

    def test_missing_magnetization_raises(self, dfpt_codes: Any) -> None:
        """Collinear DFPT requires tot_magnetization."""
        d = _si_dfpt_dict(spin="collinear")
        per_spin = {"projections": [[{"site": "Si", "ang_mtm": "sp"}]]}
        d["calculator_parameters"]["wannier90"]["up"] = per_spin
        d["calculator_parameters"]["wannier90"]["down"] = per_spin
        with pytest.raises(ValueError, match="tot_magnetization"):
            _build(d, dfpt_codes)

    def test_non_integer_channel_occupations_raise(
        self, aiida_profile: Any, dfpt_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Nelec + tot_magnetization must be even."""
        d = self._collinear_dict()
        d["calculator_parameters"]["tot_magnetization"] = 1  # nelec=8 -> 4.5/3.5
        with pytest.raises(ValueError, match="integer per-channel occupations"):
            _build(d, dfpt_codes)


class TestMultiBlockZnO:
    """The ZnO tutorial shape: four occupied blocks + one disentangled empty block.

    Mirrors ``docs/source/tutorials/zno.json`` in the ``KoopmansInput`` schema
    with the band path dropped, exercising the multi-block manifold routing
    only. Fake PseudoDojo pseudos: Zn z=20, O z=6 → nelec 52, nocc 26.
    """

    def _zno_dict(self) -> dict[str, Any]:
        return {
            "workflow": {
                "task": "singlepoint",
                "correction": "ki",
                "screening_method": "dfpt",
                "init_orbitals": "mlwfs",
                "calculate_alpha": False,
                # One alpha per orbital: 26 occupied + 2 empty (the flattened
                # per-orbital guess; the schema takes flat lists).
                "alpha_guess": [0.3] * 26 + [0.22] * 2,
                "pseudo_library": "PseudoDojo/0.4/LDA/SR/standard/upf",
                "gb_correction": True,
                "eps_inf": 5.3,
            },
            "atoms": {
                "cell_parameters": {
                    "periodic": True,
                    "ibrav": 4,
                    "celldms": {"1": 6.14057, "3": 1.60204},
                },
                "atomic_positions": {
                    "units": "crystal",
                    "positions": [
                        ["Zn", 0.33330, 0.66670, 0.50000],
                        ["Zn", 0.66670, 0.33330, 0.00000],
                        ["O", 0.33330, 0.66670, 0.11725],
                        ["O", 0.66670, 0.33330, 0.61725],
                    ],
                },
            },
            "kpoints": {"grid": [4, 4, 4], "offset": [0, 0, 0]},
            "calculator_parameters": {
                "ecutwfc": 50.0,
                "pw": {"system": {"nbnd": 52}},
                "wannier90": {
                    # Occupied: Zn s (2) + Zn p (6) + O s (2) + [Zn d + O p]
                    # (16) = 26 Wannier functions; empty: Zn s (2) over the
                    # 26 remaining bands (disentangled).
                    "projections": [
                        [{"site": "Zn", "ang_mtm": "l=0"}],
                        [{"site": "Zn", "ang_mtm": "l=1"}],
                        [{"site": "O", "ang_mtm": "l=0"}],
                        [{"site": "Zn", "ang_mtm": "l=2"}, {"site": "O", "ang_mtm": "l=1"}],
                        [{"site": "Zn", "ang_mtm": "l=0"}],
                    ],
                },
            },
        }

    def test_multi_block_manifolds(
        self, aiida_profile: Any, dfpt_codes: Any, fake_pseudodojo_lda_family: Any
    ) -> None:
        """Every occupied block wannierizes separately; the kcw chain sees totals."""
        wg = _build(self._zno_dict(), dfpt_codes)
        names = wg.get_task_names()
        for expected in (
            "wannierize_occ_1",
            "wannierize_occ_2",
            "wannierize_occ_3",
            "wannierize_occ_4",
            "wannierize_emp",
            "dfpt",
        ):
            assert expected in names, names
        assert names.count("scf_nscf") == 1

        dfpt_inputs = wg.tasks["dfpt"].inputs
        assert dfpt_inputs["num_wann_occ"].value == 26
        assert dfpt_inputs["num_wann_emp"].value == 2
        assert dfpt_inputs["nbnd_emp"].value == 26
        # ``is True`` would fail: socket values are TaggedValue proxies.
        assert bool(dfpt_inputs["has_disentangle"].value)
        # calculate_alpha=False: the flat 28-entry guess skips the screen step.
        assert list(dfpt_inputs["alpha_guess"].value) == [0.3] * 26 + [0.22] * 2


class TestSpinor:
    """Noncollinear / spin-orbit builds run one spinor chain."""

    @pytest.mark.parametrize("spin_value", ["non_collinear", "spin_orbit"])
    def test_single_spinor_chain(
        self, aiida_profile: Any, dfpt_codes: Any, fake_sg15_pseudo_family: Any, spin_value: str
    ) -> None:
        """Single chain with noncolin (+ lspinorb for SOC) forced on the PW runs."""
        # Spinor manifold: the sp block doubles to 8 spinor Wannier
        # functions, matching nocc = nelec = 8.
        wg = _build(_si_dfpt_dict(spin=spin_value), dfpt_codes)
        names = wg.get_task_names()
        assert "wannierize_occ" in names
        assert "dfpt" in names
        assert "dfpt_down" not in names

        pw_overrides = wg.tasks["scf_nscf"].inputs["overrides"].value
        scf_system = pw_overrides["scf"]["pw"]["parameters"]["SYSTEM"]
        assert scf_system["noncolin"] is True
        assert scf_system.get("lspinorb", False) is (spin_value == "spin_orbit")
        assert "nspin" not in scf_system


class TestOrbitalGrouping:
    """group_orbitals_by drives kcw.x's native (check_spread) grouping."""

    def test_default_groups_by_self_hartree(
        self, aiida_profile: Any, dfpt_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The wannier-init default (self_hartree, tol 1e-4) switches grouping on."""
        wg = _build(_si_dfpt_dict(), dfpt_codes)
        # `==` not `is`: graph inputs arrive as TaggedValue proxies.
        assert wg.tasks["dfpt"].inputs["group_orbitals"].value == True  # noqa: E712

    def test_none_disables_grouping(
        self, aiida_profile: Any, dfpt_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """group_orbitals_by='none' turns kcw.x's check_spread off."""
        wg = _build(_si_dfpt_dict(group_orbitals_by="none"), dfpt_codes)
        assert wg.tasks["dfpt"].inputs["group_orbitals"].value == False  # noqa: E712

    def test_non_default_tolerance_is_rejected(
        self, aiida_profile: Any, dfpt_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """kcw.x hardcodes its self-Hartree tolerance, so an override must not pass silently."""
        with pytest.raises(NotImplementedError, match="hardcoded"):
            _build(_si_dfpt_dict(group_orbitals_tol=1.0e-3), dfpt_codes)

    def test_spread_is_rejected(
        self, aiida_profile: Any, dfpt_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The spread criterion is not implemented on the DFPT route either."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            _build(_si_dfpt_dict(group_orbitals_by="spread"), dfpt_codes)


class TestWannier90Overrides:
    """User wannier90 keywords feed the per-manifold wannierization."""

    def test_keyword_reaches_wannierize_task(
        self, aiida_profile: Any, dfpt_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """A user ``num_iter`` overrides the wannierize builder default.

        The dispatcher folds the flat ``{'wannier90': {...}}`` override into
        the shared ``overrides``; the block wannierization builder then merges
        it over its own defaults, so the value surfaces on the wannierize
        task's ``overrides['wannier90']`` namespace socket.
        """
        d = _si_dfpt_dict()
        d["calculator_parameters"]["wannier90"]["num_iter"] = 17
        wg = _build(d, dfpt_codes)
        w90_overrides = wg.tasks["wannierize_occ"].inputs["overrides"]["wannier90"].value
        assert w90_overrides["num_iter"] == 17

    def test_no_keywords_omits_user_override(
        self, aiida_profile: Any, dfpt_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """With only projections set, the builder default ``num_iter`` stands.

        Discriminates the override path: absent a user keyword the wannierize
        task never sees ``num_iter = 17``.
        """
        wg = _build(_si_dfpt_dict(), dfpt_codes)
        w90_overrides = wg.tasks["wannierize_occ"].inputs["overrides"]["wannier90"].value
        assert w90_overrides.get("num_iter") != 17
