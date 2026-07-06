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


def _build(d: dict[str, Any], codes: dict[str, Any]):
    inp = KoopmansInput.model_validate(d)
    return _build_singlepoint_dfpt_workgraph(inp, codes=codes)


@pytest.fixture
def dfpt_codes(installed_pw_code, installed_kcw_code, installed_wannier_codes):
    """Assemble the full DFPT code dict from the dummy-code fixtures."""
    return {
        "pw": installed_pw_code,
        "kcw": installed_kcw_code,
        **installed_wannier_codes,
    }


@pytest.fixture
def fake_pseudodojo_lda_family(aiida_profile):
    """Install a minimal fake ``PseudoDojo/0.4/LDA/SR/standard/upf`` family.

    Zn (z=20) and O (z=6) synthetic UPF streams — enough for the dispatcher's
    electron counting, not physically meaningful pseudos. Local so the ZnO
    test doesn't fight over the shared fixtures module.
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

    def test_single_chain(self, aiida_profile, dfpt_codes, fake_sg15_pseudo_family):
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

    def test_fans_out_per_channel(self, aiida_profile, dfpt_codes, fake_sg15_pseudo_family):
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

    def test_missing_per_spin_projections_raises(self, dfpt_codes):
        """Collinear DFPT requires w90.up / w90.down projections."""
        d = _si_dfpt_dict(spin="collinear")
        d["calculator_parameters"]["tot_magnetization"] = 0
        with pytest.raises(ValueError, match="per-spin projections"):
            _build(d, dfpt_codes)

    def test_missing_magnetization_raises(self, dfpt_codes):
        """Collinear DFPT requires tot_magnetization."""
        d = _si_dfpt_dict(spin="collinear")
        per_spin = {"projections": [[{"site": "Si", "ang_mtm": "sp"}]]}
        d["calculator_parameters"]["wannier90"]["up"] = per_spin
        d["calculator_parameters"]["wannier90"]["down"] = per_spin
        with pytest.raises(ValueError, match="tot_magnetization"):
            _build(d, dfpt_codes)

    def test_non_integer_channel_occupations_raise(
        self, aiida_profile, dfpt_codes, fake_sg15_pseudo_family
    ):
        """Nelec + tot_magnetization must be even."""
        d = self._collinear_dict()
        d["calculator_parameters"]["tot_magnetization"] = 1  # nelec=8 -> 4.5/3.5
        with pytest.raises(ValueError, match="integer per-channel occupations"):
            _build(d, dfpt_codes)


class TestMultiBlockZnO:
    """The ZnO tutorial shape: four occupied blocks + one disentangled empty block.

    Mirrors ``docs/source/tutorials/zno.json`` (which is written in the
    legacy schema) translated into the ``KoopmansInput`` schema, with the
    band path dropped — this exercises the multi-block manifold routing
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
                # One alpha per orbital: 26 occupied + 2 empty (zno.json's
                # guess, flattened — the schema takes flat lists).
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

    def test_multi_block_manifolds(self, aiida_profile, dfpt_codes, fake_pseudodojo_lda_family):
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
        self, aiida_profile, dfpt_codes, fake_sg15_pseudo_family, spin_value
    ):
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
