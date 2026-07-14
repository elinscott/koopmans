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
