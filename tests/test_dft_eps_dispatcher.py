"""Dispatcher tests for the dft_eps (ph.x dielectric constant) task.

Builds real ``WorkGraph`` objects through ``build_workgraph`` against a
throwaway profile (dummy codes, fake pseudos; nothing runs) and checks the
scf → ph.x → extract chain, plus the ``eps_inf='auto'`` hook of the DFPT
singlepoint stream. Mirrors the style of ``test_dfpt_dispatcher.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.aiida.workflows import _build_singlepoint_dfpt_workgraph, build_workgraph
from koopmans.input_file import KoopmansInput


def _si_eps_dict(**workflow_updates: Any) -> dict[str, Any]:
    """Return a minimal silicon dft_eps input dict (fake SG15 cutoffs pseudos)."""
    d: dict[str, Any] = {
        "workflow": {
            "task": "dft_eps",
            # The cutoffs-family fixture label: the dft_eps builder calls
            # get_builder_from_protocol eagerly, which needs recommended
            # cutoffs (the plain fake family is invisible to its query).
            "pseudo_library": "SG15/1.0/PBE/SR",
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
            "nbnd": 10,
        },
    }
    d["workflow"].update(workflow_updates)
    return d


def _si_dfpt_auto_dict() -> dict[str, Any]:
    """Return a silicon DFPT singlepoint dict with eps_inf='auto'."""
    return {
        "workflow": {
            "task": "singlepoint",
            "correction": "ki",
            "screening_method": "dfpt",
            "init_orbitals": "mlwfs",
            "calculate_alpha": True,
            "pseudo_library": "SG15/1.2/PBE/SR",
            "eps_inf": "auto",
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
                "projections": [[{"site": "Si", "ang_mtm": "sp"}]],
            },
        },
    }


class TestDftEps:
    """task='dft_eps' routes to the scf → ph.x → extract chain."""

    def test_chain(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_ph_code: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """The full dispatch builds scf, ph, and the tensor reduction."""
        inp = KoopmansInput.model_validate(_si_eps_dict())
        wg = build_workgraph(inp)
        names = wg.get_task_names()
        assert "scf" in names
        assert "ph" in names
        assert "extract_dielectric_constant" in names

        # ph.x runs the electric-field perturbation only (legacy
        # DFTPhWorkflow: epsil=.true., trans=.false.) at q = Gamma.
        inputph = wg.tasks["ph"].inputs["ph"]["parameters"].value.get_dict()["INPUTPH"]
        assert inputph["epsil"] is True
        assert inputph["trans"] is False
        assert wg.tasks["ph"].inputs["qpoints"].value.get_kpoints_mesh() == (
            [1, 1, 1],
            [0.0, 0.0, 0.0],
        )

        # The legacy eps scf passes nbnd=None: no empty bands, but the rest
        # of the user's PW parameters survive.
        system = wg.tasks["scf"].inputs["pw"]["parameters"].value.get_dict()["SYSTEM"]
        assert "nbnd" not in system
        assert system["ecutwfc"] == pytest.approx(20.0)

    def test_missing_ph_code_raises(
        self, aiida_profile_clean: Any, installed_pw_code: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Without a ph@localhost code the dispatcher fails with a setup hint.

        Requests ``aiida_profile_clean``: earlier tests may have installed a
        ``ph@localhost`` code in the session profile.
        """
        inp = KoopmansInput.model_validate(_si_eps_dict())
        with pytest.raises(ValueError, match=r"ph\.x"):
            build_workgraph(inp)


class TestDfptAutoEps:
    """eps_inf='auto' prepends the dielectric chain to the DFPT stream."""

    @pytest.fixture
    def dfpt_codes(
        self, installed_pw_code: Any, installed_kcw_code: Any, installed_wannier_codes: Any
    ) -> dict[str, Any]:
        """Assemble the DFPT code dict from the dummy-code fixtures."""
        return {
            "pw": installed_pw_code,
            "kcw": installed_kcw_code,
            **installed_wannier_codes,
        }

    def test_auto_adds_dielectric_task(
        self,
        aiida_profile: Any,
        dfpt_codes: Any,
        installed_ph_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """A 'dielectric' task appears alongside the kcw chain."""
        inp = KoopmansInput.model_validate(_si_dfpt_auto_dict())
        wg = _build_singlepoint_dfpt_workgraph(inp, codes=dfpt_codes)
        names = wg.get_task_names()
        assert "dielectric" in names
        assert "dfpt" in names

    def test_auto_without_ph_code_raises(
        self, aiida_profile_clean: Any, dfpt_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """eps_inf='auto' without ph@localhost fails with a setup hint.

        Requests ``aiida_profile_clean``: earlier tests may have installed a
        ``ph@localhost`` code in the session profile.
        """
        inp = KoopmansInput.model_validate(_si_dfpt_auto_dict())
        with pytest.raises(ValueError, match=r"ph\.x"):
            _build_singlepoint_dfpt_workgraph(inp, codes=dfpt_codes)

    def test_unknown_eps_string_raises(self, dfpt_codes: Any) -> None:
        """A non-'auto' string eps_inf is rejected up front."""
        d = _si_dfpt_auto_dict()
        d["workflow"]["eps_inf"] = "automatic"
        inp = KoopmansInput.model_validate(d)
        with pytest.raises(ValueError, match="not understood"):
            _build_singlepoint_dfpt_workgraph(inp, codes=dfpt_codes)
