"""Dispatcher tests for the dft_bands task's occupations and spin regime.

Builds the real ``WorkGraph`` through ``build_workgraph`` against a
throwaway profile (dummy code, fake pseudos; nothing runs) and checks that
an input carrying no ``pw`` block at all still reaches pw.x with fixed
occupations. This is the end-to-end discriminator for both halves of the
fix: aiida-koopmans's ``RunPwBands`` defaulting ``electronic_type`` to
``INSULATOR``, and koopmans2's namelist dump no longer needing a user value
to survive.
"""

from __future__ import annotations

from typing import Any

from koopmans.aiida.workflows import build_workgraph
from koopmans.input_file import KoopmansInput
from tests.fixtures import silicon_pw_input


def _spin_input(spin: str, **calculator_parameters: Any) -> KoopmansInput:
    """Return a silicon ``dft_bands`` input carrying ``workflow.spin``."""
    d = silicon_pw_input(
        pseudo_library="SG15/1.0/PBE/SR",
        calculator_parameters={"ecutwfc": 20.0, **calculator_parameters},
    )
    d["workflow"]["spin"] = spin
    return KoopmansInput.model_validate(d)


def _system_namelists(inp: KoopmansInput) -> dict[str, dict[str, Any]]:
    """Return the SYSTEM namelist each pw.x step of a built dft_bands graph receives."""
    task = build_workgraph(inp).tasks["PwBandsWorkChain"]
    return {
        step: task.inputs[step]["pw"]["parameters"].value.get_dict()["SYSTEM"]
        for step in ("scf", "bands")
    }


class TestDftBandsOccupations:
    """task='dft_bands' reaches pw.x with fixed occupations by default."""

    def test_no_pw_block_fixes_occupations_on_both_steps(
        self, aiida_profile: Any, installed_pw_code: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """No ``pw`` block in the input: the ak2 ``INSULATOR`` default still fires.

        Before the fix, this needed no user input to fail: cold smearing
        (``occupations: smearing``, ``smearing: cold``, ``degauss: 0.02``)
        reached both the scf and the bands pw.x steps regardless of what
        the input said.
        """
        inp = KoopmansInput.model_validate(silicon_pw_input(pseudo_library="SG15/1.0/PBE/SR"))
        wg = build_workgraph(inp)

        task = wg.tasks["PwBandsWorkChain"]
        for step in ("scf", "bands"):
            system = task.inputs[step]["pw"]["parameters"].value.get_dict()["SYSTEM"]
            assert system["occupations"] == "fixed", step
            assert "smearing" not in system, step
            assert "degauss" not in system, step


class TestDftBandsSpin:
    """``workflow.spin`` reaches the pw.x SYSTEM namelist of both steps."""

    def test_each_regime_writes_its_own_namelist(
        self, aiida_profile: Any, installed_pw_code: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """The four regimes give four different namelists.

        The whole-of-route check: before this wiring the route named
        ``workflow.spin`` nowhere, so all four builds produced the same
        ``{'ecutwfc': ..., 'ecutrho': ..., 'occupations': 'fixed'}`` and a
        user asking for a spin-polarized band structure got an unpolarized
        one with no complaint.
        """
        namelists = {
            "none": _system_namelists(_spin_input("none")),
            "collinear": _system_namelists(_spin_input("collinear", tot_magnetization=0)),
            "non_collinear": _system_namelists(_spin_input("non_collinear")),
            "spin_orbit": _system_namelists(_spin_input("spin_orbit")),
        }

        for step in ("scf", "bands"):
            unpolarized = namelists["none"][step]
            assert "nspin" not in unpolarized
            assert "noncolin" not in unpolarized

            assert namelists["collinear"][step]["nspin"] == 2
            assert namelists["collinear"][step]["tot_magnetization"] == 0

            assert namelists["non_collinear"][step]["noncolin"] is True
            assert "lspinorb" not in namelists["non_collinear"][step]

            assert namelists["spin_orbit"][step]["noncolin"] is True
            assert namelists["spin_orbit"][step]["lspinorb"] is True

            distinct = [sorted(namelists[regime][step].items()) for regime in namelists]
            assert all(a != b for i, a in enumerate(distinct) for b in distinct[i + 1 :]), step
