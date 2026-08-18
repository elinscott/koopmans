"""Dispatcher test for the dft_bands task's occupations.

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

import pytest

from koopmans.aiida.workflows import build_workgraph
from koopmans.input_file import KoopmansInput
from tests.fixtures import silicon_pw_input


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


class TestNoWannierStep:
    """`dft_bands` runs no Wannierization: `overrides.wannier90` has nothing to reach."""

    def test_explicit_wannier90_density_raises(
        self, aiida_profile: Any, installed_pw_code: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """The route names its own `kpoints.path_density` as the alternative."""
        d = silicon_pw_input(pseudo_library="SG15/1.0/PBE/SR")
        d["kpoints"]["overrides"] = {"wannier90": {"path_density": 25.0}}
        inp = KoopmansInput.model_validate(d)
        with pytest.raises(ValueError, match=r"overrides\.wannier90\.path_density.*dft_bands"):
            build_workgraph(inp)
