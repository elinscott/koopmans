"""Unit tests for :func:`koopmans.aiida.workflows.advisories_for`.

These check the pure input-file logic only — no AiiDA profile or codes are
needed, since ``advisories_for`` never touches the ORM. The corresponding
``build_workgraph`` tests (``tests/test_singlepoint_dispatcher.py``) confirm
the routes these advisories describe still build.
"""

from __future__ import annotations

from typing import Any

from koopmans.aiida.workflows import advisories_for
from koopmans.aiida.workflows.grouping import resolve_orbital_grouping
from koopmans.input_file import KoopmansInput

_SI_ATOMS: dict[str, Any] = {
    "cell_parameters": {"periodic": True, "ibrav": 2, "celldms": {"1": 10.2622}},
    "atomic_positions": {
        "units": "crystal",
        "positions": [["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25]],
    },
}


def _si_dict(task: str, **workflow_updates: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "workflow": {"task": task, "pseudo_library": "SG15/1.2/PBE/SR"},
        "atoms": _SI_ATOMS,
        "kpoints": {"grid": [2, 2, 2], "offset": [0, 0, 0]},
        "calculator_parameters": {"ecutwfc": 20.0},
    }
    d["workflow"].update(workflow_updates)
    return d


class TestSmoothInterpolationFactorAdvisory:
    """``kpoints.smooth_interpolation_factor`` only shapes the DSCF band interpolation."""

    def test_default_factor_is_silent(self) -> None:
        """Negative control: the neutral default names nothing."""
        inp = KoopmansInput.model_validate(_si_dict("wannierize"))
        assert advisories_for(inp) == []

    def test_wannierize_is_advised(self) -> None:
        """A wannierize task Wannierizes no smooth-interpolation mesh."""
        d = _si_dict("wannierize")
        d["kpoints"]["smooth_interpolation_factor"] = 2
        inp = KoopmansInput.model_validate(d)
        assert advisories_for(inp) == [
            "kpoints.smooth_interpolation_factor has no effect on task: wannierize "
            "(it shapes the ΔSCF band-structure interpolation); it is kept for when "
            "you switch task to singlepoint."
        ]

    def test_dft_bands_is_advised(self) -> None:
        """A dft_bands task computes DFT bands directly, no ΔSCF interpolation."""
        d = _si_dict("dft_bands")
        d["kpoints"]["smooth_interpolation_factor"] = 3
        inp = KoopmansInput.model_validate(d)
        assert advisories_for(inp) == [
            "kpoints.smooth_interpolation_factor has no effect on task: dft_bands "
            "(it shapes the ΔSCF band-structure interpolation); it is kept for when "
            "you switch task to singlepoint."
        ]

    def test_dft_eps_is_advised(self) -> None:
        """A dft_eps task runs ph.x, which has no band-structure interpolation."""
        d = _si_dict("dft_eps")
        d["kpoints"]["smooth_interpolation_factor"] = 2
        inp = KoopmansInput.model_validate(d)
        assert advisories_for(inp) == [
            "kpoints.smooth_interpolation_factor has no effect on task: dft_eps "
            "(it shapes the ΔSCF band-structure interpolation); it is kept for when "
            "you switch task to singlepoint."
        ]

    def test_dfpt_singlepoint_is_advised(self) -> None:
        """DFPT screening within a singlepoint also runs no band interpolation."""
        d = _si_dict(
            "singlepoint",
            screening_method="dfpt",
            correction="ki",
            init_orbitals="mlwfs",
        )
        d["kpoints"]["smooth_interpolation_factor"] = 2
        inp = KoopmansInput.model_validate(d)
        assert advisories_for(inp) == [
            "kpoints.smooth_interpolation_factor has no effect on task: singlepoint "
            "(screening_method: dfpt; it shapes the ΔSCF band-structure "
            "interpolation); it is kept for when you switch screening_method to dscf."
        ]

    def test_dscf_singlepoint_is_not_advised(self) -> None:
        """The one route that performs the interpolation is silent."""
        d = _si_dict(
            "singlepoint",
            screening_method="dscf",
            correction="ki",
            init_orbitals="mlwfs",
        )
        d["kpoints"]["smooth_interpolation_factor"] = 2
        d["kpoints"]["path"] = "GXG"
        inp = KoopmansInput.model_validate(d)
        assert advisories_for(inp) == []


class TestOrbitalGroupingAdvisory:
    """``group_orbitals_by``/``group_orbitals_tol`` only apply to singlepoint/trajectory."""

    def test_defaults_are_silent(self) -> None:
        """Negative control: an input touching neither field names nothing."""
        inp = KoopmansInput.model_validate(_si_dict("wannierize"))
        assert advisories_for(inp) == []

    def test_wannierize_with_mlwfs_alone_is_not_advised(self) -> None:
        """Negative control: init_orbitals alone resolves group_orbitals_by on its own.

        Resolution (via ``resolve_orbital_grouping``) defaults the criterion
        to ``self_hartree`` for a Wannier-initialized DSCF run — but the
        user typed neither grouping keyword, so the field stays unset on
        the parsed model and nothing is worth flagging.
        """
        inp = KoopmansInput.model_validate(_si_dict("wannierize", init_orbitals="mlwfs"))
        assert inp.workflow.group_orbitals_by is None
        criterion, _ = resolve_orbital_grouping(inp.workflow)
        assert criterion.value == "self_hartree"
        assert advisories_for(inp) == []

    def test_wannierize_with_a_tolerance_is_advised(self) -> None:
        """Wannierize never calls the grouping helpers at all."""
        d = _si_dict("wannierize")
        d["workflow"]["group_orbitals_tol"] = 0.05
        inp = KoopmansInput.model_validate(d)
        assert advisories_for(inp) == [
            "workflow.group_orbitals_tol has no effect on task: wannierize (it "
            "groups orbitals to share a screening parameter, computed only within a "
            "singlepoint or trajectory); it is kept for when you switch task to "
            "singlepoint."
        ]

    def test_dft_bands_with_an_explicit_criterion_is_advised(self) -> None:
        """A criterion alone (no tolerance) still signals intent to group."""
        d = _si_dict("dft_bands")
        d["workflow"]["group_orbitals_by"] = "self_hartree"
        inp = KoopmansInput.model_validate(d)
        assert advisories_for(inp) == [
            "workflow.group_orbitals_by has no effect on task: dft_bands (it groups "
            "orbitals to share a screening parameter, computed only within a "
            "singlepoint or trajectory); it is kept for when you switch task to "
            "singlepoint."
        ]

    def test_dft_bands_with_both_keywords_is_advised_once_naming_both(self) -> None:
        """Both keywords typed together name both, not one message per keyword."""
        d = _si_dict("dft_bands")
        d["workflow"]["group_orbitals_by"] = "self_hartree"
        d["workflow"]["group_orbitals_tol"] = 0.05
        inp = KoopmansInput.model_validate(d)
        assert advisories_for(inp) == [
            "workflow.group_orbitals_by/group_orbitals_tol have no effect on task: "
            "dft_bands (they group orbitals to share a screening parameter, computed "
            "only within a singlepoint or trajectory); they are kept for when you "
            "switch task to singlepoint."
        ]

    def test_dscf_singlepoint_resolved_to_none_with_a_tolerance_is_advised(self) -> None:
        """``init_orbitals: pz`` resolves grouping to 'none'; the tolerance is orphaned."""
        d = _si_dict(
            "singlepoint",
            screening_method="dscf",
            correction="ki",
            init_orbitals="pz",
            group_orbitals_tol=0.05,
        )
        inp = KoopmansInput.model_validate(d)
        assert inp.workflow.group_orbitals_by is None
        criterion, _ = resolve_orbital_grouping(inp.workflow)
        assert criterion.value == "none"
        assert advisories_for(inp) == [
            "workflow.group_orbitals_tol has no effect on task: singlepoint "
            "(group_orbitals_by resolved to 'none' for init_orbitals: pz, "
            "screening_method: dscf); it is kept for when you set group_orbitals_by "
            "to a criterion this run implements (self_hartree for DSCF, spread for "
            "DFPT)."
        ]

    def test_dfpt_singlepoint_default_with_a_tolerance_is_advised(self) -> None:
        """DFPT never auto-defaults to 'spread'; a tolerance alone stays orphaned."""
        d = _si_dict(
            "singlepoint",
            screening_method="dfpt",
            correction="ki",
            init_orbitals="mlwfs",
            group_orbitals_tol=0.05,
        )
        inp = KoopmansInput.model_validate(d)
        assert inp.workflow.group_orbitals_by is None
        criterion, _ = resolve_orbital_grouping(inp.workflow)
        assert criterion.value == "none"
        assert advisories_for(inp) == [
            "workflow.group_orbitals_tol has no effect on task: singlepoint "
            "(group_orbitals_by resolved to 'none' for init_orbitals: mlwfs, "
            "screening_method: dfpt); it is kept for when you set group_orbitals_by "
            "to a criterion this run implements (self_hartree for DSCF, spread for "
            "DFPT)."
        ]

    def test_wannier_initialized_dscf_default_is_not_advised(self) -> None:
        """The route this criterion is wired for stays silent, even at its own default."""
        d = _si_dict(
            "singlepoint",
            screening_method="dscf",
            correction="ki",
            init_orbitals="mlwfs",
        )
        inp = KoopmansInput.model_validate(d)
        assert inp.workflow.group_orbitals_by is None
        criterion, _ = resolve_orbital_grouping(inp.workflow)
        assert criterion.value == "self_hartree"
        assert advisories_for(inp) == []


class TestBothAdvisoriesTogether:
    """The two checks are independent and both surface on the same input."""

    def test_two_orphaned_keywords_yield_two_messages(self) -> None:
        """One message per keyword left with no effect, not a merged one."""
        d = _si_dict("wannierize")
        d["kpoints"]["smooth_interpolation_factor"] = 2
        d["workflow"]["group_orbitals_tol"] = 0.05
        inp = KoopmansInput.model_validate(d)
        assert len(advisories_for(inp)) == 2
