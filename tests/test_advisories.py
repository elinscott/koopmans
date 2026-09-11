"""Unit tests for :func:`koopmans.aiida.workflows.advisories_for`.

These check the pure input-file logic only — no AiiDA profile or codes are
needed, since ``advisories_for`` never touches the ORM. The corresponding
``build_workgraph`` tests (``tests/test_singlepoint_dispatcher.py``) confirm
the routes these advisories describe still build.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from koopmans.aiida.workflows import advisories_for
from koopmans.input_file import KoopmansInput
from koopmans.input_file.workflow import GroupOrbitalsBy

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


def _bse_dict(**workflow_updates: Any) -> dict[str, Any]:
    """Return a minimal ``task: bse`` input, with its own ``calculator_parameters.yambo`` block."""
    d = _si_dict("bse", **workflow_updates)
    d["calculator_parameters"]["yambo"] = {
        "BndsRnXs": [1, 100],
        "NGsBlkXs": 2,
        "BSEBands": [1, 4],
        "BEnRange": [0, 10],
    }
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
    """``group_orbitals_by``/``group_orbitals_tol`` only apply to singlepoint/trajectory.

    ``group_orbitals_by`` is resolved at parse time (never ``None`` on the
    parsed model — see ``WorkflowConfig.resolve_group_orbitals_by``), so
    these tests exercise the schema resolution and the advisory together:
    the only state the advisory can still detect without extra bookkeeping
    is a criterion that *resolved* to ``'none'`` next to a typed tolerance
    (an explicit ``group_orbitals_by: 'none'`` with a tolerance is rejected
    at parse time, before ``advisories_for`` ever runs).
    """

    def test_defaults_are_silent(self) -> None:
        """Negative control: an input touching neither field names nothing."""
        inp = KoopmansInput.model_validate(_si_dict("wannierize"))
        assert inp.workflow.group_orbitals_by == GroupOrbitalsBy.NONE
        assert inp.workflow.group_orbitals_tol is None
        assert advisories_for(inp) == []

    def test_wannierize_with_mlwfs_alone_is_not_advised(self) -> None:
        """Negative control: init_orbitals alone resolves group_orbitals_by on its own.

        The criterion resolves to ``self_hartree`` for a Wannier-initialized
        DSCF run (the default screening method) with the tolerance defaulted
        alongside it, even though the user typed neither grouping keyword —
        but the resolved criterion is not ``'none'``, so nothing is flagged.
        """
        inp = KoopmansInput.model_validate(_si_dict("wannierize", init_orbitals="mlwfs"))
        assert inp.workflow.group_orbitals_by == GroupOrbitalsBy.SELF_HARTREE
        assert inp.workflow.group_orbitals_tol == pytest.approx(1.0e-4)
        assert advisories_for(inp) == []

    def test_wannierize_with_a_tolerance_is_advised(self) -> None:
        """Wannierize never calls the grouping helpers at all.

        ``init_orbitals`` stays at its ``pz`` default here, so the criterion
        resolves to ``'none'`` and the typed tolerance is the only signal
        left to flag.
        """
        d = _si_dict("wannierize")
        d["workflow"]["group_orbitals_tol"] = 0.05
        inp = KoopmansInput.model_validate(d)
        assert inp.workflow.group_orbitals_by == GroupOrbitalsBy.NONE
        assert inp.workflow.group_orbitals_tol == 0.05
        assert advisories_for(inp) == [
            "workflow.group_orbitals_tol has no effect on task: wannierize (it "
            "groups orbitals to share a screening parameter, computed only within a "
            "singlepoint or trajectory); it is kept for when you switch task to "
            "singlepoint."
        ]

    def test_dft_bands_with_an_explicit_criterion_is_silent(self) -> None:
        """A resolved (non-'none') criterion on a non-grouping task is not flagged.

        Whether ``group_orbitals_by: self_hartree`` here was typed or
        resolved on its own is no longer distinguishable on the parsed
        model — only a criterion that resolved to ``'none'`` is detectable
        without reintroducing per-field "was this typed" bookkeeping.
        """
        d = _si_dict("dft_bands")
        d["workflow"]["group_orbitals_by"] = "self_hartree"
        inp = KoopmansInput.model_validate(d)
        assert inp.workflow.group_orbitals_by == GroupOrbitalsBy.SELF_HARTREE
        assert advisories_for(inp) == []

    def test_explicit_none_with_a_tolerance_is_rejected_at_parse_time(self) -> None:
        """An explicit ``group_orbitals_by: 'none'`` next to a tolerance is a contradiction."""
        d = _si_dict("wannierize")
        d["workflow"]["group_orbitals_by"] = "none"
        d["workflow"]["group_orbitals_tol"] = 0.05
        with pytest.raises(ValueError, match="group_orbitals_tol"):
            KoopmansInput.model_validate(d)

    @pytest.mark.parametrize("with_tolerance", [False, True])
    def test_unknown_criterion_reaches_the_enum_field_error(self, with_tolerance: bool) -> None:
        """An unrecognized ``group_orbitals_by`` string surfaces the enum's own error.

        Regression: the default-tolerance lookup used to index the raw
        string before the ``group_orbitals_by`` field validated it,
        raising a bare ``KeyError`` instead of pydantic's enum message.
        """
        d = _si_dict("singlepoint", screening_method="dscf", correction="ki", init_orbitals="mlwfs")
        d["workflow"]["group_orbitals_by"] = "self-hartree"
        if with_tolerance:
            d["workflow"]["group_orbitals_tol"] = 0.05
        with pytest.raises(ValidationError, match=r"self_hartree.*spread.*none"):
            KoopmansInput.model_validate(d)

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
        assert inp.workflow.group_orbitals_by == GroupOrbitalsBy.NONE
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
        assert inp.workflow.group_orbitals_by == GroupOrbitalsBy.NONE
        assert advisories_for(inp) == [
            "workflow.group_orbitals_tol has no effect on task: singlepoint "
            "(group_orbitals_by resolved to 'none' for init_orbitals: mlwfs, "
            "screening_method: dfpt); it is kept for when you set group_orbitals_by "
            "to a criterion this run implements (self_hartree for DSCF, spread for "
            "DFPT)."
        ]

    @pytest.mark.parametrize(
        ("task", "workflow_updates"),
        [
            ("wannierize", {}),
            ("wannierize", {"init_orbitals": "mlwfs"}),
            ("wannierize", {"init_orbitals": "mlwfs", "screening_method": "dscf"}),
            ("wannierize", {"init_orbitals": "projwfs"}),
            ("dft_bands", {"init_orbitals": "mlwfs"}),
            ("dft_eps", {"init_orbitals": "mlwfs"}),
            ("wannierize", {"init_orbitals": "mlwfs", "group_orbitals_by": "none"}),
        ],
    )
    def test_typing_init_orbitals_alone_never_advises(
        self, task: str, workflow_updates: dict[str, Any]
    ) -> None:
        """``init_orbitals`` alone, on any non-grouping task, resolves silently.

        Regression for a spurious advisory the previous design (a
        point-of-use resolution helper called from ``advisories_for``
        itself) emitted for the last case: writing ``group_orbitals_by:
        'none'`` — the same value the criterion already resolves to —
        should not be treated differently from leaving it unset.
        """
        inp = KoopmansInput.model_validate(_si_dict(task, **workflow_updates))
        assert isinstance(inp.workflow.group_orbitals_by, GroupOrbitalsBy)
        assert advisories_for(inp) == []

    def test_wannier_initialized_dscf_default_is_not_advised(self) -> None:
        """The route this criterion is wired for stays silent, even at its own default."""
        d = _si_dict(
            "singlepoint",
            screening_method="dscf",
            correction="ki",
            init_orbitals="mlwfs",
        )
        inp = KoopmansInput.model_validate(d)
        assert inp.workflow.group_orbitals_by == GroupOrbitalsBy.SELF_HARTREE
        assert inp.workflow.group_orbitals_tol == pytest.approx(1.0e-4)
        assert advisories_for(inp) == []

    def test_bse_with_a_tolerance_is_advised_to_switch_task_not_criterion(self) -> None:
        """The `bse` task composes DFPT but runs no workflow-level grouping over it.

        Discriminates against advising a criterion the composed
        ``SinglepointBetheSalpeterWorkflow`` still could not act on (it
        forwards no ``group_orbitals_tol`` into its internal DFPT call): the
        fix is switching task, the same message ``wannierize``/``dft_bands``/
        ``dft_eps`` already get, not "set group_orbitals_by".
        """
        d = _bse_dict()
        d["workflow"]["group_orbitals_tol"] = 0.05
        inp = KoopmansInput.model_validate(d)
        assert inp.workflow.group_orbitals_by == GroupOrbitalsBy.NONE
        assert advisories_for(inp) == [
            "workflow.group_orbitals_tol has no effect on task: bse (it "
            "groups orbitals to share a screening parameter, computed only within a "
            "singlepoint or trajectory); it is kept for when you switch task to "
            "singlepoint."
        ]


class TestResolveGroupOrbitalsByDoesNotMutateCaller:
    """``resolve_group_orbitals_by`` must not write resolved fields back onto the caller's dict.

    Regression: the before-validator used to write ``group_orbitals_by``/
    ``group_orbitals_tol`` directly into the dict passed to
    ``model_validate``, so re-validating the same dict after editing it saw
    the previous parse's resolved values as if the user had typed them.
    """

    def test_callers_dict_is_untouched(self) -> None:
        """A resolved criterion and tolerance never land back in the caller's dict."""
        d = _si_dict("wannierize", init_orbitals="mlwfs")
        inp = KoopmansInput.model_validate(d)
        assert inp.workflow.group_orbitals_by == GroupOrbitalsBy.SELF_HARTREE
        assert "group_orbitals_by" not in d["workflow"]
        assert "group_orbitals_tol" not in d["workflow"]

    def test_editing_and_revalidating_resolves_afresh(self) -> None:
        """Flipping ``init_orbitals`` and re-validating must not inherit the earlier resolution."""
        d = _si_dict("wannierize", init_orbitals="mlwfs")
        first = KoopmansInput.model_validate(d)
        assert first.workflow.group_orbitals_by == GroupOrbitalsBy.SELF_HARTREE

        d["workflow"]["init_orbitals"] = "pz"
        second = KoopmansInput.model_validate(d)
        assert second.workflow.group_orbitals_by == GroupOrbitalsBy.NONE
        assert second.workflow.group_orbitals_tol is None

    def test_adding_a_tolerance_after_a_silent_first_parse_is_advised(self) -> None:
        """A tolerance added between two parses reads as freshly typed, not a contradiction.

        The first parse resolves ``group_orbitals_by`` to ``'none'`` without
        writing it into the caller's dict, so a tolerance added afterwards
        is indistinguishable from one typed alongside an unset criterion —
        it is orphaned and advised, not rejected as an explicit-none
        contradiction.
        """
        d = _si_dict("wannierize")
        first = KoopmansInput.model_validate(d)
        assert first.workflow.group_orbitals_by == GroupOrbitalsBy.NONE

        d["workflow"]["group_orbitals_tol"] = 0.05
        second = KoopmansInput.model_validate(d)
        assert second.workflow.group_orbitals_by == GroupOrbitalsBy.NONE
        assert advisories_for(second) == [
            "workflow.group_orbitals_tol has no effect on task: wannierize (it "
            "groups orbitals to share a screening parameter, computed only within a "
            "singlepoint or trajectory); it is kept for when you switch task to "
            "singlepoint."
        ]


class TestBothAdvisoriesTogether:
    """The two checks are independent and both surface on the same input."""

    def test_two_orphaned_keywords_yield_two_messages(self) -> None:
        """One message per keyword left with no effect, not a merged one."""
        d = _si_dict("wannierize")
        d["kpoints"]["smooth_interpolation_factor"] = 2
        d["workflow"]["group_orbitals_tol"] = 0.05
        inp = KoopmansInput.model_validate(d)
        assert len(advisories_for(inp)) == 2
