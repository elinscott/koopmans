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
    if task == "singlepoint":
        d["workflow"]["formalism"] = "dscf"
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
        "LongDrXs": [1, 0, 0],
        "BLongDir": [1, 0, 0],
    }
    return d


class TestFormalismAdvisory:
    """``workflow.formalism`` only matters to a task that evaluates the Koopmans correction."""

    @pytest.mark.parametrize("task", ["dft_bands", "dft_eps", "wannierize"])
    def test_stated_on_a_task_without_a_koopmans_step_is_advised(self, task: str) -> None:
        """A stated formalism is kept, with a notice naming the task."""
        inp = KoopmansInput.model_validate(_si_dict(task, formalism="dfpt"))
        (message,) = advisories_for(inp)
        assert f"no effect on task: {task}" in message

    def test_unstated_is_silent_and_singlepoint_is_not_advised(self) -> None:
        """Negative controls: unset on a non-Koopmans task, and set on a singlepoint."""
        assert advisories_for(KoopmansInput.model_validate(_si_dict("dft_bands"))) == []
        assert advisories_for(KoopmansInput.model_validate(_si_dict("singlepoint"))) == []


_SMOOTH_NOT_HERE = (
    "kpoints.overrides.wannierize_smooth has no effect on this run: only a singlepoint "
    "initialized from Wannier functions (init_orbitals: mlwfs or projwfs) runs the smooth "
    "Wannierization; it is kept for when you switch to one."
)


def _with_overrides(d: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    d["kpoints"]["overrides"] = overrides
    return d


class TestKpointOverrideAdvisory:
    """A ``kpoints.overrides`` entry naming a step the run lacks is kept, with a notice."""

    def test_no_entries_are_silent(self) -> None:
        """Negative control: an input with no entries names nothing."""
        inp = KoopmansInput.model_validate(_si_dict("wannierize"))
        assert advisories_for(inp) == []

    @pytest.mark.parametrize("task", ["wannierize", "dft_bands", "dft_eps"])
    def test_smooth_wannierization_off_singlepoint_is_advised(self, task: str) -> None:
        """Only a singlepoint runs the smooth Wannierization."""
        d = _with_overrides(_si_dict(task), wannierize_smooth={"factor": 2})
        assert advisories_for(KoopmansInput.model_validate(d)) == [_SMOOTH_NOT_HERE]

    @pytest.mark.parametrize("formalism", ["dscf", "dfpt"])
    def test_neither_singlepoint_route_is_advised(self, formalism: str) -> None:
        """Both routes run the smooth Wannierization, so neither entry is inert."""
        d = _si_dict(
            "singlepoint",
            formalism=formalism,
            correction="ki",
            init_orbitals="mlwfs",
        )
        d["kpoints"]["path"] = "GXG"
        _with_overrides(d, wannierize={"path_density": 20.0}, wannierize_smooth={"factor": 2})
        assert advisories_for(KoopmansInput.model_validate(d)) == []

    def test_smooth_wannierization_without_a_path_is_advised(self) -> None:
        """Without a path there is no band structure for it to serve."""
        d = _si_dict("singlepoint", formalism="dfpt", correction="ki", init_orbitals="mlwfs")
        _with_overrides(d, wannierize_smooth={"factor": 2})
        assert advisories_for(KoopmansInput.model_validate(d)) == [
            "kpoints.overrides.wannierize_smooth has no effect without a kpoints.path: the "
            "smooth Wannierization serves only the band structure interpolated along it; it "
            "is kept for when you add one."
        ]

    def test_smooth_wannierization_at_factor_one_is_advised(self) -> None:
        """A factor of 1 densifies nothing, so no smooth Wannierization runs."""
        d = _si_dict("singlepoint", formalism="dfpt", correction="ki", init_orbitals="mlwfs")
        d["kpoints"]["path"] = "GXG"
        _with_overrides(d, wannierize_smooth={"factor": 1})
        assert advisories_for(KoopmansInput.model_validate(d)) == [
            "kpoints.overrides.wannierize_smooth has no effect at factor [1, 1, 1], which "
            "densifies nothing; it is kept for when you raise the factor."
        ]

    def test_wannierize_density_off_a_wannierization_is_advised(self) -> None:
        """A dft_bands task Wannierizes nothing."""
        d = _with_overrides(_si_dict("dft_bands"), wannierize={"path_density": 80.0})
        assert advisories_for(KoopmansInput.model_validate(d)) == [
            "kpoints.overrides.wannierize has no effect on task: dft_bands, which runs no "
            "Wannierization; it is kept for when you switch to a run that does."
        ]

    def test_wannierize_density_without_a_path_is_advised(self) -> None:
        """A wannierize task with no path interpolates no band structure."""
        d = _with_overrides(_si_dict("wannierize"), wannierize={"path_density": 80.0})
        assert advisories_for(KoopmansInput.model_validate(d)) == [
            "kpoints.overrides.wannierize has no effect without a kpoints.path: it sets the "
            "density of the band structure wannier90 interpolates along that path; it is "
            "kept for when you add one."
        ]

    def test_wannierize_density_on_bse_names_a_task_that_interpolates(self) -> None:
        """A bse run refuses a kpoints.path, so adding one is not the fix."""
        d = _with_overrides(
            _bse_dict(formalism="dfpt", correction="ki", init_orbitals="mlwfs"),
            wannierize={"path_density": 80.0},
            wannierize_smooth={"factor": 2},
        )
        reason = (
            "because this run refuses a kpoints.path, so it interpolates no band structure; "
            "it is kept for when you switch to a task that does, such as task: wannierize or "
            "a singlepoint without atoms.snapshots."
        )
        assert advisories_for(KoopmansInput.model_validate(d)) == [
            f"kpoints.overrides.wannierize has no effect {reason}",
            "kpoints.overrides.wannierize_smooth has no effect on this run: only a "
            "singlepoint initialized from Wannier functions (init_orbitals: mlwfs or "
            "projwfs) runs the smooth Wannierization; it is kept for when you switch to one.",
        ]

    def test_gamma_only_entries_other_than_scf_and_nscf_are_advised(self) -> None:
        """Gamma-only keeps the entries it can ignore and refuses the ones it cannot."""
        d = _with_overrides(
            _si_dict("wannierize"),
            wannierize={"path_density": 80.0},
            wannierize_smooth={"factor": 2},
            dielectric={"grid": [4, 4, 4]},
        )
        d["kpoints"] = {"gamma_only": True, "overrides": d["kpoints"]["overrides"]}
        assert advisories_for(KoopmansInput.model_validate(d)) == [
            f"kpoints.overrides.{step} has no effect on a gamma-only run, which samples Gamma "
            "alone; it is kept for when you switch to a k-point grid."
            for step in ("wannierize", "wannierize_smooth", "dielectric")
        ]

    @pytest.mark.parametrize(
        ("nscf", "factor", "advised"),
        [
            ([4, 4, 4], 2, True),
            ([4, 4, 4], 4, False),
            (None, 2, False),
            ([4, 4, 4], [2, 2, 1], True),
        ],
    )
    def test_a_smooth_mesh_no_denser_than_the_wannier_mesh_is_advised(
        self, nscf: list[int] | None, factor: Any, advised: bool
    ) -> None:
        """The smooth mesh is the top-level grid times the factor, whatever nscf states."""
        d = _si_dict("singlepoint", formalism="dfpt", correction="ki", init_orbitals="mlwfs")
        d["kpoints"]["grid"] = [2, 2, 2]
        d["kpoints"]["path"] = "GXG"
        overrides: dict[str, Any] = {"wannierize_smooth": {"factor": factor}}
        if nscf is not None:
            overrides["nscf"] = {"grid": nscf}
        _with_overrides(d, **overrides)
        messages = advisories_for(KoopmansInput.model_validate(d))
        if not advised:
            assert messages == []
            return
        assert len(messages) == 1
        assert "(2, 2, 2)" in messages[0]
        assert "(4, 4, 4) mesh of the main Wannierization" in messages[0]

    @pytest.mark.parametrize("task", ["dft_bands", "dft_eps"])
    def test_nscf_mesh_without_an_nscf_step_is_advised(self, task: str) -> None:
        """Neither task runs an nscf step."""
        d = _with_overrides(_si_dict(task), nscf={"grid": [4, 4, 4]})
        assert advisories_for(KoopmansInput.model_validate(d)) == [
            f"kpoints.overrides.nscf has no effect on task: {task}, which runs no nscf "
            "step; it is kept for when you switch to a task that Wannierizes."
        ]

    def test_dielectric_mesh_on_dft_eps_names_the_scf_entry(self) -> None:
        """The task's own dielectric calculation samples ``scf``, not ``dielectric``."""
        d = _with_overrides(_si_dict("dft_eps"), dielectric={"grid": [4, 4, 4]})
        assert advisories_for(KoopmansInput.model_validate(d)) == [
            "kpoints.overrides.dielectric has no effect on task: dft_eps, whose dielectric "
            "calculation is its own scf: set kpoints.overrides.scf or kpoints.grid for it; "
            "it is kept for when you switch task to singlepoint or bse."
        ]

    def test_dielectric_mesh_with_a_numeric_eps_inf_is_advised(self) -> None:
        """A numeric ``eps_inf`` runs no dielectric step for the mesh to reach."""
        d = _si_dict("singlepoint", formalism="dfpt", correction="ki", init_orbitals="mlwfs")
        d["workflow"]["eps_inf"] = 11.7
        _with_overrides(d, dielectric={"grid": [4, 4, 4]})
        assert advisories_for(KoopmansInput.model_validate(d)) == [
            "kpoints.overrides.dielectric has no effect with workflow.eps_inf not 'auto' (it "
            "sets the mesh of the step that computes it); it is kept for when you set "
            "eps_inf: auto."
        ]

    def test_dielectric_mesh_on_bse_is_silent(self) -> None:
        """The bse task runs the same dielectric step the DFPT singlepoint does."""
        d = _with_overrides(
            _bse_dict(formalism="dfpt", correction="ki", init_orbitals="mlwfs", eps_inf="auto"),
            dielectric={"grid": [4, 4, 4]},
        )
        assert advisories_for(KoopmansInput.model_validate(d)) == []


class TestDielectricMeshAdvisory:
    """``workflow.eps_inf: auto`` without ``kpoints.overrides.dielectric`` names its mesh."""

    _ADVICE = (
        "; it converges slowly with k-points, so that mesh may be too coarse. Set "
        "kpoints.overrides.dielectric to give the dielectric-constant step a denser mesh "
        "of its own."
    )

    @staticmethod
    def _dfpt_dict(**kpoints_updates: Any) -> dict[str, Any]:
        d = _si_dict(
            "singlepoint",
            formalism="dfpt",
            correction="ki",
            init_orbitals="mlwfs",
            eps_inf="auto",
        )
        d["kpoints"].update(kpoints_updates)
        return d

    def test_names_kpoints_grid_when_scf_is_unset(self) -> None:
        """Without an ``overrides.scf`` mesh, the dielectric step samples ``kpoints.grid``.

        A mutant that drops the mesh from the message, or that fires
        regardless of the actual grid, would still pass a bare "advisory
        present" check — this pins the printed mesh too.
        """
        inp = KoopmansInput.model_validate(self._dfpt_dict(grid=[4, 4, 4]))
        assert advisories_for(inp) == [
            "workflow.eps_inf: auto computes the dielectric constant on the "
            "kpoints.grid mesh (4, 4, 4)" + self._ADVICE
        ]

    def test_names_the_scf_override_grid(self) -> None:
        """With ``overrides.scf.grid`` set, the dielectric step samples THAT mesh.

        A mutant that always names ``kpoints.grid`` would print (2, 2, 2)
        here while the calculation actually runs on (6, 6, 6).
        """
        d = self._dfpt_dict(grid=[2, 2, 2])
        d["kpoints"]["overrides"] = {"scf": {"grid": [6, 6, 6]}}
        inp = KoopmansInput.model_validate(d)
        assert advisories_for(inp) == [
            "workflow.eps_inf: auto computes the dielectric constant on the "
            "kpoints.overrides.scf.grid mesh (6, 6, 6)" + self._ADVICE
        ]

    def test_names_the_scf_override_spacing(self) -> None:
        """With ``overrides.scf.grid_spacing`` set, no concrete mesh is knowable at parse.

        A mutant that invents dimensions from ``kpoints.grid`` regardless
        would misdescribe a run whose mesh the protocol only fixes later.
        """
        d = self._dfpt_dict(grid=[2, 2, 2])
        d["kpoints"]["overrides"] = {"scf": {"grid_spacing": 0.15}}
        inp = KoopmansInput.model_validate(d)
        assert advisories_for(inp) == [
            "workflow.eps_inf: auto computes the dielectric constant on the mesh "
            "kpoints.overrides.scf.grid_spacing = 0.15 builds (dimensions unknown "
            "until the calculation runs)" + self._ADVICE
        ]

    def test_a_dielectric_entry_is_silent(self) -> None:
        """The negative control: a mesh of its own silences the advisory."""
        d = self._dfpt_dict()
        d["kpoints"]["overrides"] = {"dielectric": {"grid": [4, 4, 4]}}
        assert advisories_for(KoopmansInput.model_validate(d)) == []

    def test_numeric_eps_inf_is_silent(self) -> None:
        """A numeric ``eps_inf`` runs no dielectric step, so no mesh to advise on."""
        d = _si_dict("singlepoint", formalism="dfpt", correction="ki", init_orbitals="mlwfs")
        d["workflow"]["eps_inf"] = 11.7
        inp = KoopmansInput.model_validate(d)
        assert advisories_for(inp) == []

    def test_dscf_screening_names_kpoints_grid(self) -> None:
        """The DSCF dielectric step samples ``kpoints.grid``, the one mesh that route takes."""
        d = _si_dict("singlepoint", formalism="dscf", correction="ki", init_orbitals="mlwfs")
        d["workflow"]["eps_inf"] = "auto"
        d["kpoints"]["grid"] = [3, 3, 3]
        inp = KoopmansInput.model_validate(d)
        assert advisories_for(inp) == [
            "workflow.eps_inf: auto computes the dielectric constant on the "
            "kpoints.grid mesh (3, 3, 3)" + self._ADVICE
        ]

    def test_bse_is_advised_like_the_dfpt_singlepoint(self) -> None:
        """The bse task runs the DFPT chain's dielectric step, so it gets the same notice."""
        d = _bse_dict(formalism="dfpt", correction="ki", init_orbitals="mlwfs", eps_inf="auto")
        assert advisories_for(KoopmansInput.model_validate(d)) == [
            "workflow.eps_inf: auto computes the dielectric constant on the "
            "kpoints.grid mesh (2, 2, 2)" + self._ADVICE
        ]


class TestEpsInfInertAdvisory:
    """``workflow.eps_inf`` is kept, with a notice, where no step consumes it."""

    _MESH_ADVISORY = "workflow.eps_inf: auto computes the dielectric constant"

    @staticmethod
    def _dscf(**workflow_updates: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "formalism": "dscf",
            "correction": "ki",
            "init_orbitals": "mlwfs",
            "eps_inf": "auto",
        }
        return _si_dict("singlepoint", **{**base, **workflow_updates})

    def test_calculate_alpha_off_is_advised_instead_of_the_mesh(self) -> None:
        """No screening runs, so the dielectric constant is unused and no mesh is named."""
        inp = KoopmansInput.model_validate(self._dscf(calculate_alpha=False))
        assert advisories_for(inp) == [
            "workflow.eps_inf has no effect with workflow.calculate_alpha: false: the "
            "dielectric constant enters only the screening step, which this run skips; "
            "it is kept for when you switch screening back on."
        ]

    def test_calculate_alpha_off_on_dfpt_is_advised(self) -> None:
        """The DFPT screening step is skipped too, so its dielectric constant is unused."""
        d = self._dscf(calculate_alpha=False, formalism="dfpt")
        inp = KoopmansInput.model_validate(d)
        assert [a for a in advisories_for(inp) if "calculate_alpha" in a]
        assert not [a for a in advisories_for(inp) if self._MESH_ADVISORY in a]

    def test_mp_correction_off_is_advised(self) -> None:
        """Without the Makov-Payne correction the dielectric constant has no consumer."""
        inp = KoopmansInput.model_validate(self._dscf(mp_correction=False))
        assert advisories_for(inp) == [
            "workflow.eps_inf has no effect with workflow.mp_correction: false: the "
            "dielectric constant enters only the Makov-Payne correction; it is kept "
            "for when you switch the correction back on."
        ]

    def test_ml_predict_is_advised(self) -> None:
        """A trained model replaces the screening step that would use the constant."""
        d = self._dscf()
        d["ml"] = {"mode": "predict"}
        inp = KoopmansInput.model_validate(d)
        assert [a for a in advisories_for(inp) if "ml.mode: predict" in a]
        assert not [a for a in advisories_for(inp) if self._MESH_ADVISORY in a]

    def test_a_consumed_eps_inf_gets_the_mesh_advisory_only(self) -> None:
        """Negative control: with a consumer, the inert notices stay silent."""
        advisories = advisories_for(KoopmansInput.model_validate(self._dscf()))
        assert len(advisories) == 1
        assert advisories[0].startswith(self._MESH_ADVISORY)


class TestOrbitalGroupingAdvisory:
    """``group_orbitals_by``/``group_orbitals_tol`` only apply to the singlepoint task.

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

    def test_singlepoint_with_mlwfs_alone_is_not_advised(self) -> None:
        """Negative control: init_orbitals alone resolves group_orbitals_by on its own.

        The criterion resolves to ``self_hartree`` for a Wannier-initialized
        DSCF run with the tolerance defaulted alongside it, even though the
        user typed neither grouping keyword — but the resolved criterion is
        not ``'none'``, so nothing is flagged.
        """
        inp = KoopmansInput.model_validate(_si_dict("singlepoint", init_orbitals="mlwfs"))
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
            "singlepoint); it is kept for when you switch task to "
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
        d = _si_dict("singlepoint", formalism="dscf", correction="ki", init_orbitals="mlwfs")
        d["workflow"]["group_orbitals_by"] = "self-hartree"
        if with_tolerance:
            d["workflow"]["group_orbitals_tol"] = 0.05
        with pytest.raises(ValidationError, match=r"self_hartree.*spread.*none"):
            KoopmansInput.model_validate(d)

    def test_dscf_singlepoint_resolved_to_none_with_a_tolerance_is_advised(self) -> None:
        """``init_orbitals: pz`` resolves grouping to 'none'; the tolerance is orphaned."""
        d = _si_dict(
            "singlepoint",
            formalism="dscf",
            correction="ki",
            init_orbitals="pz",
            group_orbitals_tol=0.05,
        )
        inp = KoopmansInput.model_validate(d)
        assert inp.workflow.group_orbitals_by == GroupOrbitalsBy.NONE
        assert advisories_for(inp) == [
            "workflow.group_orbitals_tol has no effect on task: singlepoint "
            "(group_orbitals_by resolved to 'none' for init_orbitals: pz, "
            "formalism: dscf); it is kept for when you set group_orbitals_by "
            "to a criterion this run implements (self_hartree for DSCF, spread for "
            "DFPT)."
        ]

    def test_dfpt_singlepoint_default_with_a_tolerance_is_advised(self) -> None:
        """DFPT never auto-defaults to 'spread'; a tolerance alone stays orphaned."""
        d = _si_dict(
            "singlepoint",
            formalism="dfpt",
            correction="ki",
            init_orbitals="mlwfs",
            group_orbitals_tol=0.05,
        )
        inp = KoopmansInput.model_validate(d)
        assert inp.workflow.group_orbitals_by == GroupOrbitalsBy.NONE
        assert advisories_for(inp) == [
            "workflow.group_orbitals_tol has no effect on task: singlepoint "
            "(group_orbitals_by resolved to 'none' for init_orbitals: mlwfs, "
            "formalism: dfpt); it is kept for when you set group_orbitals_by "
            "to a criterion this run implements (self_hartree for DSCF, spread for "
            "DFPT)."
        ]

    @pytest.mark.parametrize(
        ("task", "workflow_updates"),
        [
            ("wannierize", {}),
            ("wannierize", {"init_orbitals": "mlwfs"}),
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
            formalism="dscf",
            correction="ki",
            init_orbitals="mlwfs",
        )
        inp = KoopmansInput.model_validate(d)
        assert inp.workflow.group_orbitals_by == GroupOrbitalsBy.SELF_HARTREE
        assert inp.workflow.group_orbitals_tol == pytest.approx(1.0e-4)
        assert advisories_for(inp) == []

    def test_bse_advisory_matches_the_equivalent_dfpt_singlepoint(self) -> None:
        """`bse` now shares the DFPT route's own tolerance advisory, task name aside.

        ``group_orbitals_tol`` reaches `bse`'s composed DFPT chain
        unchanged (aiida-koopmans#137), so an orphaned tolerance is advised
        with the formalism fixed at ``dfpt``, as under ``task: singlepoint`` with the same
        ``formalism``/``init_orbitals`` — not the "switch task"
        message ``wannierize``/``dft_bands``/``dft_eps`` get, since those
        compose no DFPT chain at all.
        """
        common = {
            "formalism": "dfpt",
            "correction": "ki",
            "init_orbitals": "mlwfs",
            "group_orbitals_tol": 0.05,
        }
        bse_input = KoopmansInput.model_validate(_bse_dict(**common))
        singlepoint_input = KoopmansInput.model_validate(_si_dict("singlepoint", **common))
        assert bse_input.workflow.group_orbitals_by == GroupOrbitalsBy.NONE
        assert advisories_for(bse_input) == [
            "workflow.group_orbitals_tol has no effect on task: bse "
            "(group_orbitals_by resolved to 'none' for init_orbitals: mlwfs, "
            "formalism: dfpt); it is kept for when you set group_orbitals_by "
            "to a criterion this run implements (spread)."
        ]
        assert advisories_for(singlepoint_input) != []


class TestResolveGroupOrbitalsByDoesNotMutateCaller:
    """``resolve_group_orbitals_by`` must not write resolved fields back onto the caller's dict.

    Regression: the before-validator used to write ``group_orbitals_by``/
    ``group_orbitals_tol`` directly into the dict passed to
    ``model_validate``, so re-validating the same dict after editing it saw
    the previous parse's resolved values as if the user had typed them.
    """

    def test_callers_dict_is_untouched(self) -> None:
        """A resolved criterion and tolerance never land back in the caller's dict."""
        d = _si_dict("singlepoint", init_orbitals="mlwfs")
        inp = KoopmansInput.model_validate(d)
        assert inp.workflow.group_orbitals_by == GroupOrbitalsBy.SELF_HARTREE
        assert "group_orbitals_by" not in d["workflow"]
        assert "group_orbitals_tol" not in d["workflow"]

    def test_editing_and_revalidating_resolves_afresh(self) -> None:
        """Flipping ``init_orbitals`` and re-validating must not inherit the earlier resolution."""
        d = _si_dict("singlepoint", init_orbitals="mlwfs")
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
            "singlepoint); it is kept for when you switch task to "
            "singlepoint."
        ]


class TestBothAdvisoriesTogether:
    """The two checks are independent and both surface on the same input."""

    def test_two_orphaned_keywords_yield_two_messages(self) -> None:
        """One message per keyword left with no effect, not a merged one."""
        d = _with_overrides(_si_dict("wannierize"), wannierize_smooth={"factor": 2})
        d["workflow"]["group_orbitals_tol"] = 0.05
        inp = KoopmansInput.model_validate(d)
        assert len(advisories_for(inp)) == 2
