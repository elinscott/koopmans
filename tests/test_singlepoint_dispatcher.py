"""Tests for the Koopmans singlepoint dispatcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from koopmans.aiida.workflows import advisories_for
from koopmans.aiida.workflows.dscf import (
    _KcpDscfInputs,
    build_singlepoint_workgraph,
    kcp_dscf_inputs,
)
from koopmans.input_file import KoopmansInput, read_input_file
from koopmans.input_file.atomic_positions import AtomicPositionsInput
from koopmans.input_file.workflow import (
    Correction,
    Formalism,
    Task,
    VariationalOrbitalType,
)


@pytest.fixture
def ozone_yaml(tutorials_dir: Path) -> Path:
    """Return the path to the ozone tutorial input file."""
    return tutorials_dir / "orbital_energies/ozone/ozone.yaml"


@pytest.fixture
def ozone_input(ozone_yaml: Path) -> KoopmansInput:
    """Return a freshly parsed ozone ``KoopmansInput``."""
    return read_input_file(ozone_yaml)


def _copy_with_calc_overrides(inp: KoopmansInput, **calc_param_updates: object) -> KoopmansInput:
    """Return a fresh ``KoopmansInput`` with patched ``calculator_parameters``.

    Each key in ``calc_param_updates`` is a dotted path into
    ``calculator_parameters`` (e.g. ``"kcp.system.nbnd"``). Pydantic's
    ``model_copy`` is shallow, so we dump, mutate, and re-validate.
    """
    d = inp.model_dump()
    calc = d["calculator_parameters"]
    for dotted, value in calc_param_updates.items():
        target = calc
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return KoopmansInput.model_validate(d)


class TestOzoneInputParse:
    """Parsing checks for the ozone tutorial JSON."""

    def test_ozone_parses_with_expected_workflow_fields(
        self, ozone_yaml: Path, ozone_input: KoopmansInput
    ) -> None:
        """The ozone tutorial should parse with the expected workflow fields."""
        assert ozone_yaml.exists(), f"Tutorial file not found: {ozone_yaml}"

        assert isinstance(ozone_input, KoopmansInput)
        assert ozone_input.workflow.task == Task.SINGLEPOINT
        assert ozone_input.workflow.correction == Correction.KI
        assert ozone_input.workflow.formalism == Formalism.DSCF
        assert ozone_input.workflow.init_orbitals == VariationalOrbitalType.KOHN_SHAM
        assert ozone_input.workflow.alpha_numsteps == 1
        assert ozone_input.workflow.pseudo_library == "SG15/1.2/PBE/SR"

    def test_ozone_is_non_periodic(self, ozone_input: KoopmansInput) -> None:
        """The ozone tutorial should be non-periodic (molecule in a box).

        The file writes a single ``periodic: false``, which validation
        expands to one entry per cell vector.
        """
        assert ozone_input.atoms.cell_parameters.periodic == (False, False, False)

    def test_ozone_has_three_oxygen_atoms(self, ozone_input: KoopmansInput) -> None:
        """The ozone tutorial should have three oxygens at the expected positions."""
        atomic_positions = ozone_input.atoms.atomic_positions
        assert isinstance(atomic_positions, AtomicPositionsInput)
        positions = atomic_positions.positions
        assert len(positions) == 3
        assert all(atom[0] == "O" for atom in positions)

        expected = [
            ("O", 4.0869, 3.0, 2.89),
            ("O", 5.1738, 3.0, 3.55),
            ("O", 3.0, 3.0, 3.55),
        ]
        for got, want in zip(positions, expected, strict=True):
            assert got[0] == want[0]
            assert got[1] == pytest.approx(want[1])
            assert got[2] == pytest.approx(want[2])
            assert got[3] == pytest.approx(want[3])


def _scalars(inputs: _KcpDscfInputs) -> tuple[float, float, int, int]:
    """Project the (ecutwfc, ecutrho, nbnd, nspin) corner of a kcp input bundle."""
    return (inputs["ecutwfc"], inputs["ecutrho"], inputs["nbnd"], inputs["nspin"])


class TestKcpDscfInputs:
    """Unit tests for ``kcp_dscf_inputs``."""

    def test_ozone_default(self, ozone_input: KoopmansInput) -> None:
        """Ozone input should yield (50.0, 200.0, 10, 2)."""
        assert _scalars(kcp_dscf_inputs(ozone_input)) == (50.0, 200.0, 10, 2)

    def test_ecutrho_default_with_custom_ecutwfc(self, ozone_input: KoopmansInput) -> None:
        """With ecutwfc=30, ecutrho should follow at 120.0 (4 * 30)."""
        inp = _copy_with_calc_overrides(ozone_input, ecutwfc=30.0)
        assert _scalars(kcp_dscf_inputs(inp)) == (30.0, 120.0, 10, 2)

    def test_missing_ecutwfc_raises_valueerror(self, ozone_input: KoopmansInput) -> None:
        """A missing ecutwfc should raise ValueError."""
        inp = _copy_with_calc_overrides(ozone_input, ecutwfc=None)
        with pytest.raises(ValueError, match="ecutwfc is required"):
            kcp_dscf_inputs(inp)

    def test_missing_nbnd_raises_valueerror(self, ozone_input: KoopmansInput) -> None:
        """Missing nbnd (both top-level and kcp.system) should raise ValueError."""
        inp = _copy_with_calc_overrides(
            ozone_input,
            nbnd=None,
            **{"kcp.system.nbnd": None},
        )
        with pytest.raises(ValueError, match="nbnd is required"):
            kcp_dscf_inputs(inp)

    def test_workflow_fields_forwarded(self, ozone_input: KoopmansInput) -> None:
        """The workflow-level fields should land in the bundle unchanged."""
        inputs = kcp_dscf_inputs(ozone_input)
        workflow = ozone_input.workflow
        assert inputs["pseudo_family"] == workflow.pseudo_library
        assert inputs["correction"] == workflow.correction
        assert inputs["init_orbitals"] == workflow.init_orbitals
        assert inputs["alpha_numsteps"] == workflow.alpha_numsteps
        assert inputs["initial_alpha"] == workflow.alpha_guess


class TestPwAndKcpCutoffsAgree:
    """The one ``ecutwfc`` field always puts pw.x and kcp.x on the same grid."""

    def test_both_codes_receive_the_same_pair(
        self, ozone_input: KoopmansInput, aiida_profile: object
    ) -> None:
        """Reads the numbers each code is handed, rather than trusting the schema alone.

        pw.x resolves its pair from the ``pw`` block and kcp.x from the
        ``kcp`` block, by separate code paths that a shared input value is no
        guarantee of.
        """
        from koopmans.aiida.conversion import input_to_pw_parameters

        inp = _copy_with_calc_overrides(ozone_input, ecutwfc=45.0)

        pw_system = input_to_pw_parameters(inp)["SYSTEM"]
        kcp = kcp_dscf_inputs(inp)

        assert pw_system["ecutwfc"] == pytest.approx(kcp["ecutwfc"])
        assert pw_system["ecutrho"] == pytest.approx(kcp["ecutrho"])
        assert (kcp["ecutwfc"], kcp["ecutrho"]) == pytest.approx((45.0, 180.0))


class TestBuildSinglepointWorkgraphScopeGuards:
    """Scope-guard tests for ``build_singlepoint_workgraph``.

    These fire BEFORE ``ensure_pseudo_family_installed``, so they are testable
    without an AiiDA profile or database.
    """

    @pytest.mark.parametrize(
        "correction_value",
        ["pkipz", "none", "all"],
    )
    def test_unsupported_correction_raises_notimplemented(
        self, ozone_input: KoopmansInput, correction_value: str
    ) -> None:
        """Only KI/KIPZ are implemented; other corrections should raise NotImplementedError."""
        d = ozone_input.model_dump()
        d["workflow"]["correction"] = correction_value
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match="correction="):
            build_singlepoint_workgraph(inp)

    @pytest.mark.parametrize("correction_value", ["kipz", "pkipz", "none", "all"])
    def test_dfpt_rejects_non_ki_corrections(
        self, ozone_input: KoopmansInput, correction_value: str
    ) -> None:
        """The DFPT route (kcw.x) implements KI only; anything else must raise loudly.

        Guards against silently running KI physics for a requested KIPZ (etc.)
        correction: the DFPT branch dispatches on ``formalism`` before
        the DSCF correction guard is reached.
        """
        d = ozone_input.model_dump()
        d["workflow"]["formalism"] = "dfpt"
        d["workflow"]["correction"] = correction_value
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match="only implements the KI correction"):
            build_singlepoint_workgraph(inp)

    @pytest.mark.parametrize("formalism", ["dscf", "dfpt"])
    def test_external_projectors_rejected(self, ozone_input: KoopmansInput, formalism: str) -> None:
        """``atom_proj_ext`` is rejected on both singlepoint streams.

        Neither stream consults the external projector keywords when it
        builds its Wannierization, so accepting the switch would silently
        drop it.
        """
        d = ozone_input.model_dump()
        d["workflow"]["formalism"] = formalism
        d["calculator_parameters"]["pw2wannier90"] = {"atom_proj_ext": True}
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match="not wired into the singlepoint route"):
            build_singlepoint_workgraph(inp)

    @pytest.mark.parametrize("task", ["singlepoint", "dft_bands", "dft_eps"])
    def test_auto_projections_rejected_outside_wannierize(
        self, ozone_input: KoopmansInput, task: str
    ) -> None:
        """``workflow.auto_projections`` is rejected before dispatch on every other task.

        The internal Wannierizations of these routes require explicit
        projections, so accepting the flag would silently drop it. The
        guard runs before any codes are loaded.
        """
        from koopmans.aiida.workflows import build_workgraph

        d = ozone_input.model_dump()
        d["workflow"]["task"] = task
        d["workflow"]["auto_projections"] = True
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match=f"not wired into the {task} route"):
            build_workgraph(inp)

    @pytest.mark.parametrize("task", ["singlepoint", "dft_bands", "dft_eps"])
    def test_dfpt_coarse_grid_rejected(self, ozone_input: KoopmansInput, task: str) -> None:
        """``workflow.dfpt_coarse_grid`` is rejected before dispatch on every task.

        No route reads it, so accepting it would silently run the dense grid
        alone.
        """
        from koopmans.aiida.workflows import build_workgraph

        d = ozone_input.model_dump()
        d["workflow"]["task"] = task
        d["workflow"]["dfpt_coarse_grid"] = (2, 2, 2)
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match=r"dfpt_coarse_grid.*not wired"):
            build_workgraph(inp)


class TestSmoothWannierizationAdvisedOffSinglepoint:
    """``kpoints.overrides.wannierize_smooth`` only shapes a singlepoint or bse band interpolation.

    Every other task runs no smooth Wannierization, so the entry takes no
    effect there. It is a non-fatal advisory
    (:func:`koopmans.aiida.workflows.advisories_for`), so the route builds
    exactly as it would without the entry.
    """

    _ADVISORY = (
        "kpoints.overrides.wannierize_smooth has no effect on this run: only bse and a "
        "singlepoint initialized from Wannier functions (init_orbitals: mlwfs or projwfs) "
        "run the smooth Wannierization; it is kept for when you switch to one."
    )

    @staticmethod
    def _si_dict(task: str, **workflow_updates: Any) -> dict[str, Any]:
        d: dict[str, Any] = {
            "workflow": {"task": task, "pseudo_library": "SG15/1.2/PBE/SR"},
            "atoms": {
                "cell_parameters": {"periodic": True, "ibrav": 2, "celldms": {"1": 10.2622}},
                "atomic_positions": {
                    "units": "crystal",
                    "positions": [["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25]],
                },
            },
            "kpoints": {
                "grid": [2, 2, 2],
                "offset": [0, 0, 0],
                "overrides": {"wannierize_smooth": {"factor": 2}},
            },
            "calculator_parameters": {"ecutwfc": 20.0},
        }
        d["workflow"].update(workflow_updates)
        return d

    def test_dft_bands_builds_and_is_advised(
        self, aiida_profile: Any, installed_pw_code: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """dft_bands builds normally; the factor is flagged, not refused."""
        from koopmans.aiida.workflows import advisories_for, build_workgraph

        inp = KoopmansInput.model_validate(self._si_dict("dft_bands"))
        wg = build_workgraph(inp)
        assert wg is not None

        assert advisories_for(inp) == [self._ADVISORY]

    def test_wannierize_builds_and_is_advised(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_wannier_codes: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """Wannierize builds normally; the factor is flagged, not refused."""
        from koopmans.aiida.workflows import advisories_for, build_workgraph

        d = self._si_dict("wannierize")
        d["calculator_parameters"]["wannier90"] = {
            "projections": [[{"site": "Si", "ang_mtm": "sp3"}]]
        }
        inp = KoopmansInput.model_validate(d)
        wg = build_workgraph(inp)
        assert wg is not None
        assert "wannierize_smooth" not in wg.get_task_names()

        assert advisories_for(inp) == [self._ADVISORY]

    def test_no_entry_is_not_advised(self) -> None:
        """Negative control: the advisory itself is silent without the entry."""
        from koopmans.aiida.workflows import advisories_for

        d = self._si_dict("dft_bands")
        del d["kpoints"]["overrides"]
        inp = KoopmansInput.model_validate(d)
        assert advisories_for(inp) == []

    def test_gamma_only_dscf_route_advises_the_entry(self, ozone_input: KoopmansInput) -> None:
        """A gamma-only run has no mesh to densify; the entry waits for a k-point grid."""
        from koopmans.aiida.conversion import smooth_factor
        from koopmans.aiida.workflows import advisories_for

        d = ozone_input.model_dump()
        d["kpoints"]["overrides"]["wannierize_smooth"] = {"factor": 2}
        inp = KoopmansInput.model_validate(d)
        assert smooth_factor(inp.kpoints) is None
        assert advisories_for(inp) == [
            "kpoints.overrides.wannierize_smooth has no effect on a gamma-only run, which "
            "samples Gamma alone; it is kept for when you switch to a k-point grid."
        ]


class TestEpsInfOnAMolecule:
    """A molecule applies no Makov-Payne correction, so ``eps_inf`` is kept but inert."""

    @pytest.mark.parametrize("eps_inf", ["auto", 11.7])
    def test_any_eps_inf_builds_no_dielectric_step_and_is_advised(
        self,
        ozone_input: KoopmansInput,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_kcp_code: Any,
        fake_sg15_pseudo_family: Any,
        eps_inf: float | str,
    ) -> None:
        """The build succeeds without a ``dielectric`` task, and the advisory says why."""
        d = ozone_input.model_dump()
        d["workflow"]["eps_inf"] = eps_inf
        inp = KoopmansInput.model_validate(d)

        wg = build_singlepoint_workgraph(inp)

        assert "dielectric" not in [t.name for t in wg.tasks]
        assert any("isolated molecule" in a for a in advisories_for(inp))


class TestExplicitOrbitalGroupsRejected:
    """An explicit ``orbital_groups`` list is not wired into the screening fan-out.

    The field parses and validates but reaches nothing downstream, so the
    dispatcher must reject it loudly rather than silently grouping by the
    resolved criterion instead.
    """

    def test_dscf_kcp_inputs_reject_orbital_groups(self, ozone_input: KoopmansInput) -> None:
        """The DSCF (kcp.x) grouping choke point rejects an explicit grouping."""
        d = ozone_input.model_dump()
        d["workflow"]["orbital_groups"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match="orbital_groups are not yet threaded"):
            kcp_dscf_inputs(inp)

    def test_dscf_without_orbital_groups_does_not_raise(self, ozone_input: KoopmansInput) -> None:
        """Negative control: the guard fires only when ``orbital_groups`` is set."""
        assert ozone_input.workflow.orbital_groups is None
        kcp_dscf_inputs(ozone_input)  # must not raise

    def test_dfpt_dispatcher_rejects_orbital_groups(self, ozone_input: KoopmansInput) -> None:
        """A tutorial-2-shaped DFPT input with ``orbital_groups`` must raise loudly.

        Mirrors ``tutorials/tutorial_2/si.json`` (DFPT + KI + MLWFs +
        ``orbital_groups``); the guard fires at the DFPT grouping choke point,
        before any structure conversion or pseudopotential install.
        """
        d = ozone_input.model_dump()
        d["workflow"]["formalism"] = "dfpt"
        d["workflow"]["correction"] = "ki"
        d["workflow"]["init_orbitals"] = "mlwfs"
        d["workflow"]["init_empty_orbitals"] = "mlwfs"
        d["workflow"]["orbital_groups"] = [0, 0, 0, 0, 1, 1, 1, 1]
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match="orbital_groups are not yet threaded"):
            build_singlepoint_workgraph(inp)


class TestInitialAlphaFromGuess:
    """The DSCF route seeds one alpha for all orbitals; guard the list handling."""

    def test_scalar_and_uniform_list_pass_through(self) -> None:
        """A scalar or an all-equal list collapses to that value."""
        from koopmans.aiida.workflows.dscf import _initial_alpha_from_guess

        assert _initial_alpha_from_guess(0.3) == 0.3
        assert _initial_alpha_from_guess([0.3, 0.3, 0.3]) == 0.3

    def test_distinct_per_orbital_values_raise(self) -> None:
        """Distinct per-orbital guesses must not be silently collapsed to the first."""
        from koopmans.aiida.workflows.dscf import _initial_alpha_from_guess

        with pytest.raises(NotImplementedError, match="per-orbital alpha_guess"):
            _initial_alpha_from_guess([0.3, 0.5, 0.7])
