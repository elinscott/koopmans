"""Tests for the Koopmans singlepoint dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from koopmans.aiida.workflows.dscf import (
    _KcpDscfInputs,
    build_singlepoint_workgraph,
    kcp_dscf_inputs,
)
from koopmans.input_file import KoopmansInput, read_input_file
from koopmans.input_file.atomic_positions import AtomicPositionsInput
from koopmans.input_file.workflow import (
    CalculateScreeningMethod,
    Correction,
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
    ``calculator_parameters`` (e.g. ``"kcp.system.ecutrho"``). Pydantic's
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
        assert ozone_input.workflow.screening_method == CalculateScreeningMethod.DSCF
        assert ozone_input.workflow.init_orbitals == VariationalOrbitalType.KOHN_SHAM
        assert ozone_input.workflow.alpha_numsteps == 1
        assert ozone_input.workflow.pseudo_library == "SG15/1.2/PBE/SR"

    def test_ozone_is_non_periodic(self, ozone_input: KoopmansInput) -> None:
        """The ozone tutorial should be non-periodic (molecule in a box)."""
        assert ozone_input.atoms.cell_parameters.periodic is False

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

    def test_ecutrho_defaults_to_four_times_ecutwfc(self, ozone_input: KoopmansInput) -> None:
        """With ecutrho unset, it should default to 4 * ecutwfc (4 * 50 = 200)."""
        inp = _copy_with_calc_overrides(ozone_input, **{"kcp.system.ecutrho": 0.0})
        assert _scalars(kcp_dscf_inputs(inp)) == (50.0, 200.0, 10, 2)

    def test_ecutrho_default_with_custom_ecutwfc(self, ozone_input: KoopmansInput) -> None:
        """With ecutwfc=30 and no ecutrho, ecutrho should fall back to 120.0."""
        inp = _copy_with_calc_overrides(
            ozone_input,
            ecutwfc=30.0,
            **{"kcp.system.ecutrho": 0.0},
        )
        assert _scalars(kcp_dscf_inputs(inp)) == (30.0, 120.0, 10, 2)

    def test_missing_ecutwfc_raises_valueerror(self, ozone_input: KoopmansInput) -> None:
        """Missing ecutwfc (both top-level and kcp.system) should raise ValueError."""
        inp = _copy_with_calc_overrides(
            ozone_input,
            ecutwfc=None,
            **{"kcp.system.ecutwfc": 0.0},
        )
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


_ACCEPTED_CUTOFF_SHAPES = [
    ("the shorthand alone", {"ecutwfc": 45.0}),
    ("a pw block restating the shorthand", {"ecutwfc": 45.0, "pw.system.ecutwfc": 45.0}),
    ("pw and kcp stating one cutoff", {"pw.system.ecutwfc": 45.0, "kcp.system.ecutwfc": 45.0}),
    ("a kcp density cutoff", {"ecutwfc": 45.0, "kcp.system.ecutrho": 180.0}),
    ("a pw density cutoff", {"ecutwfc": 45.0, "pw.system.ecutrho": 180.0}),
]


class TestPwAndKcpCutoffsAgree:
    """Every input the schema accepts puts pw.x and kcp.x on one grid."""

    @pytest.mark.parametrize(
        ("label", "overrides"),
        _ACCEPTED_CUTOFF_SHAPES,
        ids=[case[0] for case in _ACCEPTED_CUTOFF_SHAPES],
    )
    def test_both_codes_receive_the_same_pair(
        self,
        ozone_input: KoopmansInput,
        aiida_profile: object,
        label: str,
        overrides: dict[str, float],
    ) -> None:
        """Reads the numbers each code is handed, rather than trusting the parse check.

        The schema rejects an input that states two different cutoffs, which
        says nothing about the ones it accepts: pw.x resolves its pair from the
        ``pw`` block and kcp.x from the ``kcp`` block, by separate rules that a
        shared input value is no guarantee of. Each shape below reaches the two
        resolvers by a different route.
        """
        from koopmans.aiida.conversion import input_to_pw_parameters

        # The tutorial's own shorthand is cleared first, so each shape states
        # only the keys it names.
        inp = _copy_with_calc_overrides(ozone_input, **{"ecutwfc": None, **overrides})

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
            build_singlepoint_workgraph(inp, codes={}, parallelization={})

    @pytest.mark.parametrize("correction_value", ["kipz", "pkipz", "none", "all"])
    def test_dfpt_rejects_non_ki_corrections(
        self, ozone_input: KoopmansInput, correction_value: str
    ) -> None:
        """The DFPT route (kcw.x) implements KI only; anything else must raise loudly.

        Guards against silently running KI physics for a requested KIPZ (etc.)
        correction: the DFPT branch dispatches on ``screening_method`` before
        the DSCF correction guard is reached.
        """
        d = ozone_input.model_dump()
        d["workflow"]["screening_method"] = "dfpt"
        d["workflow"]["correction"] = correction_value
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match="only implements the KI correction"):
            build_singlepoint_workgraph(inp, codes={}, parallelization={})

    @pytest.mark.parametrize("screening_method", ["dscf", "dfpt"])
    def test_external_projectors_rejected(
        self, ozone_input: KoopmansInput, screening_method: str
    ) -> None:
        """``atom_proj_ext`` is rejected on both singlepoint streams.

        Neither stream consults the external projector keywords when it
        builds its Wannierization, so accepting the switch would silently
        drop it.
        """
        d = ozone_input.model_dump()
        d["workflow"]["screening_method"] = screening_method
        d["calculator_parameters"]["pw2wannier90"] = {"atom_proj_ext": True}
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match="not wired into the singlepoint route"):
            build_singlepoint_workgraph(inp, codes={}, parallelization={})

    @pytest.mark.parametrize("task", ["singlepoint", "dft_bands", "trajectory", "dft_eps"])
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
        d["workflow"]["screening_method"] = "dfpt"
        d["workflow"]["correction"] = "ki"
        d["workflow"]["init_orbitals"] = "mlwfs"
        d["workflow"]["init_empty_orbitals"] = "mlwfs"
        d["workflow"]["orbital_groups"] = [0, 0, 0, 0, 1, 1, 1, 1]
        inp = KoopmansInput.model_validate(d)

        with pytest.raises(NotImplementedError, match="orbital_groups are not yet threaded"):
            build_singlepoint_workgraph(inp, codes={}, parallelization={})


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
