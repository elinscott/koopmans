"""Dispatcher tests for the Bethe-Salpeter (yambo BSE on Koopmans eigenvalues) route.

Builds real ``WorkGraph`` objects through ``build_bse_workgraph`` against a
throwaway profile (dummy codes, fake pseudos; nothing runs). The route
composes the same ground-state/wannierization/screening chain the DFPT
singlepoint route tests in ``test_dfpt_dispatcher.py`` exercise directly, so
this file borrows its silicon input dict (``_si_dfpt_dict``) rather than
duplicating it, and focuses on what the BSE route adds: the
``calculator_parameters.yambo`` block's mapping onto the yambo runcard, the
remaining scope guards (``screening_method``, ``spin``, manifold-reaching
``BSEBands``), that ``eps_inf``/``gb_correction``/``kcw`` overrides and
orbital grouping pass straight through to the composed DFPT chain, and the
derived ``ecutwfc``/``ecutrho`` a caller who states neither still gets.
"""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.aiida.workflows.bse import build_bse_workgraph
from koopmans.input_file import KoopmansInput
from tests.test_dfpt_dispatcher import _si_dfpt_dict


def _si_bse_dict(**yambo_updates: Any) -> dict[str, Any]:
    """Return ``_si_dfpt_dict()`` routed through ``task: bse``, with a ``yambo`` block.

    ``BSEBands=[1, 4]`` matches ``_si_dfpt_dict``'s own manifold: one occupied
    block (sp hybrids on 2 Si sites, 4 Wannier functions) and no empty
    block, so the Wannierized range this route reads kcw.x eigenvalues from
    is exactly bands 1-4.
    """
    d = _si_dfpt_dict()
    d["workflow"]["task"] = "bse"
    d["calculator_parameters"]["yambo"] = {
        "BndsRnXs": [1, 100],
        "NGsBlkXs": 2,
        "BSEBands": [1, 4],
        "BEnRange": [0, 10],
        **yambo_updates,
    }
    return d


def _build(d: dict[str, Any]) -> Any:
    inp = KoopmansInput.model_validate(d)
    return build_bse_workgraph(inp)


@pytest.fixture
def bse_codes(
    installed_pw_code: Any,
    installed_kcw_code: Any,
    installed_wannier_codes: Any,
    installed_bse_codes: Any,
) -> dict[str, Any]:
    """Register every dummy code the BSE route resolves as ``<name>@localhost``."""
    return {
        "pw": installed_pw_code,
        "kcw": installed_kcw_code,
        **installed_wannier_codes,
        **installed_bse_codes,
    }


class TestBuild:
    """The composed graph carries both the DFPT chain and the yambo BSE steps."""

    def test_composes_dfpt_and_bse(
        self, aiida_profile: Any, bse_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The composed DFPT chain, its channel selector, and the BSE step all appear.

        ``SinglepointBetheSalpeterWorkflow`` nests the whole DFPT chain as
        one ``dfpt`` task (its own inner ``wannierize``/``scf_nscf`` tasks
        stay inside that nested graph, as ``TestProjwfcQualityCheck`` in
        ``test_dfpt_dispatcher.py`` notes for the same nesting) — so this
        outer graph's own task list holds ``dfpt``, ``select_channel`` and
        ``bse``, not the DFPT chain's internals.
        """
        wg = _build(_si_bse_dict())
        names = wg.get_task_names()
        assert "dfpt" in names
        assert "select_channel" in names
        assert "bse" in names

    def test_bse_parameters_match_the_schema_mapping(
        self, aiida_profile: Any, bse_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The ``bse`` task's ``bse_parameters`` socket is exactly the documented mapping.

        The discriminating check for the conversion wiring: every keyword
        the schema maps, and nothing this route owns for itself.
        """
        wg = _build(_si_bse_dict(NGsBlkXs=3, BndsRnXs=[1, 80], BEnSteps=200))
        bse_parameters = wg.tasks["bse"].inputs["bse_parameters"].value
        assert bse_parameters == {
            "arguments": ["rim_cut", "WRbsWF", "NLCC"],
            "variables": {
                "BndsRnXs": [[1, 80], ""],
                "NGsBlkXs": [3, "Ry"],
                "BSENGBlk": [3, "Ry"],
                "BSEBands": [[1, 4], ""],
                "BEnRange": [[0, 10], "eV"],
                "BEnSteps": [200, ""],
                "BDmRange": [[0.1, 0.1], "eV"],
            },
        }

    def test_eigenvalues_reaches_the_graph(
        self, aiida_profile: Any, bse_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """``build_bse_workgraph`` always passes ``eigenvalues='ki'`` to ``RunBetheSalpeter``.

        The flavour knob returns once a producer exists (aiida-koopmans#134);
        for now the route hardcodes it rather than exposing a keyword nothing
        can act on.
        """
        wg = _build(_si_bse_dict())
        assert wg.tasks["bse"].inputs["eigenvalues"].value == "ki"

    def test_manifold_reaches_dfpt_unchanged(
        self, aiida_profile: Any, bse_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The composed nested ``SinglepointDFPTWorkflow`` call sees the plain-DFPT manifold.

        ``dfpt`` here is the *composed* call to ``SinglepointDFPTWorkflow``
        (the whole DFPT chain as a single nested task), so its own inputs
        are that graph's top-level arguments — including ``manifolds``
        unchanged from ``assemble_dfpt_chain_inputs``, one occupied block
        of 4 orbitals.
        """
        wg = _build(_si_bse_dict())
        manifolds = wg.tasks["dfpt"].inputs["manifolds"].value
        assert set(manifolds) == {"none"}
        assert len(manifolds["none"]["occ"]) == 1
        assert "emp" not in manifolds["none"]

    def test_missing_ecutwfc_derives_from_the_pseudo_family(
        self, aiida_profile: Any, bse_codes: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Left unset, ecutwfc/ecutrho come from the family's own recommendation.

        Checked against the family's own ``get_recommended_cutoffs`` call
        directly (same elements, same ``unit='Ry'``) rather than a
        hardcoded converted constant: ``fake_sg15_cutoffs_family`` stores
        30.0/240.0 in aiida-pseudo's own default unit, eV, not Ry, so the
        Ry value this route needs is neither of those literals. The
        discriminating check against
        ``test_missing_ecutwfc_is_refused_against_a_cutoff_less_family``,
        which uses a family with no recommendation to derive from instead.
        ``RunBetheSalpeter``'s own fresh scf/nscf reads the literal value
        straight off the shared overrides ``assemble_dfpt_chain_inputs``
        builds (see ``bse._ensure_explicit_pw_cutoffs``).
        """
        d = _si_bse_dict()
        d["workflow"]["pseudo_library"] = "SG15/1.0/PBE/SR"
        del d["calculator_parameters"]["ecutwfc"]
        wg = _build(d)
        expected_ecutwfc, expected_ecutrho = fake_sg15_cutoffs_family.get_recommended_cutoffs(
            elements=("Si",), unit="Ry"
        )
        assert wg.tasks["bse"].inputs["ecutwfc"].value == pytest.approx(expected_ecutwfc)
        assert wg.tasks["bse"].inputs["ecutrho"].value == pytest.approx(expected_ecutrho)

    def test_orbital_grouping_reaches_the_composed_dfpt_chain(
        self, aiida_profile: Any, bse_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """A resolved grouping criterion threads into the composed DFPT chain's own input.

        Mirrors ``TestOrbitalGrouping.test_spread_honours_an_explicit_tolerance``
        in ``test_dfpt_dispatcher.py``: the ``bse`` route shares
        ``assemble_dfpt_chain_inputs`` with the plain DFPT route, and now
        forwards ``group_orbitals_tol`` into ``SinglepointBetheSalpeterWorkflow``
        unchanged, so a spread-grouped BSE run groups exactly as the same
        input would under ``task: singlepoint``.
        """
        d = _si_bse_dict()
        d["workflow"]["group_orbitals_by"] = "spread"
        d["workflow"]["group_orbitals_tol"] = 0.05
        wg = _build(d)
        assert wg.tasks["dfpt"].inputs["group_orbitals_tol"].value == 0.05

    def test_eps_inf_reaches_the_composed_dfpt_chain(
        self, aiida_profile: Any, bse_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """``workflow.eps_inf`` lands on the composed ``dfpt`` task's own input, unchanged."""
        d = _si_bse_dict()
        d["workflow"]["eps_inf"] = 5.3
        wg = _build(d)
        assert wg.tasks["dfpt"].inputs["eps_inf"].value == pytest.approx(5.3)

    def test_gb_correction_reaches_the_composed_dfpt_chain(
        self, aiida_profile: Any, bse_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """``workflow.gb_correction`` lands on the composed ``dfpt`` task's ``l_vcut`` input."""
        d = _si_bse_dict()
        d["workflow"]["gb_correction"] = True
        wg = _build(d)
        assert bool(wg.tasks["dfpt"].inputs["l_vcut"].value)

    def test_kcw_overrides_reach_the_composed_dfpt_chain(
        self, aiida_profile: Any, bse_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """``calculator_parameters.kcw`` lands on the composed ``dfpt`` task, split per namelist.

        Mirrors ``TestKcwOverrides.test_keywords_reach_dfpt_split_per_namelist``
        in ``test_dfpt_dispatcher.py``.
        """
        d = _si_bse_dict()
        d["calculator_parameters"]["kcw"] = {
            "control": {"lrpa": True},
            "screen": {"tr2": 1.0e-16},
        }
        wg = _build(d)
        overrides = wg.tasks["dfpt"].inputs["kcw_overrides"]
        assert overrides["control"].value == {"lrpa": True}
        assert overrides["screen"].value == {"tr2": pytest.approx(1.0e-16)}
        assert not overrides["ham"]._links


class TestRouteRefusals:
    """Scope guards for inputs the composed ``SinglepointBetheSalpeterWorkflow`` has no socket for.

    Most of these are pure-Python checks on the parsed input needing no
    profile, codes, or pseudopotentials, firing before anything is built;
    the ones that resolve a pseudo family or the Wannierized manifold say
    so in their own docstring.
    """

    def test_dscf_screening_method_is_refused(self) -> None:
        """The route reads kcw.x ``ham`` eigenvalues; a ΔSCF run produces none."""
        d = _si_bse_dict()
        d["workflow"]["screening_method"] = "dscf"
        with pytest.raises(NotImplementedError, match="screening_method"):
            _build(d)

    def test_collinear_spin_is_refused(self) -> None:
        """No single channel exists for the composed graph to read.

        Fires before the plain-DFPT collinear input-shape checks (missing
        ``w90.up``/``w90.down``, missing magnetization): a reader fixing
        those would only learn afterwards that collinear is unsupported
        here regardless.
        """
        d = _si_bse_dict()
        d["workflow"]["spin"] = "collinear"
        # A magnetization alone (no per-spin projections) so the input clears
        # KoopmansInput's own parse-time check and reaches the BSE route's
        # scope guard, not a schema error about the missing magnetization.
        d["calculator_parameters"]["tot_magnetization"] = 0
        with pytest.raises(NotImplementedError, match="spin") as excinfo:
            _build(d)
        assert "per-spin projections" not in str(excinfo.value)

    def test_missing_ecutwfc_is_refused_against_a_cutoff_less_family(
        self, aiida_profile: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """No literal cutoff, and no family recommendation to derive one from, is fatal.

        Needs a profile and pseudos: this guard runs after
        ``assemble_dfpt_chain_inputs``, whose own
        ``require_cutoffs_for_family`` check raises first for
        ``fake_sg15_pseudo_family`` (no recommended cutoffs) once
        ``calculator_parameters.ecutwfc`` is gone too.
        """
        d = _si_bse_dict()
        del d["calculator_parameters"]["ecutwfc"]
        with pytest.raises(ValueError, match="ecutwfc"):
            _build(d)

    def test_bands_reaching_past_the_manifold_is_refused(
        self, aiida_profile: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """``yambo.BSEBands`` past the Wannierized range (4 orbitals) is refused with the count."""
        d = _si_bse_dict(BSEBands=[1, 5])
        with pytest.raises(ValueError, match=r"Wannierized manifold.*1\.\.4"):
            _build(d)

    def test_bands_within_the_manifold_builds(
        self, aiida_profile: Any, bse_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Negative control: the same manifold with a range that fits builds cleanly."""
        wg = _build(_si_bse_dict(BSEBands=[1, 4]))
        assert "bse" in wg.get_task_names()
