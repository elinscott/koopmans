"""Dispatcher tests for the Bethe-Salpeter (yambo BSE on Koopmans eigenvalues) route.

Builds real ``WorkGraph`` objects through ``build_bse_workgraph`` against a
throwaway profile (dummy codes, fake pseudos; nothing runs). The route
composes the same ground-state/wannierization/screening chain the DFPT
singlepoint route tests in ``test_dfpt_dispatcher.py`` exercise directly, so
this file borrows its silicon input dict (``_si_dfpt_dict``) rather than
duplicating it, and focuses on what the BSE route adds: the ``bse`` block's
mapping onto the yambo runcard, and the scope guards the composed workflow's
missing sockets (eps_inf, gb_correction, orbital grouping, kcw overrides,
band interpolation) make necessary.
"""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.aiida.workflows.bse import build_bse_workgraph
from koopmans.input_file import KoopmansInput
from tests.test_dfpt_dispatcher import _si_dfpt_dict


def _si_bse_dict(**bse_updates: Any) -> dict[str, Any]:
    """Return ``_si_dfpt_dict()`` routed through ``task: bse``, with a ``bse`` block.

    ``bands=[1, 4]`` matches ``_si_dfpt_dict``'s own manifold: one occupied
    block (sp hybrids on 2 Si sites, 4 Wannier functions) and no empty
    block, so the Wannierized range this route reads kcw.x eigenvalues from
    is exactly bands 1-4.
    """
    d = _si_dfpt_dict()
    d["workflow"]["task"] = "bse"
    d["bse"] = {
        "screening_bands": 100,
        "g_cutoff": 2,
        "bands": [1, 4],
        "energy_range": [0, 10],
        **bse_updates,
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
        wg = _build(_si_bse_dict(g_cutoff=3, screening_bands=80, energy_steps=200))
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
        """``bse.eigenvalues`` reaches ``RunBetheSalpeter``'s own socket of the same name."""
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


class TestRouteRefusals:
    """Scope guards for inputs the composed ``SinglepointBetheSalpeterWorkflow`` has no socket for.

    Each of these is a pure-Python check on the parsed input and needs no
    profile, codes, or pseudopotentials — it fires before anything is built.
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

    def test_eps_inf_is_refused(self) -> None:
        """``eps_inf`` has no socket on the composed graph."""
        d = _si_bse_dict()
        d["workflow"]["eps_inf"] = 5.3
        with pytest.raises(NotImplementedError, match="eps_inf"):
            _build(d)

    def test_gb_correction_is_refused(self) -> None:
        """``gb_correction`` has no socket on the composed graph, whichever value it names."""
        d = _si_bse_dict()
        d["workflow"]["gb_correction"] = True
        with pytest.raises(NotImplementedError, match="gb_correction"):
            _build(d)

    def test_orbital_grouping_is_refused(self) -> None:
        """Workflow-level orbital grouping has no socket on the composed graph."""
        d = _si_bse_dict()
        d["workflow"]["group_orbitals_by"] = "spread"
        d["workflow"]["group_orbitals_tol"] = 0.05
        with pytest.raises(NotImplementedError, match="group_orbitals_by"):
            _build(d)

    def test_missing_ecutwfc_is_refused(self) -> None:
        """The fresh yambo scf/nscf must match the DFPT chain's cutoff exactly."""
        d = _si_bse_dict()
        del d["calculator_parameters"]["ecutwfc"]
        with pytest.raises(ValueError, match="ecutwfc"):
            _build(d)

    def test_kcw_overrides_are_refused(
        self, aiida_profile: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """``calculator_parameters.kcw`` has no socket on the composed graph.

        Needs a profile and pseudos: this guard runs after
        ``assemble_dfpt_chain_inputs``, which resolves the pseudopotential
        family to derive the manifold before this check ever looks at it.
        """
        d = _si_bse_dict()
        d["calculator_parameters"]["kcw"] = {"control": {"lrpa": True}}
        with pytest.raises(NotImplementedError, match=r"calculator_parameters\.kcw"):
            _build(d)

    def test_bands_reaching_past_the_manifold_is_refused(
        self, aiida_profile: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """``bse.bands`` past the Wannierized range (4 orbitals here) is refused with the count."""
        d = _si_bse_dict(bands=[1, 5])
        with pytest.raises(ValueError, match=r"Wannierized manifold.*1\.\.4"):
            _build(d)

    def test_bands_within_the_manifold_builds(
        self, aiida_profile: Any, bse_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """Negative control: the same manifold with a range that fits builds cleanly."""
        wg = _build(_si_bse_dict(bands=[1, 4]))
        assert "bse" in wg.get_task_names()
