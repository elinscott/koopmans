"""The dispatcher's TypedDict-driven code loading.

``load_codes`` reads a workflow's codes TypedDict — the plugin's single
declaration of what each workflow wires — and loads every member, required
and ``NotRequired`` alike, that has a configured ``<name>@localhost`` code;
an unconfigured member is left out, whether the route needs it or not. Which
codes a route actually needs, and whether a missing one is fatal, is the
plugin graph's own structural requirement now: a route that turns on a code
it was never given surfaces ``MissingRequiredInputsError`` at graph
validation (``check_before_run`` / submission), which ``advice_for``
translates to install advice naming the code and its declared purpose.

That structural check is not the whole story, though: some entry graphs
still bind a *required* code by direct dict subscript rather than through
``node_graph.ref`` — a plain ``codes["name"]`` inside an ``@task.graph`` body
executes immediately, during ``build()``, so an absent key would be a bare
``KeyError`` (or, for one route, an upstream library's own eager
``ValueError``) with no unlinked socket for ``check_before_run`` to ever
catch. ``require_configured_codes`` is a build-time pre-flight against
exactly this: checking a codes TypedDict's ``__required_keys__`` alone (never
``NotRequired`` — that conditional knowledge stays the socket layer's job at
submit) against the loaded mapping, and raising the same install advice
``advice_for`` would have produced, before any eager body gets the chance to
crash. ``TestPreFlightAdvice`` reproduces every route this currently
intercepts, each noting where the underlying eager bind still lives and
whether that is aiida-koopmans' to fix, upstream's, or a deliberate,
permanent choice. ``TestStructuralAdvice`` covers what is left: genuinely
``NotRequired``, input-conditional codes, which still only fail at
``check_before_run`` / submit, exactly as before.
"""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.aiida.workflows import advice_for, build_workgraph, load_codes
from koopmans.input_file import KoopmansInput
from tests.test_dscf_mlwf_dispatcher import _si_dscf_dict


class TestCodesSpecRequiredness:
    """Pin required/optional members at every codes TypedDict's definition site.

    Neither ``load_codes`` nor ``require_configured_codes`` care about this
    split at the loading step — every member is loaded when configured,
    regardless — but ``require_configured_codes`` reads ``__required_keys__``
    for its pre-flight and the plugin graphs' own structural checks read it
    too, so a ``from __future__ import annotations`` sneaking into a
    definition module (which silently flips every member to required,
    python/cpython#97727) is still worth screaming about here.
    """

    def test_required_and_optional_members(self, aiida_profile: Any) -> None:
        """Each codes TypedDict declares exactly the expected member sets.

        Requests a profile: the definition modules import workflow code,
        which loads the AiiDA configuration at import.
        """
        from aiida_koopmans.workgraphs.auto_wannierize import SplitBlockCodes
        from aiida_koopmans.workgraphs.block_wannierize import (
            WannierizeBlockCodes,
            WannierizeBlocksCodes,
        )
        from aiida_koopmans.workgraphs.dfpt import DfptCodes
        from aiida_koopmans.workgraphs.folding import FoldingCodes
        from aiida_koopmans.workgraphs.kcp import DscfCodes
        from aiida_koopmans.workgraphs.mlwf_init import MlwfInitCodes
        from aiida_koopmans.workgraphs.pdos import PdosCodes
        from aiida_koopmans.workgraphs.ph import DielectricCodes
        from aiida_koopmans.workgraphs.pw import PwBandsCodes
        from aiida_koopmans.workgraphs.wannier90 import WannierizeCodes

        wannier = {"pw", "pw2wannier90", "wannier90"}
        expected = {
            PwBandsCodes: ({"pw"}, set()),
            DielectricCodes: ({"pw", "ph"}, set()),
            WannierizeCodes: (wannier, {"projwfc"}),
            WannierizeBlockCodes: (wannier, set()),
            WannierizeBlocksCodes: (wannier, {"wannierjl", "projwfc"}),
            SplitBlockCodes: (wannier | {"wannierjl"}, set()),
            MlwfInitCodes: (wannier | {"wann2kcp", "merge_evc", "kcp"}, set()),
            FoldingCodes: ({"wann2kcp", "merge_evc"}, set()),
            DscfCodes: ({"kcp"}, wannier | {"wann2kcp", "merge_evc"}),
            DfptCodes: (wannier | {"kcw"}, {"ph", "projwfc"}),
            PdosCodes: ({"pw", "dos", "projwfc"}, set()),
        }
        for spec, (required, optional) in expected.items():
            assert set(spec.__required_keys__) == required, spec.__name__
            assert set(spec.__optional_keys__) == optional, spec.__name__


class TestLoadCodes:
    """Pass-everything: every configured member is loaded, nothing else decides."""

    def test_loads_every_configured_member(
        self, aiida_profile_clean: Any, installed_kcp_code: Any, installed_pw_code: Any
    ) -> None:
        """Required and ``NotRequired`` members are loaded alike, once configured."""
        from aiida_koopmans.workgraphs.kcp import DscfCodes

        codes = load_codes(DscfCodes)
        assert set(codes) == {"kcp", "pw"}

    def test_unconfigured_member_is_left_out_silently(self, aiida_profile_clean: Any) -> None:
        """An unconfigured member — required or not — is left out, never raised by the loader.

        The discriminating case against the pre-migration contract:
        ``load_codes`` used to raise for a missing *required* member
        itself, before a graph ever saw it. That check moved out of the
        loader entirely — an empty profile now gets an empty mapping
        back from ``load_codes`` alone (the pre-flight, a separate call
        the routes make next, is what raises now; see
        ``TestPreFlightAdvice``).
        """
        from aiida_koopmans.workgraphs.pw import PwBandsCodes

        assert load_codes(PwBandsCodes) == {}

    def test_configured_notrequired_member_now_rides_along(
        self, aiida_profile_clean: Any, installed_kcp_code: Any, installed_pw_code: Any
    ) -> None:
        """A configured ``NotRequired`` member is loaded whether the route needs it or not.

        The other half of the flip: what the graph receives is now fixed
        by which codes the profile holds, not by which the input turns
        on — the reverse of the pre-migration contract, where an
        unneeded configured code was deliberately left out. Whether a
        route actually *uses* a member it never asked for is the graph's
        own decision now, not the loader's.
        """
        from aiida_koopmans.workgraphs.kcp import DscfCodes

        codes = load_codes(DscfCodes)
        assert "pw" in codes


class TestPreFlightAdvice:
    """Every route below reaches its ``require_configured_codes`` pre-flight before ``build()``.

    Each is a *required* codes-TypedDict member missing entirely — so the
    dispatcher's pre-flight (:func:`~koopmans.aiida.workflows.require_configured_codes`,
    called right after :func:`~koopmans.aiida.workflows.load_codes` in every
    route module) raises the same install advice ``advice_for`` renders from
    a structural ``MissingRequiredInputsError``, before the route's own
    ``@task.graph`` body ever gets a chance to run. The entry-graph bind
    that *would* have crashed underneath still exists in every case; the
    pre-flight is what intercepts first now, not a fix to that bind:

    * ``aiida_koopmans.workgraphs.pw.RunPwBands`` (``pw``) and
      ``aiida_koopmans.workgraphs.ph.DielectricTask`` (``pw``/``ph``, also
      covered by ``tests/test_dft_eps_dispatcher.py``'s
      ``test_missing_ph_code_earns_preflight_advice``) feed the code
      straight into their own eager ``get_builder_from_protocol`` call.
      Fixable in aiida-koopmans alone (working precedent: ``pw.py``'s own
      ``RunScfNscf``/``assemble_pw_base_step`` pair, and aiida-koopmans#90's
      own ``RunProjwfc``, both wrap that call in a small ``@task.graph`` so
      the *caller* can ``ref()`` the code in). This is the gap
      aiida-koopmans#88 flagged against upstream node-graph#169 and said
      would "convert in a follow-up"; #90 (the ``ref()`` migration this PR
      pairs with) has not reached these call sites yet.
    * ``block_wannierize.py``'s optional quality-check-bands helper (entered
      only when a ``kpoints.path`` is given) has the identical eager-builder
      shape; without a path it is never entered, but the pre-flight still
      demands ``pw`` up front, since ``pw`` is unconditionally required by
      ``WannierizeBlocksCodes`` regardless of whether that helper runs.
    * ``aiida_koopmans.workgraphs.kcp.KoopmansDSCFWorkflow``'s ``kcp_code =
      codes["kcp"]`` bind — reached by both the singlepoint and the
      trajectory routes — is aiida-koopmans#90's own deliberate, permanent
      choice: ``kcp`` is never route-conditional, so the maintainer judged
      threading a ``ref()`` through 11 sibling functions not worth it for a
      value that never varies.
    * ``aiida_koopmans.workgraphs.wannier90.Wannierize`` (whole-manifold
      ``auto_projections``) hands its entire ``codes`` namespace to
      ``Wannier90WorkChain.get_builder_from_protocol(codes=codes, ...)`` in
      upstream ``aiida-wannier90-workflows``, whose own eager membership
      check would otherwise raise a plain ``ValueError`` before any
      aiida-koopmans or koopmans2 code runs again — not fixable in either
      repo alone.
    """

    def test_dft_bands_missing_pw_earns_preflight_advice(
        self, aiida_profile_clean: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """``RunPwBands`` binds ``codes["pw"]`` eagerly; the pre-flight names it first."""
        from tests.fixtures import silicon_pw_input

        inp = KoopmansInput.model_validate(silicon_pw_input())
        with pytest.raises(ValueError, match="`pw@localhost`") as excinfo:
            build_workgraph(inp)
        assert "koopmans install" in str(excinfo.value)

    def test_molecular_dscf_missing_kcp_earns_preflight_advice(
        self, aiida_profile_clean: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """``KoopmansDSCFWorkflow`` binds ``codes["kcp"]`` eagerly; the pre-flight is first."""
        inp = KoopmansInput.model_validate(_si_dscf_dict(init_orbitals="kohn-sham"))
        with pytest.raises(ValueError, match="`kcp@localhost`") as excinfo:
            build_workgraph(inp)
        assert "koopmans install" in str(excinfo.value)

    def test_trajectory_missing_kcp_earns_preflight_advice(
        self,
        aiida_profile_clean: Any,
        fake_sg15_pseudo_family: Any,
        tmp_path: Any,
        write_multiframe_xyz: Any,
    ) -> None:
        """The trajectory route forwards codes into the same eager kcp bind."""
        from tests.test_trajectory_dispatcher import _trajectory_input_dict

        inp = KoopmansInput.model_validate(
            _trajectory_input_dict(str(write_multiframe_xyz(tmp_path, 1)))
        )
        with pytest.raises(ValueError, match="`kcp@localhost`") as excinfo:
            build_workgraph(inp)
        assert "koopmans install" in str(excinfo.value)

    def test_wannierize_blocks_with_path_missing_pw_earns_preflight_advice(
        self,
        aiida_profile_clean: Any,
        installed_wannier_codes: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """A ``kpoints.path`` reaches the quality-check-bands helper, which bare-binds pw."""
        from tests.test_wannierize_blocks_dispatcher import _si_split_dict

        d = _si_split_dict(block_wannierization_threshold=None)
        inp = KoopmansInput.model_validate(d)
        with pytest.raises(ValueError, match="`pw@localhost`") as excinfo:
            build_workgraph(inp)
        assert "koopmans install" in str(excinfo.value)

    def test_wannierize_blocks_without_path_missing_pw_earns_preflight_advice(
        self,
        aiida_profile_clean: Any,
        installed_wannier_codes: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """Demand pw even where the eager bind is unreachable.

        ``pw`` is unconditionally required by ``WannierizeBlocksCodes``, so
        the pre-flight fires whether or not the optional quality-check-bands
        helper — the site that actually bare-binds it — would ever run.
        """
        from tests.test_wannierize_blocks_dispatcher import _si_split_dict

        d = _si_split_dict(block_wannierization_threshold=None)
        d["kpoints"].pop("path")
        inp = KoopmansInput.model_validate(d)
        with pytest.raises(ValueError, match="`pw@localhost`") as excinfo:
            build_workgraph(inp)
        assert "koopmans install" in str(excinfo.value)

    def test_wannierize_whole_auto_missing_pw_earns_preflight_advice(
        self, aiida_profile_clean: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """The pre-flight gets here before upstream's own eager ValueError can."""
        from tests.test_wannierize_blocks_dispatcher import _si_auto_dict

        d = _si_auto_dict()
        del d["workflow"]["block_wannierization_threshold"]
        inp = KoopmansInput.model_validate(d)
        with pytest.raises(ValueError, match="`pw@localhost`") as excinfo:
            build_workgraph(inp)
        assert "koopmans install" in str(excinfo.value)
        assert "does not contain the required key" not in str(excinfo.value)

    def test_dfpt_missing_pw_earns_preflight_advice(
        self,
        aiida_profile_clean: Any,
        installed_kcw_code: Any,
        installed_wannier_codes: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """Intercept pw before its several sockets can fan out.

        ``pw`` is required in ``DfptCodes``, so the pre-flight fires before
        the fan-out forms at all. What a genuinely deferred, ``NotRequired``
        member's fan-out looks like is covered synthetically in
        ``test_error_translation.py``.
        """
        from tests.test_dfpt_dispatcher import _si_dfpt_dict

        inp = KoopmansInput.model_validate(_si_dfpt_dict())
        with pytest.raises(ValueError, match="`pw@localhost`") as excinfo:
            build_workgraph(inp)
        assert "koopmans install" in str(excinfo.value)


class TestStructuralAdvice:
    """Codes the plugin graphs wire structurally: ``NotRequired``, input-conditional.

    ``require_configured_codes`` never looks at these — only
    ``__required_keys__`` — so they still build fine and fail only where the
    ``@task.graph`` body's own ``ref()``-threaded socket goes unfilled, at
    ``check_before_run`` / submit.
    """

    def test_dfpt_eps_auto_missing_ph_passes_preflight_fails_at_submit(
        self,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        installed_kcw_code: Any,
        installed_wannier_codes: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """``ph`` is ``NotRequired`` in ``DfptCodes``: the pre-flight has no opinion on it.

        Pins the design boundary the user drew explicitly: the pre-flight
        must not demand a settings-conditional member (``ph`` is only
        actually needed because this input sets ``eps_inf: auto``), so the
        build succeeds and the structural check at submit is still what
        catches it — unchanged by this PR's pre-flight addition.
        """
        from tests.test_dfpt_dispatcher import _si_dfpt_dict

        inp = KoopmansInput.model_validate(_si_dfpt_dict(eps_inf="auto"))
        wg = build_workgraph(inp)

        from aiida_workgraph.errors import MissingRequiredInputsError

        with pytest.raises(MissingRequiredInputsError) as excinfo:
            wg.check_before_run()

        advice = advice_for(excinfo.value)
        assert advice is not None
        assert "`ph@localhost`" in advice
        assert "koopmans install" in advice

    def test_wannierize_blocks_missing_wannierjl_and_projwfc(
        self,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        installed_wannier_codes: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """The split threshold turns on wannierjl; both missing codes are named, not merged.

        Both ``wannierjl`` and ``projwfc`` are ``NotRequired`` in
        ``WannierizeBlocksCodes``, so the pre-flight passes and this still
        reaches ``check_before_run``. Discriminates the dedup against
        over-merging: two distinct missing codes must earn two advice
        lines, not one.
        """
        from tests.test_wannierize_blocks_dispatcher import _si_split_dict

        inp = KoopmansInput.model_validate(_si_split_dict())
        wg = build_workgraph(inp)

        from aiida_workgraph.errors import MissingRequiredInputsError

        with pytest.raises(MissingRequiredInputsError) as excinfo:
            wg.check_before_run()

        advice = advice_for(excinfo.value)
        assert advice is not None
        assert "`wannierjl@localhost`" in advice
        assert "`projwfc@localhost`" in advice
        assert advice.count("wannierjl@localhost") == 1
