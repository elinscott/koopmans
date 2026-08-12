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

That translation only fires where the plugin graph actually structures the
missing code as an unlinked socket. Some entry graphs still bind their own
codes eagerly (a plain ``codes["name"]`` subscript inside an ``@task.graph``
body executes immediately, during ``build()``, so an absent key is a bare
``KeyError`` there — not a socket the framework ever gets to validate); one
route hands its whole codes namespace to an *upstream* builder that raises
its own eager ``ValueError`` instead. ``TestKnownEntryPointGap`` pins these
as known gaps — some fixable in aiida-koopmans alone, one only upstream, one
a deliberate permanent choice — not a k2 defect.
"""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.aiida.workflows import advice_for, build_workgraph, load_codes
from koopmans.input_file import KoopmansInput
from tests.test_dscf_mlwf_dispatcher import _si_dscf_dict


class TestCodesSpecRequiredness:
    """Pin required/optional members at every codes TypedDict's definition site.

    ``load_codes`` no longer reads these sets — every member is loaded when
    configured, regardless — but the plugin graphs' own structural checks
    do, so a ``from __future__ import annotations`` sneaking into a
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
        """An unconfigured member — required or not — is left out, never raised.

        The discriminating case against the pre-migration contract:
        ``load_codes`` used to raise for a missing *required* member
        before a graph ever saw it. That check is gone; an empty profile
        now gets an empty mapping back, and whichever member the graph
        actually needed finds out on its own.
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


class TestStructuralAdvice:
    """Codes the plugin graphs wire structurally: builds fine, fails at check/submit.

    Each case here reaches its ``MissingRequiredInputsError`` because the
    consuming ``@task.graph`` threads the missing member through
    ``node_graph.ref`` rather than subscripting it directly — the socket
    stays unlinked instead of a Python dict access dying immediately.
    """

    def test_dfpt_missing_pw_fans_out_and_collapses_to_one_line(
        self,
        aiida_profile_clean: Any,
        installed_kcw_code: Any,
        installed_wannier_codes: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """Pw feeds several nested sockets; the advice still names it once.

        Live reproduction of the fan-out ``advice_for`` has to collapse:
        the missing top-level ``codes.pw``, the shared scf/nscf step's own
        ``pw_code`` kwarg, and the nested wannierize sub-graph's
        ``codes.pw`` all trace back to the one unconfigured code.
        """
        from tests.test_dfpt_dispatcher import _si_dfpt_dict

        inp = KoopmansInput.model_validate(_si_dfpt_dict())
        wg = build_workgraph(inp)

        from aiida_workgraph.errors import MissingRequiredInputsError

        with pytest.raises(MissingRequiredInputsError) as excinfo:
            wg.check_before_run()
        paths = {entry.socket_path for entry in excinfo.value.missing}
        assert len(paths) > 1, "the fan-out this test exists to collapse did not happen"

        advice = advice_for(excinfo.value)
        assert advice is not None
        assert advice.count("pw@localhost") == 1
        assert "koopmans install" in advice

    # eps_inf='auto' missing ph is the same mechanism, one hop simpler (no
    # fan-out): see test_dft_eps_dispatcher.py's
    # test_auto_without_ph_code_builds_then_fails_at_check.

    def test_wannierize_blocks_missing_wannierjl_and_projwfc(
        self,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        installed_wannier_codes: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """The split threshold turns on wannierjl; both missing codes are named, not merged.

        Discriminates the dedup against over-merging: wannierjl and
        projwfc are two distinct missing codes and must earn two advice
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

    def test_wannierize_blocks_missing_pw_without_a_path(
        self,
        aiida_profile_clean: Any,
        installed_wannier_codes: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """Without a quality-check bands run, the whole block route defers pw safely.

        Discriminates the block route's *own* per-block Wannierization
        wiring (already ``ref()``-threaded) from its optional
        quality-check-bands add-on (:class:`TestKnownEntryPointGap`'s
        ``test_wannierize_blocks_with_path_missing_pw_keyerrors_at_build`,
        the same input with a ``kpoints.path`` added): dropping the path
        removes the only bare ``codes["pw"]`` subscript on this route, and
        the build reaches ``check_before_run`` cleanly, fanning out across
        the same three sockets as ``test_dfpt_missing_pw_fans_out_...``.
        """
        from tests.test_wannierize_blocks_dispatcher import _si_split_dict

        d = _si_split_dict(block_wannierization_threshold=None)
        d["kpoints"].pop("path")
        inp = KoopmansInput.model_validate(d)
        wg = build_workgraph(inp)

        from aiida_workgraph.errors import MissingRequiredInputsError

        with pytest.raises(MissingRequiredInputsError) as excinfo:
            wg.check_before_run()

        advice = advice_for(excinfo.value)
        assert advice is not None
        assert advice.count("pw@localhost") == 1
        assert "koopmans install" in advice


class TestKnownEntryPointGap:
    """A route whose entry graph binds its own code eagerly still crashes at build.

    Not a k2 defect: the dispatcher builds a plain mapping and hands it to
    the graph exactly as designed (:func:`load_codes`); what a route's
    *own* top-level ``@task.graph`` body — or an upstream builder it calls
    into — does with that mapping is out of the dispatcher's hands. Three
    different reasons, each pinned here so a future fix on the other side
    shows up as an unexpected pass rather than going unnoticed:

    * **ak2's own eager builder call, fixable there alone.**
      ``aiida_koopmans.workgraphs.pw.RunPwBands`` and
      ``aiida_koopmans.workgraphs.ph.DielectricTask`` (the latter also
      covered by ``tests/test_dft_eps_dispatcher.py``'s
      ``test_missing_ph_code_keyerrors_at_build``) feed the code straight
      into an eager ``get_builder_from_protocol`` call in their *own*
      body, which needs a concrete ``orm.Code`` and cannot take a lazy
      ``ref()``. ``block_wannierize.py``'s optional quality-check-bands
      helper (entered only when a ``kpoints.path`` is given — see
      ``test_wannierize_blocks_with_path_missing_pw_keyerrors_at_build``
      below, and contrast ``TestStructuralAdvice``'s
      ``test_wannierize_blocks_missing_pw_without_a_path``, where the
      same route without that add-on defers safely) has the identical
      shape. All of these already have a working precedent in the same
      codebase — ``pw.py``'s own ``RunScfNscf`` / ``assemble_pw_base_step``
      pair, and aiida-koopmans#90's own ``RunProjwfc`` — wrapping the
      ``get_builder_from_protocol`` call in its own small ``@task.graph``
      so the *caller* can ``ref()`` the code into it. This is the gap
      aiida-koopmans#88 flagged against upstream node-graph#169 and said
      would "convert in a follow-up"; #90 (the ``ref()`` migration this PR
      pairs with) has not reached these call sites yet.
    * **ak2's own choice, deliberate and permanent.**
      ``aiida_koopmans.workgraphs.kcp.KoopmansDSCFWorkflow``'s own
      ``kcp_code = codes["kcp"]`` bind is different in kind, not just in
      permanence: unlike the sites above, its ~25 downstream consumers
      already accept a plain ``kcp_code: orm.AbstractCode`` parameter —
      the same shape ``RunScfNscf.pw_code`` takes a working ``ref()``
      value into elsewhere in this codebase — so swapping this one bind to
      ``ref(codes, "kcp")`` looks mechanically available with no consumer
      changes. aiida-koopmans#90's own description nonetheless treats it
      as settled: ``kcp`` is never route-conditional, so the maintainer
      judged the extra indirection not worth carrying across 11 sibling
      functions for a value that never varies. A user call, not a
      technical wall — flagged as such rather than asserted as fixed.
    * **Blocked upstream, not ak2's or k2's code at all.**
      ``aiida_koopmans.workgraphs.wannier90.Wannierize`` (the whole-manifold
      ``auto_projections`` route) hands its entire ``codes`` namespace to
      ``Wannier90WorkChain.get_builder_from_protocol(codes=codes, ...)`` —
      upstream ``aiida-wannier90-workflows``. That library's own
      ``utils/workflows/builder/submit.py`` does an eager membership check
      and raises a plain ``ValueError`` (``codes does not contain the
      required key: pw``) — not even a bare ``KeyError``, and not a line
      either k2 or aiida-koopmans owns. Closing this needs either an
      upstream change accepting a lazy code reference, or wrapping ak2's
      own call to that upstream builder in an indirection ``@task.graph``
      the same way the ak2-side sites above would.
    """

    def test_dft_bands_missing_pw_keyerrors_at_build(
        self, aiida_profile_clean: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """RunPwBands binds ``codes["pw"]`` directly; a missing pw dies in build()."""
        from tests.fixtures import silicon_pw_input

        inp = KoopmansInput.model_validate(silicon_pw_input())
        with pytest.raises(KeyError, match="pw"):
            build_workgraph(inp)

    def test_molecular_dscf_missing_kcp_keyerrors_at_build(
        self, aiida_profile_clean: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """KoopmansDSCFWorkflow binds ``codes["kcp"]`` directly — a deliberate, permanent choice."""
        inp = KoopmansInput.model_validate(_si_dscf_dict(init_orbitals="kohn-sham"))
        with pytest.raises(KeyError, match="kcp"):
            build_workgraph(inp)

    def test_wannierize_blocks_with_path_missing_pw_keyerrors_at_build(
        self,
        aiida_profile_clean: Any,
        installed_wannier_codes: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """A kpoints.path reaches the quality-check-bands helper, which bare-binds pw."""
        from tests.test_wannierize_blocks_dispatcher import _si_split_dict

        d = _si_split_dict(block_wannierization_threshold=None)
        inp = KoopmansInput.model_validate(d)
        with pytest.raises(KeyError, match="pw"):
            build_workgraph(inp)

    def test_wannierize_whole_auto_missing_pw_raises_upstream_valueerror(
        self, aiida_profile_clean: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Wannierize hands upstream its whole codes namespace; upstream's own check fires first.

        Not a ``KeyError`` like the other sites here: upstream
        aiida-wannier90-workflows raises its own ``ValueError`` before
        ak2 or k2 code runs again.
        """
        from tests.test_wannierize_blocks_dispatcher import _si_auto_dict

        d = _si_auto_dict()
        del d["workflow"]["block_wannierization_threshold"]
        inp = KoopmansInput.model_validate(d)
        with pytest.raises(ValueError, match="does not contain the required key"):
            build_workgraph(inp)
