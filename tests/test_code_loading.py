"""The dispatcher's TypedDict-driven code loading.

``load_codes`` reads a workflow's codes TypedDict — the plugin's single
declaration of what each workflow wires — and passes every configured
``<name>@localhost`` member, required and ``NotRequired`` alike. No
requiredness logic: a missing required code passes the build and surfaces
at submit as a structured ``MissingRequiredInputsError``, which
``advice_for`` translates into install advice quoting the member's
declared purpose. ``load_codes_by_need`` keeps the demanding #143-shape
loading for the routes whose graph bodies still subscript ``codes`` at
build time (wannierize, dielectric).
"""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.aiida.workflows import (
    advice_for,
    build_workgraph,
    load_codes,
    load_codes_by_need,
)
from koopmans.input_file import KoopmansInput
from tests.test_dscf_mlwf_dispatcher import _si_dscf_dict


class TestCodesSpecRequiredness:
    """Pin required/optional members at every codes TypedDict's definition site.

    The loaders' and the socket layer's contracts read off
    ``__required_keys__`` / ``__optional_keys__``, and a ``from __future__
    import annotations`` sneaking into a definition module silently flips
    every member to required (python/cpython#97727). This screams instead.
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
            WannierizeBlocksCodes: (wannier, {"wannierjl"}),
            SplitBlockCodes: (wannier | {"wannierjl"}, set()),
            MlwfInitCodes: (wannier | {"wann2kcp", "merge_evc", "kcp"}, set()),
            FoldingCodes: ({"wann2kcp", "merge_evc"}, set()),
            DscfCodes: ({"kcp"}, wannier | {"wann2kcp", "merge_evc"}),
            DfptCodes: (wannier | {"kcw"}, {"ph"}),
            PdosCodes: ({"pw", "dos", "projwfc"}, set()),
        }
        for spec, (required, optional) in expected.items():
            assert set(spec.__required_keys__) == required, spec.__name__
            assert set(spec.__optional_keys__) == optional, spec.__name__


class TestLoadCodes:
    """Pass-everything: the profile's configured members ride along, nothing is demanded."""

    def test_configured_members_ride_along(
        self, aiida_profile_clean: Any, installed_kcp_code: Any, installed_pw_code: Any
    ) -> None:
        """A configured ``NotRequired`` member is passed alongside the required ones.

        What the graph receives follows the profile: requiredness is the
        socket layer's business, so the loader has no reason to hold a
        configured optional code back.
        """
        from aiida_koopmans.workgraphs.kcp import DscfCodes

        codes = load_codes(DscfCodes)
        assert set(codes) == {"kcp", "pw"}

    def test_missing_required_member_is_not_demanded(
        self, aiida_profile_clean: Any, installed_pw_code: Any
    ) -> None:
        """A required member the profile lacks is simply left out.

        Its absence is the socket layer's to report — at submit, as a
        ``MissingRequiredInputsError`` naming ``graph_inputs.codes.kcp``.
        """
        from aiida_koopmans.workgraphs.kcp import DscfCodes

        codes = load_codes(DscfCodes)
        assert set(codes) == {"pw"}

    def test_empty_profile_raises_install_advice_for_required(
        self, aiida_profile_clean: Any
    ) -> None:
        """No configured member at all raises install advice naming the required ones.

        A workaround: aiida-workgraph drops an explicitly-passed empty
        mapping from the build inputs, so an empty ``codes`` would die as
        a bare ``TypeError`` instead of the structured report.
        """
        from aiida_koopmans.workgraphs.kcp import DscfCodes

        with pytest.raises(ValueError, match="`kcp@localhost`") as excinfo:
            load_codes(DscfCodes)
        message = str(excinfo.value)
        assert "koopmans install" in message
        # Optional members are not demanded even here.
        assert "wannier90@localhost" not in message


class TestLoadCodesByNeed:
    """The demanding loader: requiredness, ``require``, and nothing else."""

    def test_missing_required_code_earns_install_advice(self, aiida_profile_clean: Any) -> None:
        """An unconfigured required member raises naming it and `koopmans install`."""
        from aiida_koopmans.workgraphs.ph import DielectricCodes

        with pytest.raises(ValueError, match="`ph@localhost`") as excinfo:
            load_codes_by_need(DielectricCodes)
        assert "koopmans install" in str(excinfo.value)

    def test_absent_notrequired_member_is_left_out(
        self, aiida_profile_clean: Any, installed_kcp_code: Any, installed_pw_code: Any
    ) -> None:
        """A ``NotRequired`` member the route did not turn on is not demanded or passed."""
        from aiida_koopmans.workgraphs.kcp import DscfCodes

        codes = load_codes_by_need(DscfCodes)
        assert set(codes) == {"kcp"}

    def test_require_quotes_the_declared_purpose(
        self, aiida_profile_clean: Any, installed_kcp_code: Any
    ) -> None:
        """A turned-on ``NotRequired`` member missing raises with its SocketMeta help."""
        from aiida_koopmans.workgraphs.kcp import DscfCodes

        with pytest.raises(ValueError, match="`wannier90@localhost`") as excinfo:
            load_codes_by_need(DscfCodes, require=DscfCodes.__optional_keys__)
        message = str(excinfo.value)
        assert "Needed to initialize the variational orbitals as Wannier functions." in message

    def test_require_rejects_undeclared_names(self, aiida_profile_clean: Any) -> None:
        """A ``require`` name outside the TypedDict is a programming error, not advice."""
        from aiida_koopmans.workgraphs.pw import PwBandsCodes

        with pytest.raises(ValueError, match="not members of PwBandsCodes"):
            load_codes_by_need(PwBandsCodes, require=("bogus",))


class TestMissingCodesSurfaceAtSubmit:
    """A missing required code passes the build and is reported at run start.

    The pass-everything routes build their graphs with whatever the
    profile holds; ``check_before_run`` — the first thing ``run()`` /
    ``submit()`` do — raises the structured ``MissingRequiredInputsError``
    that ``advice_for`` translates into the same install advice the old
    pre-check raised at build.
    """

    def test_molecular_dscf_reports_kcp_at_run(
        self,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        fake_sg15_pseudo_family: Any,
        tutorials_dir: Any,
    ) -> None:
        """A Kohn-Sham-initialised DSCF builds without kcp.x and names it at run start.

        pw.x is configured (and rides along); no advice line may demand it.
        """
        from aiida_workgraph.errors import MissingRequiredInputsError

        from koopmans.input_file import read_input_file

        inp = read_input_file(tutorials_dir / "orbital_energies/ozone/ozone.yaml")
        wg = build_workgraph(inp)
        with pytest.raises(MissingRequiredInputsError) as excinfo:
            wg.run()
        advice = advice_for(excinfo.value)
        assert advice is not None
        assert "`kcp@localhost`" in advice
        assert "koopmans install" in advice
        assert "pw@localhost" not in advice

    def test_empty_profile_dft_bands_is_advice_not_typeerror(
        self, aiida_profile_clean: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """With nothing configured, the bands route raises install advice at build.

        The empty-mapping workaround in ``load_codes``: an empty ``codes``
        would be dropped from the build inputs and die as a bare
        ``TypeError`` instead of any structured report.
        """
        from tests.fixtures import silicon_pw_input

        inp = KoopmansInput.model_validate(silicon_pw_input())
        with pytest.raises(ValueError, match="`pw@localhost`") as excinfo:
            build_workgraph(inp)
        assert "koopmans install" in str(excinfo.value)

    def test_wannier_route_missing_fold_codes_is_a_scope_error(
        self,
        aiida_profile_clean: Any,
        installed_kcp_code: Any,
        installed_pw_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """The mlwfs route's conditional codes stay a build-time guard.

        ``DscfCodes`` declares the Wannier-route members ``NotRequired``
        (their need follows ``init_orbitals``), so the socket layer cannot
        demand them; the workflow's own ``_validate_scope`` raises at
        build, naming them in prose.
        """
        inp = KoopmansInput.model_validate(_si_dscf_dict())
        with pytest.raises(ValueError, match="wann2kcp/merge_evc codes"):
            build_workgraph(inp)
