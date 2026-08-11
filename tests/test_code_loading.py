"""The dispatcher's TypedDict-driven code loading.

``load_codes`` reads a workflow's codes TypedDict — the plugin's single
declaration of what each workflow wires — and loads each needed member as
``<name>@localhost``: required members always, ``NotRequired`` members only
when the route's ``require`` turns them on. A missing needed code raises
install advice quoting the member's declared purpose, before any graph
body can trip over it.
"""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.aiida.workflows import build_workgraph, load_codes
from koopmans.input_file import KoopmansInput
from tests.test_dscf_mlwf_dispatcher import _si_dscf_dict


class TestCodesSpecRequiredness:
    """Pin required/optional members at every codes TypedDict's definition site.

    The loader's whole contract reads off ``__required_keys__`` /
    ``__optional_keys__``, and a ``from __future__ import annotations``
    sneaking into a definition module silently flips every member to
    required (python/cpython#97727). This screams instead.
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
            DfptCodes: (wannier | {"kcw"}, {"ph"}),
            PdosCodes: ({"pw", "dos", "projwfc"}, set()),
        }
        for spec, (required, optional) in expected.items():
            assert set(spec.__required_keys__) == required, spec.__name__
            assert set(spec.__optional_keys__) == optional, spec.__name__


class TestLoadCodes:
    """Loading follows the TypedDict: requiredness, ``require``, and nothing else."""

    def test_missing_required_code_earns_install_advice(self, aiida_profile_clean: Any) -> None:
        """An unconfigured required member raises naming it and `koopmans install`."""
        from aiida_koopmans.workgraphs.pw import PwBandsCodes

        with pytest.raises(ValueError, match="`pw@localhost`") as excinfo:
            load_codes(PwBandsCodes)
        assert "koopmans install" in str(excinfo.value)

    def test_absent_notrequired_member_is_left_out(
        self, aiida_profile_clean: Any, installed_kcp_code: Any
    ) -> None:
        """A ``NotRequired`` member the route did not turn on is not demanded."""
        from aiida_koopmans.workgraphs.kcp import DscfCodes

        codes = load_codes(DscfCodes)
        assert set(codes) == {"kcp"}

    def test_configured_notrequired_member_is_still_left_out(
        self, aiida_profile_clean: Any, installed_kcp_code: Any, installed_pw_code: Any
    ) -> None:
        """What the graph receives is fixed by the input, not by the profile.

        The discriminating half of the previous test: ``pw@localhost``
        exists here, and the result must not change — otherwise the graph
        inputs (and their provenance) would depend on which codes a
        profile happens to hold.
        """
        from aiida_koopmans.workgraphs.kcp import DscfCodes

        codes = load_codes(DscfCodes)
        assert set(codes) == {"kcp"}

    def test_require_quotes_the_declared_purpose(
        self, aiida_profile_clean: Any, installed_kcp_code: Any
    ) -> None:
        """A turned-on ``NotRequired`` member missing raises with its SocketMeta help."""
        from aiida_koopmans.workgraphs.kcp import DscfCodes

        with pytest.raises(ValueError, match="`wannier90@localhost`") as excinfo:
            load_codes(DscfCodes, require=DscfCodes.__optional_keys__)
        message = str(excinfo.value)
        assert "Needed to initialize the variational orbitals as Wannier functions." in message

    def test_require_rejects_undeclared_names(self, aiida_profile_clean: Any) -> None:
        """A ``require`` name outside the TypedDict is a programming error, not advice."""
        from aiida_koopmans.workgraphs.pw import PwBandsCodes

        with pytest.raises(ValueError, match="not members of PwBandsCodes"):
            load_codes(PwBandsCodes, require=("bogus",))


class TestDispatcherPreCheck:
    """The routes demand exactly what their input turns on.

    Every route builds its graph eagerly, and an eager body subscripting a
    missing code member dies with a bare ``KeyError`` before any socket
    validation — so the pre-check is the only guard, and each test here
    pins that the advice error fires instead of a ``KeyError``.
    """

    def test_dft_bands_missing_pw_is_advice_not_keyerror(
        self, aiida_profile_clean: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """The bands route names its missing pw.x before the eager build can trip."""
        from tests.fixtures import silicon_pw_input

        inp = KoopmansInput.model_validate(silicon_pw_input())
        with pytest.raises(ValueError, match="`pw@localhost`") as excinfo:
            build_workgraph(inp)
        assert "koopmans install" in str(excinfo.value)

    def test_molecular_dscf_demands_kcp_alone(
        self, aiida_profile_clean: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """A Kohn-Sham-initialised DSCF names only its kcp.x code.

        No pw.x runs on the molecular route, so the old dispatcher's
        pw-for-every-task load must not resurface as a demand here.
        """
        inp = KoopmansInput.model_validate(_si_dscf_dict(init_orbitals="kohn-sham"))
        with pytest.raises(ValueError, match="`kcp@localhost`") as excinfo:
            build_workgraph(inp)
        assert "pw@localhost" not in str(excinfo.value)

    def test_wannier_route_demands_the_fold_codes(
        self,
        aiida_profile_clean: Any,
        installed_kcp_code: Any,
        installed_pw_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """The mlwfs route turns every DscfCodes member on, with its purpose quoted."""
        inp = KoopmansInput.model_validate(_si_dscf_dict())
        with pytest.raises(ValueError, match="`wann2kcp@localhost`") as excinfo:
            build_workgraph(inp)
        message = str(excinfo.value)
        assert "`merge_evc@localhost`" in message
        assert "Needed to initialize the variational orbitals as Wannier functions." in message
