"""The dispatch boundary's translation of typed plugin errors.

``build_workgraph`` attaches input-file advice, as a PEP 678 note, to the
typed errors aiida-koopmans raises; ``advice_for`` dispatches on the
exception's type. Every advice-table entry gets one case in the dispatch
table below, with the plugin's own raise site firing; the plugin's untyped
errors — including the derivation-invariant rejections that used to share
a class with the user faults — are pinned to pass through untranslated.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from koopmans.aiida.workflows import advice_for, build_workgraph
from koopmans.input_file import KoopmansInput
from tests.test_conversion import _pw_input
from tests.test_dfpt_dispatcher import _si_dfpt_dict
from tests.test_dscf_mlwf_dispatcher import _si_dscf_dict
from tests.test_trajectory_dispatcher import _trajectory_input_dict
from tests.test_wannierize_blocks_dispatcher import _si_split_dict

if TYPE_CHECKING:
    from collections.abc import Callable


class TestAdviceFor:
    """``advice_for`` dispatches on the exception's type alone."""

    def test_user_fault_earns_advice(self, aiida_profile: Any) -> None:
        """A real unknown-site raise earns advice naming the site."""
        from koopmans.aiida.conversion import atoms_input_to_structure

        with pytest.raises(ValueError, match="does not match any atom") as excinfo:
            _derive_si_blocks(
                atoms_input_to_structure(KoopmansInput.model_validate(_si_dscf_dict()).atoms),
                site="Ge",
            )
        advice = advice_for(excinfo.value)
        assert advice is not None
        assert "'Ge'" in advice
        assert "atomic_positions" in advice

    def test_derivation_invariant_gets_no_advice(self, aiida_profile: Any) -> None:
        """A fault only the block derivation can produce passes untranslated.

        The discriminating half of the fault split: the reversed layout is
        rejected by the same validator family, but as a plain ValueError,
        so no projection advice can attach to an internal bug.
        """
        from aiida_koopmans.projections import validate_projection_block_sequence

        from koopmans.aiida.conversion import atoms_input_to_structure

        structure = atoms_input_to_structure(KoopmansInput.model_validate(_si_dscf_dict()).atoms)
        with pytest.raises(ValueError, match="ascending band order") as excinfo:
            validate_projection_block_sequence(list(reversed(_derive_si_blocks(structure))))
        assert advice_for(excinfo.value) is None

    def test_untyped_plugin_error_gets_no_advice(self) -> None:
        """A plain ValueError from a plugin module passes through untranslated."""
        from aiida_koopmans.projections import projection_win_string

        class _SitelessProjection:
            site = None
            fractional_site = None
            cartesian_site = None
            ang_mtm = "sp3"

        with pytest.raises(ValueError, match="defines no site") as excinfo:
            projection_win_string(_SitelessProjection())
        assert advice_for(excinfo.value) is None

    def test_local_error_gets_no_advice(self) -> None:
        """An error raised outside the plugin gets no advice either."""
        with pytest.raises(ValueError, match="not the plugin") as excinfo:
            raise ValueError("raised by the dispatcher, not the plugin")
        assert advice_for(excinfo.value) is None


def _derive_si_blocks(structure: Any, *, site: str = "Si") -> list[Any]:
    """Derive the two-block silicon layout the DSCF input dict describes."""
    from aiida_koopmans.spin import SpinChannel

    from koopmans.aiida.workflows.blocks import create_explicit_blocks

    d = _si_dscf_dict()
    d["calculator_parameters"]["wannier90"]["projections"] = [
        [{"site": site, "ang_mtm": "sp"}],
        [{"site": site, "ang_mtm": "sp"}],
    ]
    projections = KoopmansInput.model_validate(d).calculator_parameters.wannier90.projections
    return create_explicit_blocks(structure, projections, 8, 4, SpinChannel.NONE)


def _unknown_site_input() -> dict[str, Any]:
    """Return the DSCF input with a projection site no atom matches."""
    d = _si_dscf_dict()
    d["calculator_parameters"]["wannier90"]["projections"] = [[{"site": "Ge", "ang_mtm": "sp"}]]
    return d


def _straddling_dfpt_input() -> dict[str, Any]:
    """Return the DFPT input with one sp3 block spanning both manifolds."""
    d = _si_dfpt_dict()
    d["calculator_parameters"]["wannier90"]["projections"] = [[{"site": "Si", "ang_mtm": "sp3"}]]
    return d


def _undercovered_dfpt_input() -> dict[str, Any]:
    """Return the DFPT input with fewer occupied projections than bands."""
    d = _si_dfpt_dict()
    d["calculator_parameters"]["wannier90"]["projections"] = [[{"site": "Si", "ang_mtm": "s"}]]
    return d


def _cramped_dfpt_input() -> dict[str, Any]:
    """Return the DFPT input whose ``nbnd`` undercuts its empty projections."""
    d = _si_dfpt_dict()
    d["calculator_parameters"]["wannier90"]["projections"] = [
        [{"site": "Si", "ang_mtm": "sp"}],
        [{"site": "Si", "ang_mtm": "s"}],
    ]
    d["calculator_parameters"]["nbnd"] = 5
    return d


def _patch_lower_disentanglement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the route's plugin entry run the sequence validator on a bad layout.

    The derivation gives extra bands only to a channel's top block, so the
    layout is built and then a lower block's ``num_bands`` inflated — the
    raise (and the class) are the plugin's own.
    """
    import aiida_koopmans.workgraphs.kcp as kcp_module
    from aiida_koopmans.projections import validate_projection_block_sequence

    def build_with_lower_disentanglement(**kwargs: Any) -> Any:
        """Run the real sequence validator on a lower-disentangling layout."""
        blocks = _derive_si_blocks(kwargs["structure"])
        blocks[0]["num_bands"] = 8
        validate_projection_block_sequence(blocks)

    monkeypatch.setattr(kcp_module.KoopmansDSCFWorkflow, "build", build_with_lower_disentanglement)


def _patch_frozen_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the route's plugin entry run the window validator on bands it rejects.

    The window check reads nscf eigenvalues, which exist only at runtime —
    in production it raises daemon-side, past the build boundary, so its
    entry translates nothing today; this pins the entry it provisions.
    """
    import aiida_koopmans.workgraphs.block_wannierize as bw_module
    from aiida.orm import BandsData
    from aiida_koopmans.workgraphs.block_wannierize import validate_frozen_window

    def build_with_bad_window(**kwargs: Any) -> Any:
        """Run the real window validator on bands it must reject."""
        bands = BandsData()
        bands.set_kpoints(np.zeros((1, 3)))  # type: ignore[no-untyped-call]
        bands.set_bands(np.zeros((1, 4)))  # type: ignore[no-untyped-call]
        validate_frozen_window("occ_1", {"dis_froz_max": 10.0, "num_wann": 2}, "none", bands)

    monkeypatch.setattr(bw_module.WannierizeBlocks, "build", build_with_bad_window)


def _patch_bogus_parallelization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Feed the plugin's parallelization validator a mapping the schema rejects."""
    from koopmans.input_file.parallelization import ParallelizationInput

    monkeypatch.setattr(ParallelizationInput, "as_mapping", lambda self: {"bogus": {"ntasks": 2}})


def _patch_mismatched_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the route's plugin entry run the model-stamp check on a mismatch.

    The stamp check runs inside the prediction task, which needs the trial
    KI's descriptors — in production it raises daemon-side, past the build
    boundary, so its entry translates nothing today; this pins the entry it
    provisions. ``_callable`` is the raw function under the @task handle;
    the descriptor check raises before descriptors or orbitals are read.
    """
    import aiida_koopmans.workgraphs.ml as ml_module
    from aiida_koopmans.workgraphs.kcp import predict_alpha_screening

    def build_with_bad_model(**kwargs: Any) -> Any:
        """Run the real stamp check on a model it must reject."""
        predict_alpha_screening._callable(
            model={"descriptor": "power_spectrum"},
            descriptors=[],
            orbitals=[],
            correction="ki",
            init_orbitals="mlwfs",
        )

    monkeypatch.setattr(ml_module.TrajectoryWorkflow, "build", build_with_bad_model)


def _trajectory_input(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Return the water trajectory input, writing its snapshots file."""
    xyz = request.getfixturevalue("write_multiframe_xyz")(request.getfixturevalue("tmp_path"), 1)
    return _trajectory_input_dict(str(xyz))


@dataclass(frozen=True)
class _DispatchCase:
    """One advice-table entry exercised through ``build_workgraph``."""

    id: str
    error: str  # dotted path of the expected exception class
    fixtures: tuple[str, ...]
    input_builder: Callable[[pytest.FixtureRequest], dict[str, Any]]
    match: str
    note: tuple[str, ...]  # fragments the attached advice must carry
    patch: Callable[[pytest.MonkeyPatch], None] | None = None


_DSCF_FIXTURES = ("installed_pw_code", "installed_kcp_code", "fake_sg15_pseudo_family")
_DFPT_FIXTURES = ("installed_pw_code", "installed_kcw_code", "fake_sg15_pseudo_family")

_DISPATCH_CASES = (
    _DispatchCase(
        id="projection_site",
        error="aiida_koopmans.projections.ProjectionSiteError",
        fixtures=_DSCF_FIXTURES,
        input_builder=lambda request: _unknown_site_input(),
        match="does not match any atom",
        note=("'Ge'", "atomic_positions"),
    ),
    _DispatchCase(
        id="block_boundary",
        error="aiida_koopmans.projections.BlockBoundaryError",
        fixtures=_DFPT_FIXTURES,
        input_builder=lambda request: _straddling_dfpt_input(),
        match="straddles",
        note=("split at the occupied/empty boundary",),
    ),
    _DispatchCase(
        id="occupied_coverage",
        error="aiida_koopmans.projections.OccupiedCoverageError",
        fixtures=_DFPT_FIXTURES,
        input_builder=lambda request: _undercovered_dfpt_input(),
        match="occupied projection blocks span",
        note=("one Wannier function per occupied band",),
    ),
    _DispatchCase(
        id="empty_coverage",
        error="aiida_koopmans.projections.EmptyCoverageError",
        fixtures=_DFPT_FIXTURES,
        input_builder=lambda request: _cramped_dfpt_input(),
        match="leaves only",
        note=("raise `calculator_parameters.nbnd`",),
    ),
    _DispatchCase(
        id="block_disentanglement",
        error="aiida_koopmans.projections.BlockDisentanglementError",
        fixtures=_DSCF_FIXTURES,
        input_builder=lambda request: _si_dscf_dict(init_orbitals="kohn-sham"),
        match="uppermost block",
        note=("only the last of the projection blocks",),
        patch=_patch_lower_disentanglement,
    ),
    _DispatchCase(
        id="frozen_window",
        error="aiida_koopmans.workgraphs.block_wannierize.FrozenWindowError",
        fixtures=("installed_pw_code", "installed_wannier_codes", "fake_sg15_cutoffs_family"),
        input_builder=lambda request: _si_split_dict(block_wannierization_threshold=None),
        match="frozen",
        note=("`dis_froz_max`", "block 'occ_1'"),
        patch=_patch_frozen_window,
    ),
    _DispatchCase(
        id="parallelization",
        error="aiida_koopmans.parallelization.ParallelizationError",
        fixtures=("installed_pw_code", "fake_sg15_cutoffs_family"),
        input_builder=lambda request: _pw_input(),
        match="unknown parallelization code",
        note=("`parallelization` block",),
        patch=_patch_bogus_parallelization,
    ),
    _DispatchCase(
        id="model_mismatch",
        error="aiida_koopmans.ml.ModelMismatchError",
        fixtures=("installed_pw_code", "installed_kcp_code", "fake_sg15_pseudo_family"),
        input_builder=_trajectory_input,
        match="descriptor",
        note=("ml.model_file", "'descriptor' stamp"),
        patch=_patch_mismatched_model,
    ),
)


class TestDispatchTranslation:
    """Each advice-table entry crosses ``build_workgraph`` with its advice."""

    @pytest.mark.parametrize("case", _DISPATCH_CASES, ids=lambda case: case.id)
    def test_entry_attaches_its_advice(
        self,
        case: _DispatchCase,
        aiida_profile: Any,
        request: pytest.FixtureRequest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The entry's class carries its specific advice as a PEP 678 note."""
        for name in case.fixtures:
            request.getfixturevalue(name)
        if case.patch is not None:
            case.patch(monkeypatch)
        module_name, _, class_name = case.error.rpartition(".")
        exc_type = getattr(import_module(module_name), class_name)

        inp = KoopmansInput.model_validate(case.input_builder(request))
        with pytest.raises(exc_type, match=case.match) as excinfo:
            build_workgraph(inp)
        notes = excinfo.value.__notes__
        for fragment in case.note:
            assert any(fragment in note for note in notes), (fragment, notes)

    def test_derivation_invariant_crosses_without_advice(
        self,
        aiida_profile: Any,
        request: pytest.FixtureRequest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An internal-invariant rejection crosses the dispatcher bare.

        The reversed-blocks derivation is a layout only our own builders
        could produce; it must arrive as a plain ValueError with no advice
        note — the projection advice belongs to user faults alone.
        """
        from aiida_koopmans.projections import ProjectionBlockError

        import koopmans.aiida.workflows.dscf as dscf_module
        from koopmans.aiida.workflows.blocks import create_explicit_blocks as derive

        for name in (*_DSCF_FIXTURES, "installed_wannier_codes", "installed_fold_codes"):
            request.getfixturevalue(name)

        def reversed_blocks(*args: Any, **kwargs: Any) -> Any:
            """Derive the real blocks, reversed."""
            return list(reversed(derive(*args, **kwargs)))

        monkeypatch.setattr(dscf_module, "create_explicit_blocks", reversed_blocks)
        with pytest.raises(ValueError, match="ascending band order") as excinfo:
            build_workgraph(KoopmansInput.model_validate(_si_dscf_dict()))
        assert not isinstance(excinfo.value, ProjectionBlockError)
        assert not getattr(excinfo.value, "__notes__", [])
