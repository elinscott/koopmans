"""The dispatch boundary's translation of typed plugin errors.

``build_workgraph`` attaches input-file advice, as a PEP 678 note, to the
typed errors aiida-koopmans raises; ``advice_for`` dispatches on the
exception's type. Every advice-table entry gets one plain test below, with
the plugin's own raise site firing; the plugin's derivation-invariant
rejections, untyped by design, are pinned to pass through untranslated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from koopmans.aiida.workflows import advice_for, build_workgraph
from koopmans.input_file import KoopmansInput
from tests.fixtures import silicon_pw_input as _pw_input
from tests.test_dfpt_dispatcher import _si_dfpt_dict
from tests.test_dscf_mlwf_dispatcher import _si_dscf_dict
from tests.test_trajectory_dispatcher import _trajectory_input_dict
from tests.test_wannierize_blocks_dispatcher import _si_split_dict


def _build_expecting(
    input_dict: dict[str, Any], error: type[Exception], match: str
) -> pytest.ExceptionInfo[Exception]:
    """Build ``input_dict`` through the dispatcher, expecting ``error``."""
    with pytest.raises(error, match=match) as excinfo:
        build_workgraph(KoopmansInput.model_validate(input_dict))
    return excinfo


def _derive_si_blocks(structure: Any, *, site: str = "Si") -> list[Any]:
    """Derive the two-block silicon layout the DSCF input dict describes."""
    from aiida_koopmans.spin import SpinChannel
    from wannier90_input.models.parameters import Projection

    from koopmans.aiida.workflows.blocks import create_explicit_blocks

    projections = [[Projection(site=site, ang_mtm="l=-1")], [Projection(site=site, ang_mtm="l=-1")]]
    return create_explicit_blocks(structure, projections, 8, 4, SpinChannel.NONE)


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

    def test_local_error_gets_no_advice(self) -> None:
        """An error raised outside the plugin gets no advice either."""
        with pytest.raises(ValueError, match="not the plugin") as excinfo:
            raise ValueError("raised by the dispatcher, not the plugin")
        assert advice_for(excinfo.value) is None

    def test_missing_code_socket_earns_install_advice(self) -> None:
        """A missing ``workgraph.code`` socket is translated with its help text."""
        from aiida_workgraph.errors import MissingInput, MissingRequiredInputsError

        exc = MissingRequiredInputsError(
            [
                MissingInput(
                    "wannierize_and_split_block_1.codes.wannierjl",
                    "workgraph.code",
                    "Needed for threshold-based splitting of bands into blocks.",
                )
            ]
        )
        advice = advice_for(exc)
        assert advice is not None
        assert "`wannierjl@localhost`" in advice
        assert "Needed for threshold-based splitting of bands into blocks." in advice
        assert "koopmans install" in advice

    def test_bare_code_entry_stays_generic(self) -> None:
        """A help-less code entry is named with the install pointer and no purpose.

        Every codes-TypedDict member is annotated, so a real entry carries its
        purpose in ``help``; the advice must still not die on a bare one.
        """
        from aiida_workgraph.errors import MissingInput, MissingRequiredInputsError

        exc = MissingRequiredInputsError(
            [MissingInput("some_future_step.codes.epw", "workgraph.code", None)]
        )
        advice = advice_for(exc)
        assert advice is not None
        assert "`epw@localhost`" in advice
        assert "(" not in advice.splitlines()[1]
        assert "koopmans install" in advice

    def test_missing_non_code_sockets_earn_no_advice(self) -> None:
        """An error naming only non-code sockets is not an installation problem."""
        from aiida_workgraph.errors import MissingInput, MissingRequiredInputsError

        exc = MissingRequiredInputsError([MissingInput("add1.x", "workgraph.int", None)])
        assert advice_for(exc) is None

    def test_fanned_out_code_collapses_to_one_line(self) -> None:
        """A code missing at several sockets is named once, not once per socket.

        A route's own top-level ``codes.pw`` reaches several nested
        tasks; each socket the missing code leaves unlinked is reported
        separately by the framework — one of them under a consumer's own
        ``pw_code`` kwarg name rather than ``pw``. Modelled on a live
        DFPT build missing pw (``graph_inputs.codes.pw`` +
        ``scf_nscf.pw_code`` + ``wannierize.codes.pw``): the advice must
        normalize the ``_code``-suffixed name back to the member name and
        collapse the three entries into one line, preferring the
        caller's own top-level help text over a downstream task's.
        """
        from aiida_workgraph.errors import MissingInput, MissingRequiredInputsError

        exc = MissingRequiredInputsError(
            [
                MissingInput(
                    "graph_inputs.codes.pw",
                    "workgraph.code",
                    "Needed to compute DFT ground state properties.",
                ),
                MissingInput("scf_nscf.pw_code", "workgraph.code", None),
                MissingInput(
                    "wannierize.codes.pw",
                    "workgraph.code",
                    "Needed to compute DFT ground state properties.",
                ),
            ]
        )
        advice = advice_for(exc)
        assert advice is not None
        assert advice.count("pw@localhost") == 1
        assert "Needed to compute DFT ground state properties." in advice

    def test_a_plainly_named_code_socket_earns_no_advice(self) -> None:
        """A socket named just ``code`` names nothing to install, so it is skipped.

        A wrapped WorkChain's own input is often called ``code`` rather than
        after the member it takes, and the last path segment is all the
        advice has to go on. Naming it would render ``code@localhost``,
        which the reader cannot act on; declining leaves the framework's own
        error, which at least names the socket.
        """
        from aiida_workgraph.errors import MissingInput, MissingRequiredInputsError

        exc = MissingRequiredInputsError([MissingInput("scf.pw.code", "workgraph.code", None)])
        assert advice_for(exc) is None

    def test_a_later_entry_supplies_the_purpose_a_bare_one_lacks(self) -> None:
        """A code's purpose is taken from whichever entry carries one.

        Discriminates first-seen-wins from any-help-wins: the framework
        reports sockets in graph order, so a nested consumer that declares
        no ``help`` can be seen before the one that does, and no route-level
        entry need exist to break the tie. Taking the first entry's ``None``
        would drop a purpose that was there to be shown.
        """
        from aiida_workgraph.errors import MissingInput, MissingRequiredInputsError

        exc = MissingRequiredInputsError(
            [
                MissingInput("scf_nscf.pw_code", "workgraph.code", None),
                MissingInput(
                    "wannierize.codes.pw",
                    "workgraph.code",
                    "Needed to compute DFT ground state properties.",
                ),
            ]
        )
        advice = advice_for(exc)
        assert advice is not None
        assert advice.count("pw@localhost") == 1
        assert "Needed to compute DFT ground state properties." in advice


class TestDispatchTranslation:
    """Each advice-table entry crosses ``build_workgraph`` with its advice."""

    def test_projection_site_error(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_kcp_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """An unknown projection site's advice names the offending label."""
        from aiida_koopmans.projections import ProjectionSiteError

        d = _si_dscf_dict()
        d["calculator_parameters"]["wannier90"]["projections"] = [[{"site": "Ge", "ang_mtm": "sp"}]]
        excinfo = _build_expecting(d, ProjectionSiteError, "does not match any atom")
        assert any(
            "'Ge'" in note and "atomic_positions" in note for note in excinfo.value.__notes__
        )

    def test_block_boundary_error(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_kcw_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """A block spanning both manifolds is advised to split at the boundary."""
        from aiida_koopmans.projections import BlockBoundaryError

        d = _si_dfpt_dict()
        d["calculator_parameters"]["wannier90"]["projections"] = [
            [{"site": "Si", "ang_mtm": "sp3"}]
        ]
        excinfo = _build_expecting(d, BlockBoundaryError, "straddles")
        assert any(
            "split at the occupied/empty boundary" in note for note in excinfo.value.__notes__
        )

    def test_occupied_coverage_error(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_kcw_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """Occupied projections short of the manifold point at the coverage rule."""
        from aiida_koopmans.projections import OccupiedCoverageError

        d = _si_dfpt_dict()
        d["calculator_parameters"]["wannier90"]["projections"] = [[{"site": "Si", "ang_mtm": "s"}]]
        excinfo = _build_expecting(d, OccupiedCoverageError, "occupied projection blocks span")
        assert any(
            "one Wannier function per occupied band" in note for note in excinfo.value.__notes__
        )

    def test_empty_coverage_error(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_kcw_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """An ``nbnd`` undercutting the empty projections points at ``nbnd``."""
        from aiida_koopmans.projections import EmptyCoverageError

        d = _si_dfpt_dict()
        d["calculator_parameters"]["wannier90"]["projections"] = [
            [{"site": "Si", "ang_mtm": "sp"}],
            [{"site": "Si", "ang_mtm": "s"}],
        ]
        d["calculator_parameters"]["nbnd"] = 5
        excinfo = _build_expecting(d, EmptyCoverageError, "leaves only")
        assert any("raise `calculator_parameters.nbnd`" in note for note in excinfo.value.__notes__)

    def test_block_disentanglement_error(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_kcp_code: Any,
        fake_sg15_pseudo_family: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A lower disentangling block is advised to give its bands to the top.

        The derivation gives extra bands only to a channel's top block, so
        the layout is built and then a lower block's ``num_bands`` inflated
        at the route's plugin entry — the raise and the class are the
        plugin's own.
        """
        import aiida_koopmans.workgraphs.kcp as kcp_module
        from aiida_koopmans.projections import (
            BlockDisentanglementError,
            validate_projection_block_sequence,
        )

        def build_with_lower_disentanglement(**kwargs: Any) -> Any:
            """Run the real sequence validator on a lower-disentangling layout."""
            blocks = _derive_si_blocks(kwargs["structure"])
            blocks[0]["num_bands"] = 8
            validate_projection_block_sequence(blocks)

        monkeypatch.setattr(
            kcp_module.KoopmansDSCFWorkflow, "build", build_with_lower_disentanglement
        )
        d = _si_dscf_dict(init_orbitals="kohn-sham")
        excinfo = _build_expecting(d, BlockDisentanglementError, "uppermost block")
        assert any(
            "only the last of the projection blocks" in note for note in excinfo.value.__notes__
        )

    def test_frozen_window_error(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_wannier_codes: Any,
        fake_sg15_cutoffs_family: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A frozen-window rejection points at ``dis_froz_max``.

        The window check reads nscf eigenvalues, which exist only at
        runtime — in production it raises daemon-side, past the build
        boundary, so its entry translates nothing today. The validator is
        invoked at the route's plugin entry with synthetic bands so its
        real raise site fires at build, pinning the entry it provisions.
        """
        import aiida_koopmans.workgraphs.block_wannierize as bw_module
        from aiida.orm import BandsData
        from aiida_koopmans.workgraphs.block_wannierize import (
            FrozenWindowError,
            validate_frozen_window,
        )

        def build_with_bad_window(**kwargs: Any) -> Any:
            """Run the real window validator on bands it must reject."""
            bands = BandsData()
            bands.set_kpoints(np.zeros((1, 3)))  # type: ignore[no-untyped-call]
            bands.set_bands(np.zeros((1, 4)))  # type: ignore[no-untyped-call]
            validate_frozen_window("occ_1", {"dis_froz_max": 10.0, "num_wann": 2}, "none", bands)

        monkeypatch.setattr(bw_module.WannierizeBlocks, "build", build_with_bad_window)
        d = _si_split_dict(block_wannierization_threshold=None)
        excinfo = _build_expecting(d, FrozenWindowError, "frozen")
        assert any(
            "`dis_froz_max`" in note and "block 'occ_1'" in note for note in excinfo.value.__notes__
        )

    def test_missing_required_inputs_error(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_kcp_code: Any,
        fake_sg15_pseudo_family: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A graph-level missing-code socket crosses with install advice.

        The route's pre-check demands what the input turns on, so a real
        instance can only arise past it (a workflow body wiring a member the
        route did not know it needed); the error is raised at the route's
        plugin entry to pin the translation.
        """
        import aiida_koopmans.workgraphs.kcp as kcp_module
        from aiida_workgraph.errors import MissingInput, MissingRequiredInputsError

        def build_missing_a_code(**kwargs: Any) -> Any:
            """Raise the structured error a failed socket validation produces."""
            raise MissingRequiredInputsError(
                [
                    MissingInput(
                        "wannier_initialization.codes.wann2kcp",
                        "workgraph.code",
                        "Needed to initialize the variational orbitals as Wannier functions.",
                    )
                ]
            )

        monkeypatch.setattr(kcp_module.KoopmansDSCFWorkflow, "build", build_missing_a_code)
        d = _si_dscf_dict(init_orbitals="kohn-sham")
        excinfo = _build_expecting(d, MissingRequiredInputsError, "wann2kcp")
        assert any(
            "`wann2kcp@localhost`" in note and "koopmans install" in note
            for note in excinfo.value.__notes__
        )

    def test_parallelization_error(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        fake_sg15_cutoffs_family: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unknown code name raised by the plugin points at `parallelization`.

        The schema rejects such a mapping at parse time, so the plugin's
        validator is reached by stubbing the parsed model's mapping — the
        raise itself is the plugin's.
        """
        from aiida_koopmans.parallelization import ParallelizationError

        from koopmans.input_file.parallelization import ParallelizationInput

        monkeypatch.setattr(
            ParallelizationInput, "as_mapping", lambda self: {"bogus": {"ntasks": 2}}
        )
        excinfo = _build_expecting(
            _pw_input(), ParallelizationError, "unknown parallelization code"
        )
        assert any("`parallelization` block" in note for note in excinfo.value.__notes__)

    def test_model_mismatch_error(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_kcp_code: Any,
        fake_sg15_pseudo_family: Any,
        write_multiframe_xyz: Any,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A model-stamp rejection points at ``ml.model_file``.

        The stamp check runs inside the prediction task, which needs the
        trial KI's descriptors — in production it raises daemon-side, past
        the build boundary, so its entry translates nothing today. Its raw
        callable is invoked at the route's plugin entry with a mismatched
        model so its real raise site fires at build, pinning the entry it
        provisions. ``_callable`` is the raw function under the @task
        handle; the descriptor check raises before descriptors are read.
        """
        import aiida_koopmans.workgraphs.ml as ml_module
        from aiida_koopmans.ml import ModelMismatchError
        from aiida_koopmans.workgraphs.kcp import predict_alpha_screening

        def build_with_bad_model(**kwargs: Any) -> Any:
            """Run the real stamp check on a model it must reject."""
            predict_alpha_screening._callable(
                model={"descriptor": "power_spectrum"},
                descriptor_rows={},
                orbitals=[],
                correction="ki",
                init_orbitals="mlwfs",
                descriptor="self_hartree",
            )

        monkeypatch.setattr(ml_module.TrajectoryWorkflow, "build", build_with_bad_model)
        d = _trajectory_input_dict(str(write_multiframe_xyz(tmp_path, 1)))
        excinfo = _build_expecting(d, ModelMismatchError, "descriptor")
        assert any(
            "ml.model_file" in note and "'descriptor' stamp" in note
            for note in excinfo.value.__notes__
        )

    def test_derivation_invariant_crosses_without_advice(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_kcp_code: Any,
        installed_wannier_codes: Any,
        installed_fold_codes: Any,
        fake_sg15_pseudo_family: Any,
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

        def reversed_blocks(*args: Any, **kwargs: Any) -> Any:
            """Derive the real blocks, reversed."""
            return list(reversed(derive(*args, **kwargs)))

        monkeypatch.setattr(dscf_module, "create_explicit_blocks", reversed_blocks)
        excinfo = _build_expecting(_si_dscf_dict(), ValueError, "ascending band order")
        assert not isinstance(excinfo.value, ProjectionBlockError)
        assert not getattr(excinfo.value, "__notes__", [])
