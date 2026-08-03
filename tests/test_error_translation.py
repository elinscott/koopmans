"""The dispatch boundary's translation of typed plugin errors.

``build_workgraph`` attaches input-file advice, as a PEP 678 note, to the
typed errors aiida-koopmans raises; ``advice_for`` dispatches on the
exception's type. Each typed class gets one test through the dispatcher
with the plugin's own raise site firing, and the plugin's untyped errors
are pinned to pass through untranslated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from koopmans.aiida.workflows import advice_for, build_workgraph
from koopmans.input_file import KoopmansInput
from tests.test_conversion import _pw_input
from tests.test_dscf_mlwf_dispatcher import _si_dscf_dict
from tests.test_trajectory_dispatcher import _trajectory_input_dict
from tests.test_wannierize_blocks_dispatcher import _si_split_dict


def _reversed_blocks_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the DSCF route derive its real blocks in reversed order.

    The block builder emits ascending blocks by construction, so reversing
    its output is the smallest change that makes the plugin's sequence
    validator raise for real.
    """
    import koopmans.aiida.workflows.dscf as dscf_module
    from koopmans.aiida.workflows.blocks import create_explicit_blocks as derive

    def reversed_blocks(*args: Any, **kwargs: Any) -> Any:
        """Derive the real blocks, reversed."""
        return list(reversed(derive(*args, **kwargs)))

    monkeypatch.setattr(dscf_module, "create_explicit_blocks", reversed_blocks)


class TestAdviceFor:
    """``advice_for`` dispatches on the exception's type alone."""

    def test_typed_error_earns_advice(self, aiida_profile: Any) -> None:
        """A real sequence-validator raise carries block-naming advice."""
        from aiida_koopmans.projections import validate_projection_block_sequence
        from aiida_koopmans.spin import SpinChannel

        from koopmans.aiida.conversion import atoms_input_to_structure
        from koopmans.aiida.workflows.blocks import create_explicit_blocks

        inp = KoopmansInput.model_validate(_si_dscf_dict())
        structure = atoms_input_to_structure(inp.atoms)
        blocks = create_explicit_blocks(
            structure, inp.calculator_parameters.wannier90.projections, 8, 4, SpinChannel.NONE
        )
        with pytest.raises(ValueError, match="ascending band order") as excinfo:
            validate_projection_block_sequence(list(reversed(blocks)))
        advice = advice_for(excinfo.value)
        assert advice is not None
        assert "Block 'occ_1'" in advice
        assert "calculator_parameters.w90.projections" in advice

    def test_untyped_plugin_error_gets_no_advice(self) -> None:
        """A plain ValueError from a plugin module passes through untranslated.

        Under the raise-site keying this error would have carried the
        projections advice; type dispatch is what makes it pass through.
        """
        from aiida_koopmans.projections import projection_win_string

        class _SitelessProjection:
            site = None
            ang_mtm = "sp3"

        with pytest.raises(ValueError, match="defines no site") as excinfo:
            projection_win_string(_SitelessProjection())
        assert advice_for(excinfo.value) is None

    def test_local_error_gets_no_advice(self) -> None:
        """An error raised outside the plugin gets no advice either."""
        with pytest.raises(ValueError, match="not the plugin") as excinfo:
            raise ValueError("raised by the dispatcher, not the plugin")
        assert advice_for(excinfo.value) is None


class TestDispatchTranslation:
    """Each typed plugin error crosses ``build_workgraph`` with its advice."""

    def test_projection_block_error(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_kcp_code: Any,
        installed_wannier_codes: Any,
        installed_fold_codes: Any,
        fake_sg15_pseudo_family: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The sequence validator's rejection names the block in its note."""
        from aiida_koopmans.projections import ProjectionBlockError

        _reversed_blocks_patch(monkeypatch)
        inp = KoopmansInput.model_validate(_si_dscf_dict())
        with pytest.raises(ProjectionBlockError, match="ascending band order") as excinfo:
            build_workgraph(inp)
        assert any("Block 'occ_1' is derived" in note for note in excinfo.value.__notes__)

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
        inp = KoopmansInput.model_validate(_pw_input())
        with pytest.raises(ParallelizationError, match="unknown parallelization code") as excinfo:
            build_workgraph(inp)
        assert any("`parallelization` block" in note for note in excinfo.value.__notes__)

    def test_frozen_window_error(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_wannier_codes: Any,
        fake_sg15_cutoffs_family: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A frozen-window rejection points at the `dis_froz_*` keywords.

        The window check reads nscf eigenvalues, which exist only at
        runtime; the plugin's validator is invoked at the route's plugin
        entry with synthetic bands so its real raise site fires at build.
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
        inp = KoopmansInput.model_validate(_si_split_dict(block_wannierization_threshold=None))
        with pytest.raises(FrozenWindowError, match="frozen") as excinfo:
            build_workgraph(inp)
        assert any(
            "`dis_froz_min` / `dis_froz_max`" in note and "block 'occ_1'" in note
            for note in excinfo.value.__notes__
        )

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
        """A model-stamp rejection points at `ml.model_file`.

        The stamp check runs inside the prediction task, which needs the
        trial KI's descriptors; its raw callable is invoked at the route's
        plugin entry with a mismatched model so its real raise site fires
        at build.
        """
        import aiida_koopmans.workgraphs.ml as ml_module
        from aiida_koopmans.ml import ModelMismatchError
        from aiida_koopmans.workgraphs.kcp import predict_alpha_screening

        def build_with_bad_model(**kwargs: Any) -> Any:
            """Run the real stamp check on a model it must reject."""
            # ``_callable`` is the raw function under the @task handle; the
            # descriptor check raises before descriptors or orbitals are read.
            predict_alpha_screening._callable(
                model={"descriptor": "power_spectrum"},
                descriptors=[],
                orbitals=[],
                correction="ki",
                init_orbitals="mlwfs",
            )

        monkeypatch.setattr(ml_module.TrajectoryWorkflow, "build", build_with_bad_model)
        xyz = write_multiframe_xyz(tmp_path, 1)
        inp = KoopmansInput.model_validate(_trajectory_input_dict(str(xyz)))
        with pytest.raises(ModelMismatchError, match="descriptor") as excinfo:
            build_workgraph(inp)
        assert any(
            "ml.model_file" in note and "'descriptor' stamp" in note
            for note in excinfo.value.__notes__
        )
