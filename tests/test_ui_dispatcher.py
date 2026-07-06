"""Dispatcher tests for the standalone UI (unfold-and-interpolate) task.

Builds real ``WorkGraph`` objects through ``_build_ui_workgraph`` /
``build_workgraph`` against a throwaway profile. The input Hamiltonian /
``.wout`` files in ``tests/data/ui/`` come from the legacy test suite
(silicon, 2x2x2 grid); nothing runs — only construction and the
missing-file error paths are checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from koopmans.aiida.workflows import _build_ui_workgraph, build_workgraph, load_codes_for_task
from koopmans.input_file import KoopmansInput

DATA_DIR = Path(__file__).parent / "data" / "ui"


def _si_ui_dict(**ui_updates: Any) -> dict[str, Any]:
    """Return a silicon UI input dict pointing at the test data files."""
    ui: dict[str, Any] = {
        "kc_ham_file": str(DATA_DIR / "kc_ham.dat"),
        "wannier90_seedname": str(DATA_DIR / "wann"),
        "dft_ham_file": str(DATA_DIR / "dft_ham.dat"),
        "dft_smooth_ham_file": str(DATA_DIR / "smooth_dft_ham.dat"),
        "smooth_int_factor": 2,
        "do_map": True,
        "use_ws_distance": True,
        "do_dos": True,
    }
    ui.update(ui_updates)
    return {
        # pseudo_library is schema-required but unused by the UI task.
        "workflow": {"task": "ui", "pseudo_library": "SG15/1.2/PBE/SR"},
        "atoms": {
            "cell_parameters": {
                "periodic": True,
                "ibrav": 2,
                "celldms": {"1": 10.2622},
            },
            "atomic_positions": {
                "units": "crystal",
                "positions": [["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25]],
            },
        },
        "kpoints": {"grid": [2, 2, 2], "offset": [0, 0, 0], "path": "GL"},
        "calculator_parameters": {"ui": ui},
        "plotting": {"degauss": 0.05, "nstep": 1000, "Emin": -10, "Emax": 4},
    }


def _build(d: dict[str, Any]):
    inp = KoopmansInput.model_validate(d)
    return _build_ui_workgraph(inp, codes={})


class TestCodes:
    """The UI task needs no QE codes."""

    def test_load_codes_returns_empty(self):
        """load_codes_for_task must not try to load pw.x for the UI task."""
        inp = KoopmansInput.model_validate(_si_ui_dict())
        assert load_codes_for_task(inp.workflow) == {}


class TestBuild:
    """Workgraph construction from the input file."""

    def test_builds_interpolation_and_dos_tasks(self, aiida_profile):
        """The full input wires the interpolation and DOS calcfunctions."""
        wg = _build(_si_ui_dict())
        names = wg.get_task_names()
        assert "interpolate_bands" in names
        assert "compute_dos_from_bands" in names

    def test_do_dos_false_skips_the_dos(self, aiida_profile):
        """do_dos=False leaves only the interpolation task."""
        wg = _build(_si_ui_dict(do_dos=False))
        names = wg.get_task_names()
        assert "interpolate_bands" in names
        assert "compute_dos_from_bands" not in names

    def test_dispatches_via_build_workgraph(self, aiida_profile):
        """Task.UI routes through the top-level dispatcher without any codes."""
        wg = build_workgraph(KoopmansInput.model_validate(_si_ui_dict()))
        assert "interpolate_bands" in wg.get_task_names()

    def test_ui_kpath_overrides_kpoints_path(self, aiida_profile):
        """A ui.kpath string replaces the kpoints path when generating the k-path."""
        wg = _build(_si_ui_dict(kpath="GX"))
        assert "interpolate_bands" in wg.get_task_names()


class TestMissingFiles:
    """Clear errors for unset or dangling file paths."""

    def test_missing_kc_ham_file_raises(self):
        """kc_ham_file is mandatory."""
        with pytest.raises(ValueError, match="kc_ham_file"):
            _build(_si_ui_dict(kc_ham_file=None))

    def test_nonexistent_kc_ham_file_raises(self):
        """A dangling kc_ham_file path is reported."""
        with pytest.raises(ValueError, match="does not exist"):
            _build(_si_ui_dict(kc_ham_file="/nonexistent/kc_ham.dat"))

    def test_nonexistent_wout_raises(self):
        """The .wout derived from wannier90_seedname must exist."""
        with pytest.raises(ValueError, match="wannier90_seedname"):
            _build(_si_ui_dict(wannier90_seedname="/nonexistent/wann"))

    def test_smooth_interpolation_requires_smooth_ham(self):
        """smooth_int_factor > 1 without dft_smooth_ham_file is an error."""
        with pytest.raises(ValueError, match="dft_smooth_ham_file"):
            _build(_si_ui_dict(dft_smooth_ham_file=None))

    def test_no_smoothing_ignores_dft_ham_files(self, aiida_profile):
        """With smooth_int_factor=1 the DFT Hamiltonian paths are not required."""
        wg = _build(_si_ui_dict(smooth_int_factor=1, dft_ham_file=None, dft_smooth_ham_file=None))
        assert "interpolate_bands" in wg.get_task_names()
