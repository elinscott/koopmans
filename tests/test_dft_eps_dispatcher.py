"""Dispatcher tests for the dft_eps (ph.x dielectric constant) task.

Builds real ``WorkGraph`` objects through ``build_workgraph`` against a
throwaway profile (dummy codes, fake pseudos; nothing runs) and checks the
scf → ph.x → extract sequence, plus the ``eps_inf='auto'`` hook of the DFPT
singlepoint stream. Mirrors the style of ``test_dfpt_dispatcher.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.aiida.workflows import build_workgraph
from koopmans.aiida.workflows.dfpt import build_singlepoint_dfpt_workgraph
from koopmans.input_file import KoopmansInput


def _si_eps_dict(**workflow_updates: Any) -> dict[str, Any]:
    """Return a minimal silicon dft_eps input dict (fake SG15 cutoffs pseudos)."""
    d: dict[str, Any] = {
        "workflow": {
            "task": "dft_eps",
            # The cutoffs-family fixture label: the dft_eps builder calls
            # get_builder_from_protocol eagerly, which needs recommended
            # cutoffs (the plain fake family is invisible to its query).
            "pseudo_library": "SG15/1.0/PBE/SR",
        },
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
        "kpoints": {"grid": [2, 2, 2], "offset": [0, 0, 0]},
        "calculator_parameters": {
            "ecutwfc": 20.0,
            "nbnd": 10,
        },
    }
    d["workflow"].update(workflow_updates)
    return d


def _si_dfpt_auto_dict() -> dict[str, Any]:
    """Return a silicon DFPT singlepoint dict with eps_inf='auto'."""
    return {
        "workflow": {
            "task": "singlepoint",
            "correction": "ki",
            "screening_method": "dfpt",
            "init_orbitals": "mlwfs",
            "calculate_alpha": True,
            "pseudo_library": "SG15/1.2/PBE/SR",
            "eps_inf": "auto",
        },
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
        "kpoints": {"grid": [2, 2, 2], "offset": [0, 0, 0]},
        "calculator_parameters": {
            "ecutwfc": 20.0,
            "wannier90": {
                "projections": [[{"site": "Si", "ang_mtm": "sp"}]],
            },
        },
    }


class TestDftEps:
    """task='dft_eps' routes to the scf → ph.x → extract sequence."""

    def test_scf_ph_extract_sequence(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_ph_code: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """The full dispatch builds scf, ph, and the tensor reduction."""
        inp = KoopmansInput.model_validate(_si_eps_dict())
        wg = build_workgraph(inp)
        names = wg.get_task_names()
        assert "scf" in names
        assert "ph" in names
        assert "extract_dielectric_constant" in names

        # ph.x runs the electric-field perturbation only (legacy
        # DFTPhWorkflow: epsil=.true., trans=.false.) at q = Gamma.
        inputph = wg.tasks["ph"].inputs["ph"]["parameters"].value.get_dict()["INPUTPH"]
        assert inputph["epsil"] is True
        assert inputph["trans"] is False
        assert wg.tasks["ph"].inputs["qpoints"].value.get_kpoints_mesh() == (
            [1, 1, 1],
            [0.0, 0.0, 0.0],
        )

        # `nbnd` survives to the scf step like on every other route (koopmans2#162).
        system = wg.tasks["scf"].inputs["pw"]["parameters"].value.get_dict()["SYSTEM"]
        assert system["nbnd"] == 10
        assert system["ecutwfc"] == pytest.approx(20.0)

    def test_scf_samples_the_input_mesh(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_ph_code: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """The ground state the response is taken about uses the input's grid.

        Left to the protocol the scf would pick its own mesh from a k-point
        distance, so the dielectric constant would not be the one the input
        file describes.
        """
        inp = KoopmansInput.model_validate(_si_eps_dict())
        wg = build_workgraph(inp)
        scf = wg.tasks["scf"]
        assert list(scf.inputs["kpoints"].value.get_kpoints_mesh()[0]) == [2, 2, 2]
        assert scf.inputs["kpoints_distance"].value is None

    def test_the_scf_entry_moves_the_scf_mesh(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_ph_code: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """Dielectric constants converge slowly in k, so this scf may want more."""
        d = _si_eps_dict()
        d["kpoints"]["overrides"] = {"scf": {"grid": [4, 4, 4]}}
        wg = build_workgraph(KoopmansInput.model_validate(d))
        scf = wg.tasks["scf"]
        assert list(scf.inputs["kpoints"].value.get_kpoints_mesh()[0]) == [4, 4, 4]

    def test_a_grid_spacing_reaches_the_scf_as_a_distance(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_ph_code: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """The spacing has to displace the protocol's own, with no mesh alongside."""
        d = _si_eps_dict()
        d["kpoints"]["overrides"] = {"scf": {"grid_spacing": 0.11}}
        wg = build_workgraph(KoopmansInput.model_validate(d))
        scf = wg.tasks["scf"]
        assert scf.inputs["kpoints"].value is None
        assert float(scf.inputs["kpoints_distance"].value) == pytest.approx(0.11)

    def test_an_nscf_entry_is_rejected(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_ph_code: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """There is no nscf step here for the mesh to reach."""
        d = _si_eps_dict()
        d["kpoints"]["overrides"] = {"nscf": {"grid": [4, 4, 4]}}
        with pytest.raises(ValueError, match=r"overrides\.nscf.*dft_eps"):
            build_workgraph(KoopmansInput.model_validate(d))

    def test_a_band_path_is_rejected(self, read_input_dict: Any) -> None:
        """ph.x computes a dielectric constant, so a path here would never be sampled.

        Refused while the input file is read, so the reader gets the error
        report rather than a traceback out of the graph build.
        """
        d = _si_eps_dict()
        d["kpoints"]["path"] = "GX"

        with pytest.raises(ValueError) as excinfo:
            read_input_dict(d)

        message = str(excinfo.value)
        assert "Errors found in the input file" in message
        assert "`kpoints.path`" in message
        assert "dft_bands" in message

    def test_missing_ph_code_earns_preflight_advice(
        self, aiida_profile_clean: Any, installed_pw_code: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """Without a ph@localhost code, the dispatcher's pre-flight names it before build.

        ``DielectricTask`` — the dft_eps route's own entry graph — feeds
        ``codes["ph"]`` (and ``codes["pw"]``) straight into an eager
        ``get_builder_from_protocol`` call, which needs a concrete
        ``orm.Code`` and cannot take a lazy ``node_graph.reference()``; left to
        its own devices that bind would die as a bare ``KeyError`` mid-trace
        (known gap on the plugin side: aiida-koopmans#88, tracking upstream
        node-graph#169; #90 has not reached this call site yet — not a k2
        defect). ``require_configured_codes``, called right after
        ``load_codes`` in ``eps.py``, intercepts first: ``ph`` is required
        in ``DielectricCodes``, so the dispatcher's own pre-flight raises
        the install advice before ``DielectricTask.build()`` ever runs. See
        ``tests/test_code_loading.py``'s ``TestPreFlightAdvice``. Requests
        ``aiida_profile_clean``: earlier tests may have installed a
        ``ph@localhost`` code in the session profile.
        """
        inp = KoopmansInput.model_validate(_si_eps_dict())
        with pytest.raises(ValueError, match="`ph@localhost`") as excinfo:
            build_workgraph(inp)
        assert "koopmans install" in str(excinfo.value)


class TestDfptAutoEps:
    """eps_inf='auto' prepends the dielectric steps to the DFPT stream."""

    @pytest.fixture
    def dfpt_codes(
        self, installed_pw_code: Any, installed_kcw_code: Any, installed_wannier_codes: Any
    ) -> dict[str, Any]:
        """Register the dummy DFPT codes the route resolves as ``<name>@localhost``."""
        return {
            "pw": installed_pw_code,
            "kcw": installed_kcw_code,
            **installed_wannier_codes,
        }

    def test_auto_adds_dielectric_task(
        self,
        aiida_profile: Any,
        dfpt_codes: Any,
        installed_ph_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """A 'dielectric' task appears alongside the kcw steps."""
        inp = KoopmansInput.model_validate(_si_dfpt_auto_dict())
        wg = build_singlepoint_dfpt_workgraph(inp)
        names = wg.get_task_names()
        assert "dielectric" in names
        assert "dfpt" in names

    def test_both_ground_states_sample_the_input_mesh(
        self,
        aiida_profile: Any,
        dfpt_codes: Any,
        installed_ph_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """The dielectric ground state samples the input mesh, like the main one.

        Both ground states here answer to the same input file, so leaving
        one of them on the protocol would make the graph depend on the cell
        in a way the input does not record. A dielectric constant converges
        slowly with k-sampling, so a user who wants a denser mesh for it
        needs to say so; per-step meshes are koopmans#50.
        """
        inp = KoopmansInput.model_validate(_si_dfpt_auto_dict())
        wg = build_singlepoint_dfpt_workgraph(inp)
        eps_mesh = wg.tasks["dielectric"].inputs["scf_kpoints"].value
        main_mesh = wg.tasks["scf_nscf"].inputs["scf_kpoints"].value
        assert list(eps_mesh.get_kpoints_mesh()[0]) == [2, 2, 2]
        assert list(main_mesh.get_kpoints_mesh()[0]) == [2, 2, 2]

    def test_auto_without_ph_code_builds_then_fails_at_check(
        self, aiida_profile_clean: Any, dfpt_codes: Any, fake_sg15_pseudo_family: Any
    ) -> None:
        """eps_inf='auto' without ph@localhost builds fine and fails at check/submit.

        ``SinglepointDFPTWorkflow`` threads ``ph`` into its dielectric
        pre-computation step through ``node_graph.reference``, so a missing
        code leaves an unlinked socket rather than dying in the eager
        trace: the build succeeds, and ``check_before_run`` — what
        ``run``/``submit`` call before doing anything else — is what
        raises. ``advice_for`` then names the code and quotes the
        purpose its codes TypedDict declares. Requests
        ``aiida_profile_clean``: earlier tests may have installed a
        ``ph@localhost`` code in the session profile.
        """
        from aiida_workgraph.errors import MissingRequiredInputsError

        from koopmans.aiida.workflows import advice_for

        inp = KoopmansInput.model_validate(_si_dfpt_auto_dict())
        wg = build_singlepoint_dfpt_workgraph(inp)

        with pytest.raises(MissingRequiredInputsError) as excinfo:
            wg.check_before_run()

        advice = advice_for(excinfo.value)
        assert advice is not None
        assert "`ph@localhost`" in advice
        assert "Needed to compute the dielectric constant." in advice
        assert "koopmans install" in advice

    def test_unknown_eps_string_raises(self, dfpt_codes: Any) -> None:
        """A non-'auto' string eps_inf is rejected up front."""
        d = _si_dfpt_auto_dict()
        d["workflow"]["eps_inf"] = "automatic"
        inp = KoopmansInput.model_validate(d)
        with pytest.raises(ValueError, match="not understood"):
            build_singlepoint_dfpt_workgraph(inp)


class TestPhCalculatorParameters:
    """``calculator_parameters.ph`` reaches the ph.x step (koopmans2#162)."""

    def test_tr2_ph_reaches_the_ph_task(
        self,
        aiida_profile: Any,
        installed_pw_code: Any,
        installed_ph_code: Any,
        fake_sg15_cutoffs_family: Any,
    ) -> None:
        """A user-tightened ``tr2_ph`` lands in the ph.x ``INPUTPH`` namelist.

        The route's own ``epsil``/``trans`` keys still win underneath it.
        """
        d = _si_eps_dict()
        d["calculator_parameters"]["ph"] = {"tr2_ph": 1.0e-14}
        wg = build_workgraph(KoopmansInput.model_validate(d))
        inputph = wg.tasks["ph"].inputs["ph"]["parameters"].value.get_dict()["INPUTPH"]
        assert inputph["tr2_ph"] == pytest.approx(1.0e-14)
        assert inputph["epsil"] is True
        assert inputph["trans"] is False

    def test_epsil_is_rejected_at_parse(self) -> None:
        """A user-set ``epsil`` disagreeing with the route fails at parse, naming the owner."""
        d = _si_eps_dict()
        d["calculator_parameters"]["ph"] = {"epsil": False}
        with pytest.raises(ValueError, match=r"calculator_parameters\.ph\.epsil.*dft_eps route"):
            KoopmansInput.model_validate(d)
