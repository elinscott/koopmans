"""The dielectric-constant (ph.x) route."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koopmans.aiida.workflows import (
    load_codes,
    name_run,
    pin_step_kpoints,
    prepare_common_inputs,
    reject_kpoint_overrides,
    require_configured_codes,
)

if TYPE_CHECKING:
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput


def build_dft_eps_workgraph(koopmans_input: KoopmansInput) -> WorkGraph:
    """Build a workgraph for the dielectric-constant (ph.x) task.

    Port of the legacy ``DFTPhWorkflow`` (``workflows/_dft.py``): one scf,
    then ph.x with ``epsil = .true.`` / ``trans = .false.`` at q = Gamma,
    exposing the isotropic average of the dielectric tensor as ``eps_inf``.
    A ``calculator_parameters.pw.system.nbnd`` (or the top-level ``nbnd``
    shorthand) survives to the scf step like it does on every other route;
    left unset, the scf takes QE's default band count, matching legacy's
    ground-state response (no empty bands needed). ``calculator_parameters.ph``
    reaches the ph.x step underneath the route's own ``epsil``/``trans``/
    q-mesh keys.

    Args:
        koopmans_input: The parsed koopmans input.

    Returns:
        A WorkGraph chaining PwBaseWorkChain into PhBaseWorkChain.
    """
    from aiida_koopmans.workgraphs.ph import DielectricCodes, DielectricTask

    from koopmans.aiida.conversion import input_to_ph_parameters

    reject_kpoint_overrides(
        koopmans_input,
        {
            "nscf": "`kpoints.overrides.nscf` cannot take effect in a `dft_eps` "
            "calculation: it runs one scf and then ph.x, with no nscf step. Set "
            "`kpoints.overrides.scf` or `kpoints.grid`."
        },
    )

    structure, pseudo_family, overrides = prepare_common_inputs(koopmans_input, ["scf"])
    overrides["ph"] = {"ph": {"parameters": input_to_ph_parameters(koopmans_input)}}

    # DielectricTask binds both codes eagerly (aiida-koopmans#97: not
    # yet converted to node_graph.reference); the pre-flight catches a missing
    # pw or ph before that bare subscript can raise a bare KeyError.
    codes = load_codes(DielectricCodes)
    require_configured_codes(DielectricCodes, codes)

    return name_run(
        DielectricTask.build(
            codes=codes,
            structure=structure,
            pseudo_family=pseudo_family,
            overrides=overrides,
            parallelization=koopmans_input.parallelization.as_mapping() or None,
            scf_kpoints=pin_step_kpoints(overrides, "scf", koopmans_input),
        ),
        "Dielectric constant",
    )
