"""The plain-DFT bands route."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koopmans.aiida.workflows import (
    load_codes,
    name_run,
    pin_spin_regime,
    pin_step_kpoints,
    prepare_common_inputs,
    require_configured_codes,
)

if TYPE_CHECKING:
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput


def build_dft_bands_workgraph(koopmans_input: KoopmansInput) -> WorkGraph:
    """Build a workgraph for DFT bands calculation.

    All four ``workflow.spin`` regimes run here: ``collinear`` needs a
    ``calculator_parameters.tot_magnetization`` alongside it, and
    ``spin_orbit`` a fully-relativistic ``pseudo_library`` (pw.x otherwise
    averages the scalar-relativistic channels and the spin-orbit splitting
    comes out zero).

    Args:
        koopmans_input: The parsed koopmans input.

    Returns:
        A WorkGraph for PwBandsWorkChain.
    """
    from aiida_koopmans.workgraphs.pw import PwBandsCodes, RunPwBands

    from koopmans.aiida.conversion import kpoints_input_to_interpolation_path

    structure, _pseudo_family, overrides = prepare_common_inputs(koopmans_input, ["scf", "bands"])
    spin = pin_spin_regime(koopmans_input, overrides)

    # RunPwBands binds its code eagerly (aiida-koopmans#97: not yet
    # converted to node_graph.reference); the pre-flight catches a missing pw
    # before that bare subscript can raise a bare KeyError.
    codes = load_codes(PwBandsCodes, koopmans_input.computer.name)
    require_configured_codes(PwBandsCodes, codes, koopmans_input.computer.name)

    bands_kpoints = kpoints_input_to_interpolation_path(koopmans_input.kpoints, structure)

    return name_run(
        RunPwBands.build(
            codes=codes,
            structure=structure,
            overrides=overrides,
            parallelization=koopmans_input.parallelization.as_mapping(koopmans_input.computer)
            or None,
            scf_kpoints=pin_step_kpoints(overrides, "scf", koopmans_input),
            bands_kpoints=bands_kpoints,
            spin_type=spin,
        ),
        "DFT band structure",
    )
