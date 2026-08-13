"""The plain-DFT bands route."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koopmans.aiida.workflows import (
    load_codes,
    pin_step_kpoints,
    prepare_common_inputs,
    reject_kpoint_overrides,
    require_configured_codes,
)

if TYPE_CHECKING:
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput


def build_dft_bands_workgraph(koopmans_input: KoopmansInput) -> WorkGraph:
    """Build a workgraph for DFT bands calculation.

    Args:
        koopmans_input: The parsed koopmans input.

    Returns:
        A WorkGraph for PwBandsWorkChain.
    """
    from aiida_koopmans.workgraphs.pw import PwBandsCodes, RunPwBands

    from koopmans.aiida.conversion import kpoints_input_to_interpolation_path

    reject_kpoint_overrides(
        koopmans_input,
        {
            "nscf": "`kpoints.overrides.nscf` cannot take effect in a `dft_bands` "
            "calculation: it runs an scf and then samples `kpoints.path`, with no "
            "nscf mesh in between. Set `kpoints.overrides.scf` or `kpoints.grid`."
        },
    )

    structure, _pseudo_family, overrides = prepare_common_inputs(koopmans_input, ["scf", "bands"])

    # RunPwBands binds its code eagerly (aiida-koopmans#97: not yet
    # converted to node_graph.reference); the pre-flight catches a missing pw
    # before that bare subscript can raise a bare KeyError.
    codes = load_codes(PwBandsCodes)
    require_configured_codes(PwBandsCodes, codes)

    bands_kpoints = kpoints_input_to_interpolation_path(koopmans_input.kpoints, structure)

    return RunPwBands.build(
        codes=codes,
        structure=structure,
        overrides=overrides,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        scf_kpoints=pin_step_kpoints(overrides, "scf", koopmans_input),
        bands_kpoints=bands_kpoints,
    )
