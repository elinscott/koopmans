"""The plain-DFT bands route."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koopmans.aiida.workflows import (
    load_codes,
    pin_step_kpoints,
    prepare_common_inputs,
    reject_kpoint_overrides,
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

    from koopmans.aiida.conversion import kpoints_input_to_kpoints_path

    reject_kpoint_overrides(
        koopmans_input,
        {
            "nscf": "`kpoints.overrides.nscf` cannot take effect in a `dft_bands` "
            "calculation: it runs an scf and then samples `kpoints.path`, with no "
            "nscf mesh in between. Set `kpoints.overrides.scf` or `kpoints.grid`."
        },
    )

    structure, _pseudo_family, overrides = prepare_common_inputs(koopmans_input, ["scf", "bands"])

    # A gamma-only input's fixed path names the zone centre alone, so it
    # defines no segment to sample: leave the step on its protocol default.
    bands_kpoints = (
        None
        if koopmans_input.kpoints.gamma_only or koopmans_input.kpoints.path is None
        else kpoints_input_to_kpoints_path(koopmans_input.kpoints, structure)
    )

    return RunPwBands.build(
        codes=load_codes(PwBandsCodes),
        structure=structure,
        overrides=overrides,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        scf_kpoints=pin_step_kpoints(overrides, "scf", koopmans_input),
        bands_kpoints=bands_kpoints,
    )
