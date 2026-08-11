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

    reject_kpoint_overrides(
        koopmans_input,
        {
            "nscf": "`kpoints.overrides.nscf` cannot take effect in a `dft_bands` "
            "calculation: it runs an scf and then samples `kpoints.path`, with no "
            "nscf mesh in between. Set `kpoints.overrides.scf` or `kpoints.grid`."
        },
    )

    structure, _pseudo_family, overrides = prepare_common_inputs(koopmans_input, ["scf", "bands"])

    return RunPwBands.build(
        codes=load_codes(PwBandsCodes),
        structure=structure,
        overrides=overrides,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        scf_kpoints=pin_step_kpoints(overrides, "scf", koopmans_input),
    )
