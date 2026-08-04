"""The plain-DFT bands route."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koopmans.aiida.workflows import prepare_common_inputs

if TYPE_CHECKING:
    from aiida_koopmans.workgraphs import Codes
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput


def build_dft_bands_workgraph(
    koopmans_input: KoopmansInput,
    codes: Codes,
) -> WorkGraph:
    """Build a workgraph for DFT bands calculation.

    Args:
        koopmans_input: The parsed koopmans input.
        codes: Dictionary of loaded codes.

    Returns:
        A WorkGraph for PwBandsWorkChain.
    """
    from aiida_koopmans.workgraphs.pw import RunPwBands

    from koopmans.aiida.conversion import kpoints_input_to_kpoints_mesh

    structure, _pseudo_family, overrides = prepare_common_inputs(koopmans_input, ["scf", "bands"])

    return RunPwBands.build(
        code=codes["pw"],
        structure=structure,
        overrides=overrides,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        scf_kpoints=kpoints_input_to_kpoints_mesh(koopmans_input.kpoints),
    )
