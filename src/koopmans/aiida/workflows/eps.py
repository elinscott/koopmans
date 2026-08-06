"""The dielectric-constant (ph.x) route."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koopmans.aiida.workflows import prepare_common_inputs

if TYPE_CHECKING:
    from aiida import orm
    from aiida_koopmans.parallelization import ParallelizationDict
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput


def build_dft_eps_workgraph(
    koopmans_input: KoopmansInput,
    codes: dict[str, orm.AbstractCode],
    parallelization: ParallelizationDict,
) -> WorkGraph:
    """Build a workgraph for the dielectric-constant (ph.x) task.

    Port of the legacy ``DFTPhWorkflow`` (``workflows/_dft.py``): one scf,
    then ph.x with ``epsil = .true.`` / ``trans = .false.`` at q = Gamma,
    exposing the isotropic average of the dielectric tensor as ``eps_inf``.
    The legacy scf passes ``nbnd=None`` (no empty bands are needed for a
    ground-state response), so ``nbnd`` is stripped from the PW overrides.

    Args:
        koopmans_input: The parsed koopmans input.
        codes: Dictionary of loaded codes.
        parallelization: The per-code mapping, rank counts already completed.

    Returns:
        A WorkGraph chaining PwBaseWorkChain into PhBaseWorkChain.
    """
    from aiida_koopmans.workgraphs.ph import DielectricTask

    from koopmans.aiida.conversion import kpoints_input_to_kpoints_mesh

    structure, pseudo_family, overrides = prepare_common_inputs(
        koopmans_input, ["scf"], parallelization
    )
    overrides["scf"]["pw"]["parameters"].get("SYSTEM", {}).pop("nbnd", None)

    return DielectricTask.build(
        pw_code=codes["pw"],
        ph_code=codes["ph"],
        structure=structure,
        pseudo_family=pseudo_family,
        overrides=overrides,
        parallelization=parallelization or None,
        scf_kpoints=kpoints_input_to_kpoints_mesh(koopmans_input.kpoints),
    )
