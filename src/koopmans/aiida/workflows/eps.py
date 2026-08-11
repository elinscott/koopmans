"""The dielectric-constant (ph.x) route."""

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


def build_dft_eps_workgraph(koopmans_input: KoopmansInput) -> WorkGraph:
    """Build a workgraph for the dielectric-constant (ph.x) task.

    Port of the legacy ``DFTPhWorkflow`` (``workflows/_dft.py``): one scf,
    then ph.x with ``epsil = .true.`` / ``trans = .false.`` at q = Gamma,
    exposing the isotropic average of the dielectric tensor as ``eps_inf``.
    The legacy scf passes ``nbnd=None`` (no empty bands are needed for a
    ground-state response), so ``nbnd`` is stripped from the PW overrides.

    Args:
        koopmans_input: The parsed koopmans input.

    Returns:
        A WorkGraph chaining PwBaseWorkChain into PhBaseWorkChain.
    """
    from aiida_koopmans.workgraphs.ph import DielectricCodes, DielectricTask

    reject_kpoint_overrides(
        koopmans_input,
        {
            "nscf": "`kpoints.overrides.nscf` cannot take effect in a `dft_eps` "
            "calculation: it runs one scf and then ph.x, with no nscf step. Set "
            "`kpoints.overrides.scf` or `kpoints.grid`."
        },
    )

    structure, pseudo_family, overrides = prepare_common_inputs(koopmans_input, ["scf"])
    overrides["scf"]["pw"]["parameters"].get("SYSTEM", {}).pop("nbnd", None)

    return DielectricTask.build(
        codes=load_codes(DielectricCodes),
        structure=structure,
        pseudo_family=pseudo_family,
        overrides=overrides,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        scf_kpoints=pin_step_kpoints(overrides, "scf", koopmans_input),
    )
