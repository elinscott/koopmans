"""The Bethe-Salpeter (yambo BSE, seeded by Koopmans eigenvalues) route."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiida_quantumespresso.common.types import SpinType

from koopmans.aiida.workflows import load_codes, name_run, require_configured_codes
from koopmans.aiida.workflows.dfpt import assemble_dfpt_chain_inputs
from koopmans.aiida.workflows.grouping import dfpt_grouping_tol
from koopmans.input_file.workflow import CalculateScreeningMethod

if TYPE_CHECKING:
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput
    from koopmans.input_file.bse import BSEInput


def build_bse_workgraph(koopmans_input: KoopmansInput) -> WorkGraph:
    """Build a workgraph for a yambo BSE spectrum on Koopmans (KI) eigenvalues.

    Composes the DFPT singlepoint chain (scf + nscf → wannierization →
    wann2kc → screen → ham) with a fresh yambo scf → nscf → p2y → BSE chain
    seeded by the ham step's eigenvalues, via
    ``aiida_koopmans.workgraphs.bethe_salpeter.SinglepointBetheSalpeterWorkflow``.
    Shares its ground-state/screening input assembly with the plain DFPT
    route (:func:`koopmans.aiida.workflows.dfpt.assemble_dfpt_chain_inputs`),
    so every restriction that route documents applies here too.

    Phase-1 scope, refused explicitly because the composed workflow does not
    expose a socket for them: ``screening_method`` must be ``'dfpt'``,
    ``spin`` must be ``'none'`` (a single DFPT channel to read), and
    ``workflow.eps_inf``, ``workflow.gb_correction``, workflow-level orbital
    grouping (``group_orbitals_by`` other than ``'none'``), and
    ``calculator_parameters.kcw`` overrides all take no effect — the
    composed DFPT step always runs with their defaults.

    Args:
        koopmans_input: The parsed koopmans input.

    Returns:
        A WorkGraph chaining the DFPT singlepoint into the yambo BSE run.

    Raises:
        ValueError: If `koopmans_input.bse` is unset (guarded at parse
            time already), `calculator_parameters.ecutwfc` is unset, or
            `bse.bands` reaches past the Wannierized manifold.
        NotImplementedError: If the input asks for a `bse`-incompatible
            `screening_method`, `spin`, `eps_inf`, `gb_correction`,
            orbital grouping, or `kcw` override.
    """
    from aiida_koopmans.workgraphs.bethe_salpeter import (
        SinglepointBetheSalpeterCodes,
        SinglepointBetheSalpeterWorkflow,
    )

    from koopmans.aiida.conversion import input_to_bse_parameters

    workflow = koopmans_input.workflow
    bse: BSEInput | None = koopmans_input.bse
    if bse is None:
        # Already guarded by KoopmansInput.check_bse_block_matches_task; a
        # narrowing re-check so the type checker (and any future caller that
        # skips parsing) sees a validated BSEInput below.
        raise ValueError("`workflow.task: bse` needs a `bse` input block.")

    if workflow.screening_method != CalculateScreeningMethod.DFPT:
        raise NotImplementedError(
            "the `bse` task only supports `screening_method: dfpt`: it seeds the yambo "
            "BSE run off a kcw.x `ham` run's eigenvalues, which only DFPT screening "
            f"produces; got screening_method={workflow.screening_method.value!r}."
        )
    if workflow.spin != SpinType.NONE:
        raise NotImplementedError(
            "the `bse` task only supports `spin: none`: aiida-koopmans's "
            "SinglepointBetheSalpeterWorkflow reads a single DFPT channel, and a "
            f"collinear or spinor run has none; got spin={workflow.spin.value!r}. Run "
            "`task: singlepoint` and a standalone BSE workflow separately for those."
        )
    if workflow.eps_inf is not None:
        raise NotImplementedError(
            "`workflow.eps_inf` is not yet wired into the `bse` task: its composed DFPT "
            "screening step always runs kcw.x's own default dielectric constant. Leave "
            "it unset."
        )
    if workflow.gb_correction is not None:
        raise NotImplementedError(
            "`workflow.gb_correction` is not yet wired into the `bse` task: its "
            "composed DFPT screening step always applies the Gygi-Baldereschi scheme. "
            "Leave it unset."
        )
    if dfpt_grouping_tol(workflow) is not None:
        raise NotImplementedError(
            "`workflow.group_orbitals_by` is not yet wired into the `bse` task: its "
            "composed DFPT screening step runs no workflow-level orbital grouping. "
            "Leave it unset (or 'none')."
        )
    if koopmans_input.calculator_parameters.ecutwfc is None:
        raise ValueError(
            "the `bse` task needs `calculator_parameters.ecutwfc` set explicitly: the "
            "yambo BSE chain reruns its own scf/nscf/p2y ground state and must match "
            "the DFPT chain's cutoff exactly, which a pseudopotential family's "
            "recommended cutoff cannot guarantee."
        )

    chain_inputs = assemble_dfpt_chain_inputs(koopmans_input)

    kcw_stated = chain_inputs["kcw_overrides"]
    if kcw_stated:
        raise NotImplementedError(
            "`calculator_parameters.kcw` is not yet wired into the `bse` task: its "
            f"composed DFPT screening step takes no kcw.x namelist overrides. Stated: "
            f"{sorted(kcw_stated)}. Remove them."
        )

    n_orbitals = chain_inputs["manifold_band_counts"]["none"]
    first, last = bse.bands
    if last > n_orbitals:
        raise ValueError(
            f"`bse.bands` = [{first}, {last}] reaches past the Wannierized manifold, "
            f"which the DFPT chain builds for bands 1..{n_orbitals}: Koopmans "
            "quasiparticle corrections exist only there. Lower `bse.bands`, or widen "
            "the manifold (`calculator_parameters.nbnd` / the empty projections)."
        )

    codes = load_codes(SinglepointBetheSalpeterCodes, koopmans_input.computer.name)
    require_configured_codes(SinglepointBetheSalpeterCodes, codes, koopmans_input.computer.name)

    return name_run(
        SinglepointBetheSalpeterWorkflow.build(
            codes=codes,
            structure=chain_inputs["structure"],
            manifolds=chain_inputs["manifolds"],
            kpoints=chain_inputs["kpoints"],
            bse_parameters=input_to_bse_parameters(koopmans_input),
            scf_kpoints=chain_inputs["scf_kpoints"],
            pseudo_family=chain_inputs["pseudo_family"],
            overrides=chain_inputs["overrides"],
            eigenvalues=bse.eigenvalues,
            parallelization=chain_inputs["parallelization"],
        ),
        "Koopmans BSE",
    )
