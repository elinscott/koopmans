"""The Bethe-Salpeter (yambo BSE, seeded by Koopmans eigenvalues) route."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiida_quantumespresso.common.types import SpinType

from koopmans.aiida.workflows import load_codes, name_run, require_configured_codes
from koopmans.aiida.workflows.dfpt import DfptChainInputs, assemble_dfpt_chain_inputs
from koopmans.input_file.workflow import CalculateScreeningMethod

if TYPE_CHECKING:
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput
    from koopmans.input_file.yambo import YamboBseParameters


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
    ``workflow.eps_inf``, ``workflow.gb_correction``, and
    ``calculator_parameters.kcw`` overrides all take no effect — the
    composed DFPT step always runs with their defaults.

    Left unrefused, by contrast: ``calculator_parameters.ecutwfc``, if
    unset, is derived from the pseudopotential family's own recommendation
    (the same call ``PwBaseWorkChain.get_builder_from_protocol`` makes), so
    both this route's fresh yambo ground state and the composed DFPT
    chain's own run on the identical cutoff without the user typing it.
    Workflow-level orbital grouping (``group_orbitals_by`` /
    ``group_orbitals_tol``) also reaches no calculation here — the composed
    workflow forwards no grouping tolerance into its DFPT chain — but is
    not refused either: grouping only changes how the screening parameters
    are *computed* (sharing one value across orbitals presumed equivalent),
    never what they converge to, so running the full, ungrouped DFPT chain
    underneath is a strictly more faithful, only slower, substitute.

    Args:
        koopmans_input: The parsed koopmans input.

    Returns:
        A WorkGraph chaining the DFPT singlepoint into the yambo BSE run.

    Raises:
        ValueError: If `koopmans_input.calculator_parameters.yambo` is unset
            (guarded at parse time already), the pseudopotential family
            recommends no cutoffs and `calculator_parameters.ecutwfc` is
            also unset, or `yambo.BSEBands` reaches past the Wannierized
            manifold.
        NotImplementedError: If the input asks for a `bse`-incompatible
            `screening_method`, `spin`, `eps_inf`, `gb_correction`, or
            `kcw` override.
    """
    from aiida_koopmans.workgraphs.bethe_salpeter import (
        SinglepointBetheSalpeterCodes,
        SinglepointBetheSalpeterWorkflow,
    )

    from koopmans.aiida.conversion import yambo_input_to_bse_parameters

    workflow = koopmans_input.workflow
    yambo: YamboBseParameters | None = koopmans_input.calculator_parameters.yambo
    if yambo is None:
        # Already guarded by KoopmansInput.check_yambo_block_matches_task; a
        # narrowing re-check so the type checker (and any future caller that
        # skips parsing) sees a validated YamboBseParameters below.
        raise ValueError("`workflow.task: bse` needs a `calculator_parameters.yambo` input block.")

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
            "`workflow.eps_inf` is not yet wired into the `bse` task: "
            "`SinglepointBetheSalpeterWorkflow` takes no `eps_inf` of its own — its "
            "composed DFPT screening step always runs kcw.x's own default dielectric "
            "constant. Leave it unset."
        )
    if workflow.gb_correction is not None:
        raise NotImplementedError(
            "`workflow.gb_correction` is not yet wired into the `bse` task: "
            "`SinglepointBetheSalpeterWorkflow` takes no `l_vcut` of its own — its "
            "composed DFPT screening step always applies the Gygi-Baldereschi scheme. "
            "Leave it unset."
        )

    chain_inputs, manifold_band_counts = assemble_dfpt_chain_inputs(koopmans_input)
    _ensure_explicit_pw_cutoffs(chain_inputs)

    kcw_stated = chain_inputs["kcw_overrides"]
    if kcw_stated:
        raise NotImplementedError(
            "`calculator_parameters.kcw` is not yet wired into the `bse` task: "
            "`SinglepointBetheSalpeterWorkflow` takes no `kcw_overrides` of its own — "
            f"its composed DFPT screening step takes no kcw.x namelist overrides. "
            f"Stated: {sorted(kcw_stated)}. Remove them."
        )

    n_orbitals = manifold_band_counts["none"]
    first, last = yambo.BSEBands
    if last > n_orbitals:
        raise ValueError(
            f"`calculator_parameters.yambo.BSEBands` = [{first}, {last}] reaches past "
            f"the Wannierized manifold, which the DFPT chain builds for bands "
            f"1..{n_orbitals}: Koopmans quasiparticle corrections exist only there. "
            "Lower `BSEBands`, or widen the manifold (`calculator_parameters.nbnd` / "
            "the empty projections)."
        )

    codes = load_codes(SinglepointBetheSalpeterCodes, koopmans_input.computer.name)
    require_configured_codes(SinglepointBetheSalpeterCodes, codes, koopmans_input.computer.name)

    return name_run(
        SinglepointBetheSalpeterWorkflow.build(
            codes=codes,
            structure=chain_inputs["structure"],
            manifolds=chain_inputs["manifolds"],
            kpoints=chain_inputs["kpoints"],
            bse_parameters=yambo_input_to_bse_parameters(koopmans_input),
            scf_kpoints=chain_inputs["scf_kpoints"],
            pseudo_family=chain_inputs["pseudo_family"],
            overrides=chain_inputs["overrides"],
            # aiida-koopmans#134: no ak2 kcw.x `ham` parser emits `pki_eigenvalues_on_grid`
            # yet, so `pki` has no producer; the `eigenvalues` flavour knob returns once it
            # does.
            eigenvalues="ki",
            parallelization=chain_inputs["parallelization"],
        ),
        "Koopmans BSE",
    )


def _ensure_explicit_pw_cutoffs(chain_inputs: DfptChainInputs) -> None:
    """Backfill a literal ``ecutwfc``/``ecutrho`` into the shared scf overrides, in place.

    ``RunBetheSalpeter``'s own fresh scf/nscf reads both cutoffs straight
    out of ``overrides['scf']['pw']['parameters']['SYSTEM']`` rather than
    through a protocol build (see
    ``aiida_koopmans.workgraphs.bethe_salpeter._pw_cutoffs_from``), so a
    caller who left ``calculator_parameters.ecutwfc`` unset needs the
    numeric value here too. ``assemble_dfpt_chain_inputs`` already
    guarantees the pseudo family recommends one whenever the input states
    none (:func:`koopmans.aiida.workflows.require_cutoffs_for_family`);
    deriving it the same way ``PwBaseWorkChain.get_builder_from_protocol``
    would keeps this route's fresh ground state on the same cutoff as the
    composed DFPT chain's own, which reaches its cutoff through that same
    protocol machinery rather than this literal.
    """
    system: dict[str, Any] = chain_inputs["overrides"]["scf"]["pw"]["parameters"]["SYSTEM"]
    if "ecutwfc" in system:
        return

    from koopmans.aiida.setup.pseudos import get_recommended_cutoffs

    ecutwfc, ecutrho = get_recommended_cutoffs(
        chain_inputs["pseudo_family"], chain_inputs["structure"]
    )
    system["ecutwfc"] = ecutwfc
    system["ecutrho"] = ecutrho
