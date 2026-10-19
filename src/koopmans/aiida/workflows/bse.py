"""The Bethe-Salpeter (yambo BSE, seeded by Koopmans eigenvalues) route."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiida_quantumespresso.common.types import SpinType

from koopmans.aiida.workflows import load_codes, name_run, require_configured_codes
from koopmans.aiida.workflows.dfpt import (
    DfptChainInputs,
    add_smooth_interpolation_inputs,
    assemble_dfpt_chain_inputs,
)

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

    ``spin`` must be ``'none'``: the composed workflow reads a single DFPT
    channel. The route runs the DFPT chain whether ``formalism`` is ``'dfpt'`` or unset.

    A ``kpoints.path`` and ``kpoints.overrides.wannierize_smooth`` reach the
    DFPT chain as they do on the DFPT singlepoint route, so the run also
    emits the Koopmans band structure along the path. The yambo BSE step
    reads only the grid eigenvalues and is unaffected by either.

    Left unrefused: ``calculator_parameters.ecutwfc``, if unset, is derived
    from the pseudopotential family's own recommendation (the same call
    ``PwBaseWorkChain.get_builder_from_protocol`` makes), so both this
    route's fresh yambo ground state and the composed DFPT chain's own run
    on the identical cutoff without the user typing it. ``workflow.eps_inf``,
    ``workflow.gb_correction``, ``calculator_parameters.kcw`` overrides, and
    workflow-level orbital grouping (``group_orbitals_by`` /
    ``group_orbitals_tol``) all pass straight into the composed DFPT chain,
    exactly as the plain DFPT singlepoint route passes them
    (:func:`koopmans.aiida.workflows.dfpt.build_singlepoint_dfpt_workgraph`).

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
        NotImplementedError: If the input asks for a `spin` other than `none`.
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

    if workflow.spin != SpinType.NONE:
        raise NotImplementedError(
            "the `bse` task only supports `spin: none`: aiida-koopmans's "
            "SinglepointBetheSalpeterWorkflow reads a single DFPT channel, and a "
            f"collinear or spinor run has none; got spin={workflow.spin.value!r}. Run "
            "`task: singlepoint` and a standalone BSE workflow separately for those."
        )
    chain_inputs, manifold_band_counts = assemble_dfpt_chain_inputs(koopmans_input)
    add_smooth_interpolation_inputs(chain_inputs, koopmans_input)
    _ensure_explicit_pw_cutoffs(chain_inputs)

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
            bands_kpoints=chain_inputs["bands_kpoints"],
            wannierize_bands_kpoints=chain_inputs["wannierize_bands_kpoints"],
            wannierize_smooth_kpoints=chain_inputs.get("wannierize_smooth_kpoints"),
            wannierize_smooth_mp_grid=chain_inputs.get("wannierize_smooth_mp_grid"),
            wannierize_smooth_bands_kpoints=chain_inputs.get("wannierize_smooth_bands_kpoints"),
            pseudo_family=chain_inputs["pseudo_family"],
            overrides=chain_inputs["overrides"],
            # aiida-koopmans#134: no ak2 kcw.x `ham` parser emits `pki_eigenvalues_on_grid`
            # yet, so `pki` has no producer; the `eigenvalues` flavour knob returns once it
            # does.
            eigenvalues="ki",
            eps_inf=chain_inputs["eps_inf"],
            eps_kpoints=chain_inputs["eps_kpoints"],
            l_vcut=chain_inputs["l_vcut"],
            group_orbitals_tol=chain_inputs["group_orbitals_tol"],
            kcw_overrides=chain_inputs["kcw_overrides"],
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
