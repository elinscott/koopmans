"""The Wannierize route: whole-manifold or block-by-block."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from aiida_koopmans.spin import SpinChannel
from aiida_quantumespresso.common.types import SpinType

from koopmans.aiida.conversion import atoms_input_to_structure, input_to_pw_parameters
from koopmans.aiida.workflows import (
    configured_projwfc,
    load_codes,
    pin_step_kpoints,
    prepare_common_inputs,
    require_cutoffs_for_family,
)
from koopmans.aiida.workflows.blocks import (
    create_automatic_blocks,
    create_explicit_blocks,
)
from koopmans.aiida.workflows.projectors import load_external_projectors

if TYPE_CHECKING:
    from aiida import orm
    from aiida_koopmans.workgraphs.block_wannierize import WannierizeOverrides
    from aiida_workgraph import WorkGraph
    from wannier90_input.models.parameters import Projection

    from koopmans.input_file import KoopmansInput


#: Raised wherever a Wannierization is asked for and nothing says what to
#: Wannierize. Both routes that need projections reach this state, so both
#: say the same thing.
_NO_PROJECTIONS_PROVIDED_MESSAGE = (
    "Nothing defines the Wannier projections. Either (a) set "
    "`workflow.auto_projections` to `True` to derive them from the "
    "pseudopotentials, or from external projector files by also pointing "
    "`pw2wannier90.atom_proj_ext` at them; or (b) provide explicit projections "
    "in `calculator_parameters.w90.projections`."
)


def _keywords_setting_projections(
    koopmans_input: KoopmansInput, *, channels_only: bool = False
) -> list[str]:
    """List the keywords the input uses to set explicit Wannier projections.

    Covers the two spin-channel projection blocks and, unless
    ``channels_only``, the top-level one. Empty when the input sets none of
    them, so callers can both test for explicit projections and name them.
    """
    w90 = koopmans_input.calculator_parameters.wannier90
    keywords = (
        ["`calculator_parameters.w90.projections`"] if w90.projections and not channels_only else []
    )
    keywords += [
        f"`calculator_parameters.w90.{name}.projections`"
        for name in ("up", "down")
        if getattr(w90, name) is not None and getattr(w90, name).projections
    ]
    return keywords


def _and_list(items: list[str]) -> str:
    """Join items into an ``a``, ``b`` and ``c`` phrase."""
    if len(items) < 2:
        return "".join(items)
    return " and ".join([", ".join(items[:-1]), items[-1]])


def _validate_projection_sources(koopmans_input: KoopmansInput) -> None:
    """Validate how the input asks for the Wannier projections to be defined.

    Three inputs can speak for the projections: explicit blocks in
    ``calculator_parameters.w90(.up/.down).projections``, the automatic
    derivation requested by ``workflow.auto_projections``, and the external
    projector files of ``pw2wannier90.atom_proj_ext``. Explicit blocks
    define the full set themselves, so they combine with neither of the
    others; the external files only choose where the projector functions
    come from, not whether the projections are derived at all, so they
    require the flag.
    """
    external = koopmans_input.calculator_parameters.pw2wannier90.atom_proj_ext
    automatic = koopmans_input.workflow.auto_projections
    explicit = _keywords_setting_projections(koopmans_input)
    if explicit and automatic:
        raise ValueError(
            f"`workflow.auto_projections` and explicit projections (in {_and_list(explicit)}) were "
            "both given; the automatic derivation and the explicit blocks each define "
            "the full set of projections. Drop one of the two."
        )
    if explicit and external:
        raise ValueError(
            f"Explicit projections ({explicit}) and `pw2wannier90.atom_proj_ext` were "
            "both given; the explicit projections define every block, so the external "
            "projectors would be silently ignored. Drop one of the two."
        )
    if external and not automatic:
        raise ValueError(
            "`pw2wannier90.atom_proj_ext` was given without `workflow.auto_projections`; "
            "external projector files choose where the projector functions come from, "
            "but they do not by themselves ask for the projections to be derived "
            "automatically. Set `workflow.auto_projections` as well."
        )


def _kpoint_sampling(
    koopmans_input: KoopmansInput,
    overrides: dict[str, Any],
) -> tuple[orm.KpointsData | None, orm.KpointsData, list[int]]:
    """Return the scf mesh, the explicit nscf/wannier90 k-list and its dimensions.

    wannier90 and pw2wannier90 need eigenstates on the full explicit k-list
    (wannier90 ``kmesh.pl`` ordering, no symmetry reduction) and cannot
    re-derive the Monkhorst-Pack dimensions from it, so the mesh is expanded
    here and the grid carried alongside. The scf takes the mesh itself and
    may reduce it by symmetry; ``None`` where its entry leaves the mesh to a
    ``grid_spacing`` written into ``overrides["scf"]``.
    """
    from aiida_wannier90_workflows.utils.kpoints import get_explicit_kpoints

    from koopmans.aiida.conversion import step_kpoints_mesh

    scf_kpoints = pin_step_kpoints(overrides, "scf", koopmans_input)
    mesh = step_kpoints_mesh(koopmans_input.kpoints, "nscf")
    mp_grid = [int(x) for x in mesh.get_kpoints_mesh()[0]]  # type: ignore[no-untyped-call]
    return scf_kpoints, get_explicit_kpoints(mesh), mp_grid


def _interpolation_path(
    koopmans_input: KoopmansInput, structure: orm.StructureData
) -> orm.KpointsData | None:
    """Return the input's k-path as a labelled explicit k-list, or ``None``.

    ``None`` when the input states no ``kpoints.path``, and for a gamma-only
    input, whose fixed ``path`` names the zone centre alone and so defines no
    segment to interpolate along.
    """
    from koopmans.aiida.conversion import kpoints_input_to_kpoints_path

    if koopmans_input.kpoints.gamma_only or koopmans_input.kpoints.path is None:
        return None
    return kpoints_input_to_kpoints_path(koopmans_input.kpoints, structure)


def _external_projector_kwargs(
    koopmans_input: KoopmansInput, structure: orm.StructureData
) -> dict[str, Any]:
    """Return the external-projector inputs an automatic Wannierization needs.

    Empty unless ``pw2wannier90.atom_proj_ext`` asks for the projector
    functions to come from ``pw2wannier90.atom_proj_dir`` rather than from
    the pseudopotentials.
    """
    pw2w_params = koopmans_input.calculator_parameters.pw2wannier90
    if not pw2w_params.atom_proj_ext:
        return {}
    external_projectors, projector_path = load_external_projectors(
        structure, pw2w_params.atom_proj_dir
    )
    return {
        "external_projectors_path": projector_path,
        "external_projectors": external_projectors,
    }


def build_wannierize_workgraph(koopmans_input: KoopmansInput) -> WorkGraph:
    """Build a workgraph for Wannierization.

    Two routes, both sampling the Brillouin zone on ``kpoints.grid``:

    * ``workflow.auto_projections`` alone Wannierizes the whole manifold in
      one ``Wannier90WorkChain``, which reads twice as many bands as it
      Wannierizes and disentangles them against the pseudo-atomic
      projections.
    * Explicit projections or ``block_wannierization_threshold`` route to
      :func:`_build_wannierize_blocks_workgraph`, one Wannierization per
      block off a shared scf + nscf. Its automatic block reads exactly its
      projector count, so nothing disentangles there — a split needs each
      group's gauge to come from the parent's alone.

    On both routes a ``kpoints.path`` in the input reaches wannier90 as its
    bands path, so each Wannierization also emits ``interpolated_bands`` —
    its band structure Wannier-interpolated along that path. A pw.x bands
    run along the same path supplies the explicit eigenvalues the
    interpolation is judged against. The dispatcher passes a configured
    projwfc code along (:func:`~koopmans.aiida.workflows.configured_projwfc`);
    whether a projected DOS runs from the bands run — and the warning when the
    pseudopotentials' missing ``PP_PSWFC`` wavefunctions make it
    impossible — is the graphs' decision. The path always travels as an
    explicit labelled k-list: the graphs run the pw.x quality check only
    for that form — a symbolic ``kpoint_path`` leaves wannier90 to
    discretize the path itself, with no pw.x eigenvalues to compare
    against.

    Args:
        koopmans_input: The parsed koopmans input.

    Returns:
        The assembled WorkGraph.
    """
    from aiida_koopmans.workgraphs.wannier90 import Wannierize, WannierizeCodes
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    if koopmans_input.workflow.spin != SpinType.NONE:
        raise NotImplementedError(
            "Wannierization currently supports spin='none' only: no route sets "
            "`nspin`, and the per-block group detection and split are "
            "single-channel."
        )

    if koopmans_input.workflow.block_wannierization_threshold is not None:
        return _build_wannierize_blocks_workgraph(koopmans_input)

    _validate_projection_sources(koopmans_input)
    if _keywords_setting_projections(koopmans_input):
        return _build_wannierize_blocks_workgraph(koopmans_input)
    if not koopmans_input.workflow.auto_projections:
        raise ValueError(_NO_PROJECTIONS_PROVIDED_MESSAGE)

    structure, pseudo_family, overrides = prepare_common_inputs(koopmans_input, ["scf", "nscf"])

    # The automatically derived projections are the pseudopotentials' atomic
    # orbitals (upstream's ATOMIC_PROJECTORS_QE mechanism) unless external
    # projector files supply the projector functions instead.
    extra_kwargs: dict[str, Any] = _external_projector_kwargs(koopmans_input, structure)
    if extra_kwargs:
        extra_kwargs["projection_type"] = WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL

    scf_kpoints, kpoints, mp_grid = _kpoint_sampling(koopmans_input, overrides)

    bands_kpoints = _interpolation_path(koopmans_input, structure)

    # WannierizeCodes's one NotRequired member is projwfc. The upstream
    # builder wires it only for SCDM projections and frozen_type
    # energy_auto, which koopmans never asks for; here it rides along for
    # the projected DOS accompanying the quality-check bands run.
    codes = load_codes(WannierizeCodes)
    projwfc = configured_projwfc()
    if projwfc is not None:
        codes["projwfc"] = projwfc

    return Wannierize.build(
        codes=codes,
        structure=structure,
        overrides=overrides,
        pseudo_family=pseudo_family,
        print_summary=False,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        scf_kpoints=scf_kpoints,
        kpoints=kpoints,
        mp_grid=mp_grid,
        bands_kpoints=bands_kpoints,
        **extra_kwargs,
    )


def _build_wannierize_blocks_workgraph(koopmans_input: KoopmansInput) -> WorkGraph:
    """Build the Wannierization workgraph that Wannierizes block by block.

    One scf + nscf feeds a separate Wannierization per projection block.
    With explicit projections in ``calculator_parameters.w90.projections``
    each user block becomes a wannierization block. With
    ``workflow.auto_projections`` instead, a single atomic-projector block
    spans the whole manifold; its projectors come from the external
    projector directory ``pw2wannier90.atom_proj_dir`` when
    ``pw2wannier90.atom_proj_ext`` is set and from the pseudopotentials
    otherwise (:func:`create_automatic_blocks`).

    Setting ``block_wannierization_threshold`` adds the automated splitting:
    a pw.x bands run along the k-path feeds a runtime band-group detection
    (splitting at every gap wider than the threshold in eV and at the
    occupied/empty boundary), and each block whose bands fall into several
    groups is Wannierized once, split with Wannier.jl parallel transport,
    re-Wannierized group by group and its products merged back together. An
    automatic-projector block always splits this way, since its band groups
    exist only at runtime.

    A ``kpoints.path`` in the input also reaches every per-block wannier90
    run as its bands path, so each ``blocks`` entry emits
    ``interpolated_bands``.

    Current scope: ``spin = 'none'``.
    """
    from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlocks, WannierizeBlocksCodes

    from koopmans.aiida.conversion import get_pseudos_from_family
    from koopmans.aiida.setup.pseudos import ensure_pseudo_family_installed

    workflow = koopmans_input.workflow
    calc_params = koopmans_input.calculator_parameters

    threshold = workflow.block_wannierization_threshold
    _validate_projection_sources(koopmans_input)
    channels = _keywords_setting_projections(koopmans_input, channels_only=True)
    if channels:
        raise NotImplementedError(
            f"Explicit projections in {_and_list(channels)} are not wired into the block-by-block "
            "wannierize route, which is single-channel: give them in "
            "`calculator_parameters.w90.projections` instead."
        )
    projections = calc_params.wannier90.projections
    if threshold is not None and koopmans_input.kpoints.path is None:
        raise ValueError(
            "block_wannierization_threshold needs a k-point path: the band-group "
            "detection reads the eigenvalues of a bands run along it."
        )

    structure = atoms_input_to_structure(koopmans_input.atoms)
    pseudo_family = workflow.pseudo_library
    ensure_pseudo_family_installed(pseudo_family)

    pseudos = get_pseudos_from_family(pseudo_family, structure)
    nelec = round(sum(pseudos[site.kind_name].z_valence for site in structure.sites))
    if nelec % 2:
        raise NotImplementedError(
            f"Odd electron count ({nelec}) requires spin='collinear', which the "
            "occupied/empty boundary the blocks are built around does not support yet."
        )
    num_occ_bands = nelec // 2

    nbnd = calc_params.nbnd if calc_params.nbnd is not None else calc_params.pw.system.nbnd
    nbnd = int(nbnd) if nbnd is not None else None
    external_kwargs: dict[str, Any] = {}
    if projections:
        if nbnd is None:
            nbnd = _num_wann_total(structure, projections)
        blocks = create_explicit_blocks(
            structure, projections, nbnd, num_occ_bands, SpinChannel.NONE
        )
    elif workflow.auto_projections:
        external_kwargs = _external_projector_kwargs(koopmans_input, structure)
        blocks, nbnd = create_automatic_blocks(
            structure,
            pseudos,
            external_kwargs.get("external_projectors"),
            nbnd,
            num_occ_bands,
        )
    else:
        raise ValueError(_NO_PROJECTIONS_PROVIDED_MESSAGE)

    # The scf needs only the occupied bands, so nbnd is dropped from its
    # override; the nscf — and the bands run seeded from its overrides —
    # must cover every Wannierised band.
    parameters = input_to_pw_parameters(koopmans_input)
    scf_parameters = copy.deepcopy(parameters)
    scf_parameters.get("SYSTEM", {}).pop("nbnd", None)
    nscf_parameters = copy.deepcopy(parameters)
    nscf_parameters.setdefault("SYSTEM", {})["nbnd"] = nbnd
    # This route assembles its own scf/nscf overrides instead of calling
    # ``prepare_common_inputs``, so the family checks are its own too.
    from koopmans.aiida.setup.pseudos import require_norm_conserving_family

    require_norm_conserving_family(pseudo_family, structure)
    require_cutoffs_for_family(pseudo_family, parameters)
    wannier_overrides: WannierizeOverrides = {
        "scf": {
            "pseudo_family": pseudo_family,
            "pw": {"parameters": scf_parameters},
        },
        "nscf": {
            "pseudo_family": pseudo_family,
            "pw": {"parameters": nscf_parameters},
        },
    }

    # User wannier90 keywords (disentanglement windows, iteration counts, ...)
    # feed every per-block wannierisation; flat by design (see
    # ``WannierizeOverrides``).
    w90_user = calc_params.wannier90.model_dump(
        exclude_unset=True, exclude={"projections", "up", "down"}
    )
    if w90_user:
        wannier_overrides["wannier90"] = w90_user

    scf_kpoints, kpoints, mp_grid = _kpoint_sampling(koopmans_input, wannier_overrides)

    # One node serves both uses of the input's k-path: the split-mode band
    # detection samples it with pw.x, and every wannier90 run interpolates
    # its band structure along it.
    interpolation_kpoints = _interpolation_path(koopmans_input, structure)

    # ``WannierizeBlocksCodes``'s NotRequired members are turned on one by
    # one: the threshold requires wannierjl (the julia binary registered
    # via aiida_wannierjl.helpers.get_wannierjl_code) for the split
    # machinery, while projwfc merely rides along when configured — the
    # graph decides whether the projected DOS runs.
    codes = load_codes(
        WannierizeBlocksCodes,
        require=("wannierjl",) if threshold is not None else (),
    )
    projwfc = configured_projwfc()
    if projwfc is not None:
        codes["projwfc"] = projwfc

    # Without a threshold the graph splits nothing, and WannierizeBlocks
    # rejects the split-only inputs rather than ignore them.
    split_kwargs: dict[str, Any] = {}
    if threshold is not None:
        split_kwargs = {
            "bands_kpoints": interpolation_kpoints,
            "num_occ_bands": num_occ_bands,
            "split_threshold": float(threshold),
        }

    return WannierizeBlocks.build(
        codes=codes,
        structure=structure,
        blocks=blocks,
        kpoints=kpoints,
        mp_grid=mp_grid,
        scf_kpoints=scf_kpoints,
        **split_kwargs,
        interpolation_kpoints=interpolation_kpoints,
        pseudo_family=pseudo_family,
        overrides=wannier_overrides,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        **external_kwargs,
    )


def _num_wann_total(structure: orm.StructureData, projection_blocks: list[list[Projection]]) -> int:
    """Total Wannier-function count of a set of user projection blocks."""
    from aiida_koopmans.projections import projection_num_wann

    return sum(projection_num_wann(structure, p) for block in projection_blocks for p in block)
