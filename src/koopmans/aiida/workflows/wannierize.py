"""The Wannierize route: whole-manifold or block-by-block."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from aiida_koopmans.spin import SpinChannel
from aiida_quantumespresso.common.types import SpinType

from koopmans.aiida.conversion import atoms_input_to_structure, input_to_pw_parameters
from koopmans.aiida.workflows import prepare_common_inputs
from koopmans.aiida.workflows.blocks import (
    create_automatic_blocks,
    create_explicit_blocks,
)
from koopmans.aiida.workflows.projectors import load_external_projectors

if TYPE_CHECKING:
    from aiida import orm
    from aiida_koopmans.workgraphs import Codes
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


def build_wannierize_workgraph(
    koopmans_input: KoopmansInput,
    codes: Codes,
) -> WorkGraph:
    """Build a workgraph for Wannierization.

    Args:
        koopmans_input: The parsed koopmans input.
        codes: Dictionary of loaded codes.

    Returns:
        A WorkGraph wrapping a single ``Wannier90WorkChain`` over the whole
        manifold, or :func:`_build_wannierize_blocks_workgraph` when explicit
        projections or ``block_wannierization_threshold`` ask for one
        Wannierization per block.
    """
    from aiida_koopmans.workgraphs.wannier90 import Wannierize
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    if koopmans_input.workflow.spin != SpinType.NONE:
        raise NotImplementedError(
            "Wannierization currently supports spin='none' only: no route sets "
            "`nspin`, and the per-block group detection and split are "
            "single-channel."
        )

    if koopmans_input.workflow.block_wannierization_threshold is not None:
        return _build_wannierize_blocks_workgraph(koopmans_input, codes)

    _validate_projection_sources(koopmans_input)
    if _keywords_setting_projections(koopmans_input):
        return _build_wannierize_blocks_workgraph(koopmans_input, codes)
    if not koopmans_input.workflow.auto_projections:
        raise ValueError(_NO_PROJECTIONS_PROVIDED_MESSAGE)

    structure, pseudo_family, overrides = prepare_common_inputs(koopmans_input, ["scf", "nscf"])

    # The automatically derived projections are the pseudopotentials' atomic
    # orbitals (upstream's ATOMIC_PROJECTORS_QE mechanism) unless external
    # projector files supply the projector functions instead.
    pw2w_params = koopmans_input.calculator_parameters.pw2wannier90
    extra_kwargs: dict[str, Any] = {}
    if pw2w_params.atom_proj_ext:
        external_projectors, projector_path = load_external_projectors(
            structure, pw2w_params.atom_proj_dir
        )
        extra_kwargs["projection_type"] = WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL
        extra_kwargs["external_projectors_path"] = projector_path
        extra_kwargs["external_projectors"] = external_projectors

    return Wannierize.build(
        codes=codes,
        structure=structure,
        overrides=overrides,
        pseudo_family=pseudo_family,
        print_summary=False,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        **extra_kwargs,
    )


def _build_wannierize_blocks_workgraph(
    koopmans_input: KoopmansInput,
    codes: Codes,
) -> WorkGraph:
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

    Current scope: ``spin = 'none'``.
    """
    from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlocks
    from aiida_wannier90_workflows.utils.kpoints import get_explicit_kpoints

    from koopmans.aiida.conversion import (
        get_pseudos_from_family,
        kpoints_input_to_kpoints_mesh,
        kpoints_input_to_kpoints_path,
    )
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
        external_projectors = None
        if calc_params.pw2wannier90.atom_proj_ext:
            external_projectors, projector_path = load_external_projectors(
                structure, calc_params.pw2wannier90.atom_proj_dir
            )
            external_kwargs = {
                "external_projectors_path": projector_path,
                "external_projectors": external_projectors,
            }
        blocks, nbnd = create_automatic_blocks(
            structure, pseudos, external_projectors, nbnd, num_occ_bands
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
    wannier_overrides: WannierizeOverrides = {
        "scf": {"pseudo_family": pseudo_family, "pw": {"parameters": scf_parameters}},
        "nscf": {"pseudo_family": pseudo_family, "pw": {"parameters": nscf_parameters}},
    }

    # User wannier90 keywords (disentanglement windows, iteration counts, ...)
    # feed every per-block wannierisation; flat by design (see
    # ``WannierizeOverrides``).
    w90_user = calc_params.wannier90.model_dump(
        exclude_unset=True, exclude={"projections", "up", "down"}
    )
    if w90_user:
        wannier_overrides["wannier90"] = w90_user

    # wannier90 / pw2wannier90 need eigenstates on the full explicit k-list
    # (wannier90 kmesh.pl ordering, no symmetry reduction) and cannot
    # re-derive the Monkhorst-Pack dimensions from it, so expand the mesh
    # here and carry the grid separately. The scf takes the mesh itself and
    # may reduce it by symmetry.
    kmesh = kpoints_input_to_kpoints_mesh(koopmans_input.kpoints)
    mp_grid = [int(x) for x in kmesh.get_kpoints_mesh()[0]]  # type: ignore[no-untyped-call]

    # Without a threshold the graph splits nothing, and WannierizeBlocks
    # rejects the split-only inputs rather than ignore them.
    split_kwargs: dict[str, Any] = {}
    if threshold is not None:
        split_kwargs = {
            "bands_kpoints": kpoints_input_to_kpoints_path(koopmans_input.kpoints, structure),
            "num_occ_bands": num_occ_bands,
            "split_threshold": float(threshold),
        }

    return WannierizeBlocks.build(
        codes=codes,
        structure=structure,
        blocks=blocks,
        kpoints=get_explicit_kpoints(kmesh),
        mp_grid=mp_grid,
        scf_kpoints=kmesh,
        **split_kwargs,
        pseudo_family=pseudo_family,
        overrides=wannier_overrides,
        parallelization=koopmans_input.parallelization.as_mapping() or None,
        **external_kwargs,
    )


def _num_wann_total(structure: orm.StructureData, projection_blocks: list[list[Projection]]) -> int:
    """Total Wannier-function count of a set of user projection blocks."""
    from aiida_koopmans.projections import projection_num_wann

    return sum(projection_num_wann(structure, p) for block in projection_blocks for p in block)
