"""Creation and validation of the Wannier projection blocks the routes consume."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from aiida_koopmans.projections import block_occupancy, get_wannier_indices
from aiida_koopmans.spin import SpinChannel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aiida import orm
    from aiida_koopmans.projections import (
        AutomaticProjectionBlock,
        ExplicitProjectionBlock,
        ProjectionBlock,
    )
    from wannier90_input.models.parameters import Projection


class _BandRange(NamedTuple):
    """Where one projection block sits in the nscf band manifold.

    ``start`` and ``end`` are 1-based and inclusive, and span the block's
    own ``num_wann`` Wannier bands. ``num_bands`` counts the bands
    wannier90 reads, exceeding ``num_wann`` only for a block that requires
    disentanglement.
    """

    start: int
    end: int
    num_wann: int
    num_bands: int


def _assign_band_ranges(
    structure: orm.StructureData,
    projection_blocks: list[list[Projection]],
    nbnd: int,
) -> list[_BandRange]:
    """Lay consecutive projection blocks out over the nscf band manifold.

    Each block takes the ``num_wann`` bands above the one before it, so the
    blocks tile the manifold from band 1 upwards in the order given. The
    bands left above the last block become its extra disentanglement
    bands: ``num_bands`` grows to cover them while the range still spans
    only the block's own Wannier bands. Whether those extras exist at all
    is decided by the user's ``num_wann`` against ``nbnd``; the ``dis_*``
    keywords refine the window wannier90 disentangles over, they never
    create it.
    """
    from aiida_koopmans.projections import projection_num_wann

    ranges: list[_BandRange] = []
    cursor = 0
    for block in projection_blocks:
        num_wann = sum(projection_num_wann(structure, p) for p in block)
        start, end = cursor + 1, cursor + num_wann
        if end > nbnd:
            raise ValueError(f"The projection blocks span {end} bands but nbnd = {nbnd}.")
        ranges.append(_BandRange(start=start, end=end, num_wann=num_wann, num_bands=num_wann))
        cursor = end

    if ranges and cursor < nbnd:
        last = ranges[-1]
        ranges[-1] = last._replace(num_bands=last.num_wann + (nbnd - cursor))

    return ranges


def _create_explicit_blocks(
    structure: orm.StructureData,
    projection_blocks: list[list[Projection]],
    nbnd: int,
    num_occ_bands: int,
    spin_channel: SpinChannel,
) -> list[ExplicitProjectionBlock]:
    """Turn a user's explicit projections into wannierization blocks.

    Every wannierization block a Koopmans calculation consumes lies wholly
    inside the occupied or the empty manifold. A block whose read window —
    its own bands together with any extra disentanglement bands — stays on one
    side of ``num_occ_bands`` therefore has its occupancy settled here: it
    is stamped ``filled`` and named after its manifold. A block spanning
    the boundary is provisional instead, left unstamped and named by
    list position; only a route that cuts blocks at the boundary at runtime can
    finalize it, and a route that cannot must reject it
    (:func:`_validate_blocks_separate_occ_and_emp`).

    The extra disentanglement bands belong to the read window, not to the
    Wannier-function indices the block takes, so an occupied block that
    disentangles against empty bands is provisional too: its Wannier
    functions are optimized out of those bands and are not the occupied
    manifold's.

    One provisional block makes the whole set provisional, so the stamps
    go on all together or not at all: what a set of occupancies buys
    downstream is a partition of the orbitals, and a partition missing a
    block accounts for nobody.

    A block that requires disentanglement stops excluding the bands above
    it, which it does read; its own Wannier bands stay the lowest
    ``num_wann`` of them.
    """
    from aiida_koopmans.projections import (
        ExplicitProjectionBlock,
        band_range_complement,
        projection_win_string,
    )
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    ranges = _assign_band_ranges(structure, projection_blocks, nbnd)
    suffix = f"_{spin_channel.value}" if spin_channel in (SpinChannel.UP, SpinChannel.DOWN) else ""
    counts = {"occ": 0, "emp": 0}
    occupancy: dict[int, bool] = {}
    blocks: list[ExplicitProjectionBlock] = []

    for index, (band_range, block) in enumerate(zip(ranges, projection_blocks, strict=True)):
        disentangle = band_range.num_bands > band_range.num_wann
        # The extra disentanglement bands always reach nbnd, so they are
        # where the block's read window ends; without them the window ends
        # at the block's own bands.
        window_end = nbnd if disentangle else band_range.end
        if window_end <= num_occ_bands:
            filling = "occ"
        elif band_range.start > num_occ_bands:
            filling = "emp"
        else:
            filling = None

        if filling is None:
            label = f"block_{index + 1}"
        else:
            occupancy[index] = filling == "occ"
            counts[filling] += 1
            label = f"{filling}{suffix}_{counts[filling]}"

        exclude = (
            list(range(1, band_range.start)) or None
            if disentangle
            else band_range_complement(band_range.start, band_range.end, nbnd)
        )
        blocks.append(
            ExplicitProjectionBlock(
                label=label,
                spin=spin_channel,
                num_wann=band_range.num_wann,
                num_bands=band_range.num_bands,
                exclude_bands=exclude,
                projection_type=WannierProjectionType.ANALYTIC,
                projections=[projection_win_string(p) for p in block],
            )
        )

    if len(occupancy) == len(blocks):
        for index, block_dict in enumerate(blocks):
            block_dict["filled"] = occupancy[index]
    return blocks


def _pseudo_is_fully_relativistic(kind: str, upf: orm.UpfData) -> bool:
    """Sniff the ``has_so`` flag of a pseudo's UPF header.

    Upstream's ``is_soc_pseudo`` trips over a UPF v2 header that omits
    ``has_so`` (a bare ``TypeError``); real generators always write the
    flag, so convert that case into an error naming the pseudo. Old
    attribute-less v1 headers parse as scalar-relativistic upstream.
    """
    from aiida_wannier90_workflows.utils.pseudo.upf import get_upf_content, is_soc_pseudo

    try:
        return bool(is_soc_pseudo(get_upf_content(upf)))
    except TypeError as exc:
        raise ValueError(
            f"The pseudopotential for {kind} does not declare `has_so` in its UPF header, "
            "so whether it is fully relativistic cannot be determined (scalar-relativistic "
            'UPF files normally carry `has_so="F"`).'
        ) from exc


def _create_automatic_blocks(
    structure: orm.StructureData,
    pseudos: dict[str, orm.UpfData],
    external_projectors: dict[str, Any] | None,
    nbnd: int | None,
    num_occ_bands: int,
) -> tuple[list[AutomaticProjectionBlock], int]:
    """Derive the whole-manifold block taken when no projections are given.

    The manifold becomes a single automatic block whose ``num_wann`` is a
    projector count — the width of the amn matrix pw2wannier90 writes — and
    the runtime band-group detection decides how it splits. The projectors
    come from the external ``.dat`` tables when ``external_projectors`` is
    given (pw2wannier90 ``atom_proj_ext``), one band per 2l+1 multiplet
    member, and from the pseudopotentials otherwise (``atom_proj``).
    Returns the single-block list and the band count the nscf must cover.

    The block requires no disentanglement: the detected groups cover
    only the Wannierized manifold, so a block with bands above it cannot be
    split. It carries no ``filled`` stamp either — it is the provisional
    block par excellence, existing only to be cut into the groups the
    runtime detection finds.
    """
    from aiida_koopmans.projections import AutomaticProjectionBlock
    from aiida_koopmans.spin import SpinChannel
    from aiida_wannier90_workflows.common.types import WannierProjectionType
    from aiida_wannier90_workflows.utils.pseudo import (
        get_number_of_projections,
        get_number_of_projections_ext,
    )

    if external_projectors is not None:
        source = "the external projector files"
        projection_type = WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL
        num_wann = get_number_of_projections_ext(
            structure=structure,
            external_projectors=external_projectors,
            spin_non_collinear=False,
            spin_orbit_coupling=False,
        )
    else:
        source = "the pseudopotentials"
        projection_type = WannierProjectionType.ATOMIC_PROJECTORS_QE
        fully_relativistic = sorted(
            kind for kind, upf in pseudos.items() if _pseudo_is_fully_relativistic(kind, upf)
        )
        if fully_relativistic:
            raise NotImplementedError(
                f"The pseudopotentials for {', '.join(fully_relativistic)} are fully "
                "relativistic; automatic projections support scalar-relativistic "
                "pseudopotentials only (this route runs spin='none'). Provide explicit "
                "projections in `calculator_parameters.w90.projections` or use a "
                "scalar-relativistic family."
            )
        # Scalar-relativistic guaranteed by the guard above, so the projector
        # count is exact with the SOC flag pinned off.
        num_wann = get_number_of_projections(
            structure=structure,
            pseudos=pseudos,
            spin_non_collinear=False,
            spin_orbit_coupling=False,
        )

    if num_wann < num_occ_bands:
        raise ValueError(
            f"{source[0].upper()}{source[1:]} provide {num_wann} atomic projectors but the "
            f"system has {num_occ_bands} occupied bands, so automatic projections cannot "
            "span the occupied manifold. Provide explicit projections in "
            "`calculator_parameters.w90.projections`."
        )
    if nbnd is not None and nbnd < num_wann:
        raise ValueError(
            f"nbnd = {nbnd} is smaller than the {num_wann} atomic projectors of "
            f"{source}; automatic projections need one band per projector."
        )
    if nbnd is not None and nbnd > num_wann:
        raise NotImplementedError(
            f"nbnd = {nbnd} exceeds the {num_wann} atomic projectors of "
            f"{source}, which would disentangle the automatic block; splitting a "
            "disentangled block is not supported. Drop nbnd or provide explicit "
            "projections in `calculator_parameters.w90.projections`."
        )
    block = AutomaticProjectionBlock(
        label="block_1",
        spin=SpinChannel.NONE,
        num_wann=num_wann,
        num_bands=num_wann,
        exclude_bands=None,
        projection_type=projection_type,
    )
    return [block], num_wann


def _validate_blocks_separate_occ_and_emp(blocks: Sequence[ProjectionBlock], nocc: int) -> None:
    """Reject a block that spans both sides of the occupied/empty boundary.

    Every Koopmans calculation downstream wants Wannier functions that
    belong to one manifold: a block spanning both has no occupancy to
    state, and the orbitals it produces answer to neither. What counts as
    spanning is the plugin's rule and is asked of the plugin
    (``block_occupancy``): a block reads both manifolds either through its
    own bands or through extra disentanglement bands that reach across the
    boundary, and neither is visible in its Wannier-function indices alone. The
    plugin asks the same question again when it receives the blocks; this
    is the build-time answer, so the user hears it before anything is
    submitted.

    A route that derives its blocks from user projections alone has no
    block to check when there are none, so the absence of projections is
    rejected here too.
    """
    if not blocks:
        raise ValueError(
            "Wannier-function initialisation requires explicit projections in "
            "``calculator_parameters.w90.projections``."
        )

    for block in blocks:
        try:
            block_occupancy(block)
        except ValueError as exc:
            wannier_indices = get_wannier_indices(block)
            start, end = wannier_indices[0], wannier_indices[-1]
            raise ValueError(
                f"The projection block '{block['label']}' (bands {start}-{end}) straddles "
                f"the occupied/empty boundary at band {nocc}: its own bands cross it, or "
                "the extra bands it reads for disentanglement do. Its Wannier functions seed "
                "the occupied or the empty manifold of the supercell kcp.x run, so they "
                "must come from one of them. Split "
                "``calculator_parameters.w90.projections`` at the boundary, add "
                "projections for the empty manifold, or lower "
                f"``calculator_parameters.pw.system.nbnd`` to {nocc}."
            ) from exc


def _validate_blocks_cover_all_occ_bands(blocks: Sequence[ProjectionBlock], nocc: int) -> None:
    """Reject occupied blocks that leave part of the occupied manifold unseeded.

    The merged ``evc_occupied`` file seeds the complete occupied manifold
    of the supercell kcp.x run, so the occupied blocks must span every
    occupied band.

    Runs after ``validate_projection_block_sequence`` and
    :func:`_validate_blocks_separate_occ_and_emp`, whose rules this one
    assumes: the sequence rules make a block's Wannier indices band
    indices, and with every block on one side of the boundary a block's
    own bands place it in a manifold, so counting the Wannier functions
    sitting in occupied bands ensures every occupied band is covered.
    """
    covered_occ = sum(b["num_wann"] for b in blocks if get_wannier_indices(b)[-1] <= nocc)
    if covered_occ != nocc:
        raise ValueError(
            f"The occupied projection blocks span {covered_occ} Wannier functions but "
            f"the system has {nocc} occupied bands per primitive cell; every occupied "
            "band must be covered for the Wannier-seeded kcp.x initialisation."
        )
