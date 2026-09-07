"""Per-code parallelization settings.

The top-level ``parallelization`` block maps each code (``pw``, ``kcp``, …)
to a small config of MPI-rank count (``ntasks``), k-point-pool count
(``npool``), a pencil-decomposition switch (``pd``), a per-rank OpenMP/BLAS
thread count (``omp``), and a wallclock override (``walltime``). ``ntasks``
becomes the scheduler's ``tot_num_mpiprocs``; ``npool`` becomes ``-npool``
and ``pd`` becomes ``-pd true`` on the QE command line; ``omp`` sets the
``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS`` per
rank, defaulting to the localhost computer's pin of one thread; ``walltime``
overrides the top-level ``computer.walltime`` default for one code, for
every code. See :func:`resolve_effective_walltime` for that precedence, and
:func:`koopmans.aiida.conversion.code_parallelization` for the translation
into AiiDA ``metadata.options`` / ``settings.cmdline``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Self, cast

from aiida_koopmans.parallelization import CODE_NAMES, ParallelizationDict
from pydantic import Field, model_validator

from koopmans.base import BaseModel
from koopmans.input_file._utils import Walltime

if TYPE_CHECKING:
    from koopmans.input_file.computer import ComputerInput

# Every code the parallelization block recognises. Sourced from the single
# ``aiida_koopmans.parallelization`` vocabulary (``CodeName``) rather than duplicated here.
ALL_CODES: tuple[str, ...] = CODE_NAMES

# Codes that accept ``-npool`` (k-point pools) and ``-pd`` (pencil
# decomposition) on their command line. Source-verified against Quantum
# ESPRESSO (``Modules/command_line_options.f90`` parses both flags globally for
# the modern binaries; the koopmans-kcp fork behind kcp.x / wann2kcp.x reads no
# CLI flags, and wannier90 has no pool/pd concept). ``kcw`` accepts pools for
# its wann2kc / screen steps but not its ham step — a per-step distinction the
# workgraph makes; at the schema level ``kcw`` counts as pool-supporting.
POOL_SUPPORTING_CODES: frozenset[str] = frozenset({"pw", "ph", "projwfc", "pw2wannier90", "kcw"})
PD_SUPPORTING_CODES: frozenset[str] = frozenset({"pw", "ph", "projwfc", "pw2wannier90", "kcw"})


__all__ = ["CodeParallelization", "ParallelizationInput", "resolve_effective_walltime"]


# NOTE: keep this Pydantic model (and the per-code fields it validates) in
# sync with the ``CodeParallelization`` TypedDict in ``aiida_koopmans.parallelization``
# — the TypedDict is the runtime shape the graphs consume; this model is the
# user-facing validated view of the same data.
class CodeParallelization(BaseModel):
    """Parallelization settings for a single code."""

    ntasks: int | None = Field(
        default=None,
        ge=1,
        description="number of MPI ranks to run the code with (becomes the scheduler's "
        "``tot_num_mpiprocs``)",
    )
    npool: int | None = Field(
        default=None,
        ge=1,
        description="number of k-point pools to distribute the calculation over "
        "(becomes ``-npool`` on the command line; should be commensurate with the "
        "k-point grid). Only valid for pw, ph, projwfc, pw2wannier90, and kcw.",
    )
    pd: bool | None = Field(
        default=None,
        description="use pencil decomposition of the FFT grid (becomes ``-pd true`` on "
        "the command line). Only valid for pw, ph, projwfc, pw2wannier90, and kcw.",
    )
    omp: int | None = Field(
        default=None,
        ge=1,
        description="number of OpenMP / BLAS threads to run each MPI rank with (sets "
        "``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS`` per rank). "
        "Valid for every code; the default is the localhost computer's pin of one thread "
        "per rank, which stops the threaded BLAS builds oversubscribing the allocation.",
    )
    walltime: Walltime = Field(
        default=None,
        description="wallclock limit for this code's calculations, overriding the "
        "top-level ``computer.walltime`` default (``2h``, ``90m``, ``1d12h``, "
        "``HH:MM:SS``, or any pydantic-native duration). Valid for every code.",
    )


class ParallelizationInput(BaseModel):
    """Per-code parallelization settings.

    A mapping of code name to :class:`CodeParallelization`. Only the codes
    listed here are recognised; any other key is rejected. Codes left unset
    inherit the QE/AiiDA defaults (a single MPI rank, no pools).
    """

    pw: CodeParallelization | None = None
    kcp: CodeParallelization | None = None
    kcw: CodeParallelization | None = None
    ph: CodeParallelization | None = None
    projwfc: CodeParallelization | None = None
    pw2wannier90: CodeParallelization | None = None
    wann2kcp: CodeParallelization | None = None
    wannier90: CodeParallelization | None = None

    @model_validator(mode="after")
    def reject_unsupported_flags(self) -> Self:
        """Reject ``npool`` / ``pd`` for codes whose command line has no such flag."""
        for code, cfg in self.as_dict().items():
            if cfg.npool is not None and code not in POOL_SUPPORTING_CODES:
                raise ValueError(
                    f"'npool' is not valid for {code} (it does not parallelize over "
                    f"k-point pools); pools are only supported by "
                    f"{sorted(POOL_SUPPORTING_CODES)}. Set 'ntasks' instead."
                )
            if cfg.pd is not None and code not in PD_SUPPORTING_CODES:
                raise ValueError(
                    f"'pd' (pencil decomposition) is not valid for {code}; it is only "
                    f"supported by {sorted(PD_SUPPORTING_CODES)}."
                )
        return self

    def as_dict(self) -> dict[str, CodeParallelization]:
        """Return the configured (non-``None``) code entries as a plain dict."""
        return {code: cfg for code in ALL_CODES if (cfg := getattr(self, code)) is not None}

    def as_mapping(self, computer: ComputerInput | None = None) -> ParallelizationDict:
        """Return the per-code settings as the mapping the workgraphs consume.

        Each configured code maps to its set (non-``None``) fields via
        pydantic's own dump, with ``walltime`` folded into whole-second
        ``max_wallclock_seconds`` — the key ``aiida-koopmans`` expects, and
        the only field this mapping does not pass through verbatim. Passing
        ``computer`` also gives every code its ``computer.walltime`` default
        wherever the code's own ``walltime`` is unset
        (:func:`resolve_effective_walltime`); this is the one place that
        default reaches a code other than through the pw-specific seeding in
        :func:`koopmans.aiida.conversion.code_parallelization`. Omitting
        ``computer`` (the default) reproduces the old configured-codes-only
        behaviour. A code with no fields set and no computer default is
        omitted.

        This is the ``ParallelizationDict`` shape ``aiida-koopmans`` expects.
        """
        mapping: dict[str, dict[str, object]] = {}
        for code in ALL_CODES:
            cfg = getattr(self, code)
            fields = cfg.model_dump(exclude_none=True, exclude={"walltime"}) if cfg else {}
            walltime = resolve_effective_walltime(cfg, computer)
            if walltime is not None:
                fields = {**fields, "max_wallclock_seconds": int(walltime.total_seconds())}
            if fields:
                mapping[code] = fields
        return cast(ParallelizationDict, mapping)


def resolve_effective_walltime(
    config: CodeParallelization | None, computer: ComputerInput | None
) -> timedelta | None:
    """Return one code's effective walltime: its own override, else ``computer.walltime``.

    The one place this precedence is decided; both
    :meth:`ParallelizationInput.as_mapping` and
    :func:`koopmans.aiida.conversion.code_parallelization` call it rather than
    each repeating the fallback.
    """
    if config is not None and config.walltime is not None:
        return config.walltime
    return computer.walltime if computer is not None else None
