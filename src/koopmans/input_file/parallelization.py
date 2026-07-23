"""Per-code parallelization settings.

The top-level ``parallelization`` block maps each code (``pw``, ``kcp``, …)
to a small config of MPI-rank count (``ntasks``), k-point-pool count
(``npool``), and a pencil-decomposition switch (``pd``). ``ntasks`` becomes the
scheduler's ``tot_num_mpiprocs``; ``npool`` becomes ``-npool`` and ``pd``
becomes ``-pd true`` on the QE command line. See
:func:`koopmans.aiida.conversion.code_parallelization` for the translation
into AiiDA ``metadata.options`` / ``settings.cmdline``.
"""

from __future__ import annotations

from typing import Any, Self, cast

from aiida_koopmans.types import CODE_NAMES, ParallelizationDict
from pydantic import Field, model_validator

from koopmans.base import BaseModel

# Every code the parallelization block recognises. Sourced from the single
# ``aiida_koopmans.types`` vocabulary (``CodeName``) rather than duplicated here.
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


# NOTE: keep this Pydantic model (and the per-code fields it validates) in
# sync with the ``CodeParallelization`` TypedDict in ``aiida_koopmans.types``
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

    def as_mapping(self) -> ParallelizationDict:
        """Return the per-code settings as the mapping the workgraphs consume.

        Each configured code maps to a dict of its set (non-``None``) fields
        (``ntasks`` / ``npool`` / ``pd``). This is the ``ParallelizationDict``
        shape ``aiida-koopmans``'s ``resolve_parallelization`` expects — one
        dict input per graph, keyed by code name. A code with no set field is
        omitted entirely. Built as a plain dict (the code keys are dynamic) and
        cast to the ``TypedDict`` on return.
        """
        mapping: dict[str, Any] = {}
        for code, cfg in self.as_dict().items():
            fields: dict[str, int | bool] = {
                key: value
                for key, value in (("ntasks", cfg.ntasks), ("npool", cfg.npool), ("pd", cfg.pd))
                if value is not None
            }
            if fields:
                mapping[code] = fields
        return cast(ParallelizationDict, mapping)
