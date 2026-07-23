"""Per-code parallelization settings.

The top-level ``parallelization`` block maps each code (``pw``, ``kcp``, …)
to a small config of MPI-rank count (``ntasks``) and k-point-pool count
(``npool``). ``ntasks`` becomes the scheduler's ``tot_num_mpiprocs``; ``npool``
becomes ``-npool`` on the QE command line. See
:func:`koopmans.aiida.conversion.code_parallelization` for the translation
into AiiDA ``metadata.options`` / ``settings.cmdline``.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from koopmans.base import BaseModel

# Every code the parallelization block recognises.
ALL_CODES: tuple[str, ...] = (
    "pw",
    "kcp",
    "kcw",
    "ph",
    "projwfc",
    "pw2wannier90",
    "wann2kcp",
    "wannier90",
)


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
        "k-point grid). Not valid for wannier90.",
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
    def reject_npool_for_wannier90(self) -> Self:
        """wannier90 does not parallelize over k-point pools, so reject ``npool``."""
        if self.wannier90 is not None and self.wannier90.npool is not None:
            raise ValueError(
                "'npool' is not valid for wannier90 (it does not parallelize over "
                "k-point pools); set 'ntasks' instead"
            )
        return self

    def as_dict(self) -> dict[str, CodeParallelization]:
        """Return the configured (non-``None``) code entries as a plain dict."""
        return {code: cfg for code in ALL_CODES if (cfg := getattr(self, code)) is not None}

    def as_mapping(self) -> dict[str, dict[str, int]]:
        """Return the per-code settings as the plain mapping the workgraphs consume.

        Each configured code maps to a dict of its set (non-``None``) fields
        (``ntasks`` / ``npool``). This is the shape ``aiida-koopmans``'s
        ``resolve_parallelization`` expects — one dict input per graph, keyed
        by code name. A code with no set field is omitted entirely.
        """
        mapping: dict[str, dict[str, int]] = {}
        for code, cfg in self.as_dict().items():
            fields = {
                key: value
                for key, value in (("ntasks", cfg.ntasks), ("npool", cfg.npool))
                if value is not None
            }
            if fields:
                mapping[code] = fields
        return mapping
