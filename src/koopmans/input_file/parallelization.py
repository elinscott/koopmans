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
into AiiDA ``metadata.options`` / ``settings.cmdline``. The top-level
``computer.account`` and ``computer.queue`` are run-wide, with no per-code
override; :meth:`ParallelizationInput.as_mapping` folds them into every
code's entry alongside the walltime default.

``yambo`` additionally takes a per-driver MPI role split
(:class:`YamboParallelization`), since yambo parallelizes over named roles
(``k``, ``eh``, ``q``, …) baked into its own runcard rather than over
``-npool``/``-pd``::

    parallelization:
      yambo:
        ntasks: 4
        omp: 1
        bethe_salpeter:   {k: 2, eh: 2}     # becomes BS_CPU = "2 2", BS_ROLEs = "k eh"
        static_screening: {k: 4}            # becomes X_and_IO_CPU / X_and_IO_ROLEs
        dipoles:          {k: 4}            # becomes DIP_CPU / DIP_ROLEs

See :class:`YamboParallelization` for the role vocabulary and validation.
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


__all__ = [
    "CodeParallelization",
    "ParallelizationInput",
    "YamboParallelization",
    "resolve_effective_walltime",
]

# yambo's own role vocabulary per parallel driver (yambo 5.3
# ``src/interface/INIT_load.F``, ``CPU_structure_load``: DIP = (k,c,v),
# X_and_IO = (q,g,k,c,v), BS = (k,eh,t)), and each driver's runcard variable
# prefix (``src/modules/mod_parallel.F``, ``CPU_str_reset``).
_YAMBO_ROLE_VOCAB: dict[str, frozenset[str]] = {
    "bethe_salpeter": frozenset({"k", "eh", "t"}),
    "static_screening": frozenset({"q", "g", "k", "c", "v"}),
    "dipoles": frozenset({"k", "c", "v"}),
}
_YAMBO_RUNCARD_PREFIX: dict[str, str] = {
    "bethe_salpeter": "BS",
    "static_screening": "X_and_IO",
    "dipoles": "DIP",
}


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
        "top-level ``computer.walltime`` default (``HH:MM:SS``, an ISO 8601 duration "
        "such as ``PT2H``, or a plain seconds count). Valid for every code.",
    )


class YamboParallelization(CodeParallelization):
    """Parallelization settings for yambo: rank count plus per-driver MPI role splits.

    yambo has no ``-npool``/``-pd`` concept (rejected for it like every
    other field on :class:`CodeParallelization`); instead each parallel
    driver distributes its own work over the run's MPI ranks along named
    roles, in a runcard ``<DRIVER>_ROLEs``/``<DRIVER>_CPU`` pair of strings.
    ``bethe_salpeter``, ``static_screening`` and ``dipoles`` name one split
    each, as an ordered mapping of role name to rank count. yambo takes the
    role order as the nesting order of its own parallel structure (yambo 5.3
    ``src/parallel/PARALLEL_get_user_structure.F`` parses the ``_CPU``/
    ``_ROLEs`` strings positionally, and
    ``src/parallel/PARALLEL_assign_chains_and_COMMs.F`` builds nested
    communicators in that same position order, outermost first) — so the
    mapping's insertion order is passed straight through.

    A driver's role counts must multiply to ``ntasks``: yambo aborts at
    startup otherwise, so a driver named without ``ntasks`` set is rejected
    here instead. A driver left unset means yambo distributes that work over
    the ranks itself.
    """

    bethe_salpeter: dict[str, int] | None = Field(
        default=None,
        description="MPI role split for the Bethe-Salpeter kernel (yambo's ``BS_CPU``/"
        "``BS_ROLEs``), as an ordered {role: rank count} mapping. Valid roles: "
        "``k`` (k-points), ``eh`` (electron-hole pairs), ``t`` (transitions).",
    )
    static_screening: dict[str, int] | None = Field(
        default=None,
        description="MPI role split for the static screening / response function "
        "(yambo's ``X_and_IO_CPU``/``X_and_IO_ROLEs``), as an ordered {role: rank count} "
        "mapping. Valid roles: ``q`` (q-points), ``g`` (G-vectors), ``k`` (k-points), "
        "``c`` (conduction bands), ``v`` (valence bands).",
    )
    dipoles: dict[str, int] | None = Field(
        default=None,
        description="MPI role split for the dipole matrix elements (yambo's ``DIP_CPU``/"
        "``DIP_ROLEs``), as an ordered {role: rank count} mapping. Valid roles: "
        "``k`` (k-points), ``c`` (conduction bands), ``v`` (valence bands).",
    )

    @model_validator(mode="after")
    def check_role_splits(self) -> Self:
        """Validate each named driver's roles, counts, and product against ``ntasks``."""
        for driver, vocab in _YAMBO_ROLE_VOCAB.items():
            roles: dict[str, int] | None = getattr(self, driver)
            if roles is None:
                continue
            if self.ntasks is None:
                raise ValueError(
                    f"'parallelization.yambo.{driver}' needs 'parallelization.yambo.ntasks' "
                    "set: yambo's role split must multiply out to the total rank count."
                )
            unknown = sorted(role for role in roles if role not in vocab)
            if unknown:
                raise ValueError(
                    f"'parallelization.yambo.{driver}' names unknown role(s) {unknown}; "
                    f"valid roles for {driver} are {sorted(vocab)}."
                )
            non_positive = sorted(role for role, count in roles.items() if count < 1)
            if non_positive:
                raise ValueError(
                    f"'parallelization.yambo.{driver}' role(s) {non_positive} must have a "
                    "positive rank count."
                )
            product = 1
            for count in roles.values():
                product *= count
            if product != self.ntasks:
                raise ValueError(
                    f"'parallelization.yambo.{driver}' role counts {dict(roles)} multiply "
                    f"to {product}, not 'parallelization.yambo.ntasks' = {self.ntasks}; "
                    "yambo aborts unless the two agree."
                )
        return self

    def to_runcard_dict(self) -> dict[str, str]:
        """Return the named drivers' splits as yambo's own ``*_CPU``/``*_ROLEs`` strings."""
        runcard: dict[str, str] = {}
        for driver, prefix in _YAMBO_RUNCARD_PREFIX.items():
            roles: dict[str, int] | None = getattr(self, driver)
            if not roles:
                continue
            runcard[f"{prefix}_CPU"] = " ".join(str(count) for count in roles.values())
            runcard[f"{prefix}_ROLEs"] = " ".join(roles)
        return runcard


class ParallelizationInput(BaseModel):
    """Per-code parallelization settings.

    A mapping of code name to :class:`CodeParallelization`. Only the codes
    listed here are recognised; any other key is rejected. Codes left unset
    inherit the QE/AiiDA defaults (a single MPI rank, no pools). ``yambo``'s
    entry is a :class:`YamboParallelization`, adding the per-driver MPI role
    splits.
    """

    pw: CodeParallelization | None = None
    kcp: CodeParallelization | None = None
    kcw: CodeParallelization | None = None
    ph: CodeParallelization | None = None
    projwfc: CodeParallelization | None = None
    pw2wannier90: CodeParallelization | None = None
    wann2kcp: CodeParallelization | None = None
    wannier90: CodeParallelization | None = None
    yambo: YamboParallelization | None = None

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
        ``computer`` also gives every code its ``computer.walltime``,
        ``computer.account`` (as ``account``), and ``computer.queue`` (as
        ``queue_name``) — run-wide settings with no per-code override, folded
        into every code's entry the same way (:func:`resolve_effective_walltime`
        for walltime's own-vs-computer precedence). This is the one place those
        computer-level defaults reach a code other than through the
        pw-specific seeding in :func:`koopmans.aiida.conversion.code_parallelization`.
        Omitting ``computer`` (the default) reproduces the old
        configured-codes-only behaviour. A code with no fields set and no
        computer default is omitted.

        ``yambo``'s ``bethe_salpeter``/``static_screening``/``dipoles`` role
        splits do not pass through as their own keys: they are replaced by a
        single ``runcard`` key holding their serialized ``*_CPU``/``*_ROLEs``
        strings (:meth:`YamboParallelization.to_runcard_dict`), the shape
        ``aiida-koopmans``'s ``CodeParallelization`` TypedDict declares.

        This is the ``ParallelizationDict`` shape ``aiida-koopmans`` expects.
        """
        mapping: dict[str, dict[str, object]] = {}
        for code in ALL_CODES:
            cfg = getattr(self, code)
            fields = cfg.model_dump(exclude_none=True, exclude={"walltime"}) if cfg else {}
            if isinstance(cfg, YamboParallelization):
                for driver in _YAMBO_ROLE_VOCAB:
                    fields.pop(driver, None)
                runcard = cfg.to_runcard_dict()
                if runcard:
                    fields["runcard"] = runcard
            walltime = resolve_effective_walltime(cfg, computer)
            if walltime is not None:
                fields = {**fields, "max_wallclock_seconds": int(walltime.total_seconds())}
            if computer is not None and computer.account is not None:
                fields = {**fields, "account": computer.account}
            if computer is not None and computer.queue is not None:
                fields = {**fields, "queue_name": computer.queue}
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
