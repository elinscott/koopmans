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
(``k``, ``eh``, ``q``, …) rather than over ``-npool``/``-pd``::

    parallelization:
      yambo:
        ntasks: 4
        bethe_salpeter: {k: 2, eh: 2}   # becomes BS_CPU = "2 2", BS_ROLEs = "k eh"

Each named driver passes through :meth:`ParallelizationInput.as_mapping` as
its own ``{role: rank count}`` dict; ``aiida-koopmans`` turns it into
yambo's own ``*_CPU``/``*_ROLEs`` runcard strings. ``static_screening``
(roles ``q``/``g``/``k``/``c``/``v``) and ``dipoles`` (roles ``k``/``c``/
``v``) take the same shape. Each role's count must fit the system's own
phase space (a ``k`` count no larger than the number of irreducible
k-points, and so on) — a constraint this schema cannot check.

See :class:`YamboParallelization` for the role vocabulary and validation.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Self, cast

from aiida_koopmans.parallelization import CODE_NAMES, ParallelizationDict
from pydantic import Field, PositiveInt, model_validator

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
    "BetheSalpeterRoles",
    "CodeParallelization",
    "DipoleRoles",
    "ParallelizationInput",
    "StaticScreeningRoles",
    "YamboParallelization",
    "resolve_effective_walltime",
]

# The three yambo parallel drivers this schema names a role split for. Kept
# as a plain tuple (rather than duplicating aiida-koopmans's own
# ``YAMBO_ROLE_DRIVERS`` table, which also carries each driver's runcard
# prefix and role order) since this schema only needs the driver names —
# ``aiida-koopmans`` does the runcard translation.
_YAMBO_DRIVERS: tuple[str, ...] = ("bethe_salpeter", "static_screening", "dipoles")


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


class BetheSalpeterRoles(BaseModel):
    """MPI role split for yambo's Bethe-Salpeter kernel (``BS_CPU``/``BS_ROLEs``).

    Extra fields are rejected (:class:`~koopmans.base.BaseModel`'s default),
    so naming a role outside this vocabulary is refused by name.
    """

    k: PositiveInt | None = Field(default=None, description="k-points")
    eh: PositiveInt | None = Field(default=None, description="electron-hole pairs")
    t: PositiveInt | None = Field(default=None, description="transitions")


class StaticScreeningRoles(BaseModel):
    """MPI role split for yambo's static screening / response function.

    (``X_and_IO_CPU``/``X_and_IO_ROLEs``). Extra fields are rejected
    (:class:`~koopmans.base.BaseModel`'s default), so naming a role outside
    this vocabulary is refused by name.
    """

    q: PositiveInt | None = Field(default=None, description="q-points (momentum transfers)")
    g: PositiveInt | None = Field(default=None, description="G-vectors / blocks")
    k: PositiveInt | None = Field(default=None, description="k-points")
    c: PositiveInt | None = Field(default=None, description="conduction bands")
    v: PositiveInt | None = Field(default=None, description="valence bands")


class DipoleRoles(BaseModel):
    """MPI role split for yambo's dipole matrix elements (``DIP_CPU``/``DIP_ROLEs``).

    Extra fields are rejected (:class:`~koopmans.base.BaseModel`'s default),
    so naming a role outside this vocabulary is refused by name.
    """

    k: PositiveInt | None = Field(default=None, description="k-points")
    c: PositiveInt | None = Field(default=None, description="conduction bands")
    v: PositiveInt | None = Field(default=None, description="valence bands")


class YamboParallelization(CodeParallelization):
    """Parallelization settings for yambo: rank count plus per-driver MPI role splits.

    yambo has no ``-npool``/``-pd`` concept (rejected for it like every
    other field on :class:`CodeParallelization`); instead each parallel
    driver distributes its own work over the run's MPI ranks along named
    roles. ``bethe_salpeter`` (:class:`BetheSalpeterRoles`), ``static_screening``
    (:class:`StaticScreeningRoles`) and ``dipoles`` (:class:`DipoleRoles`)
    each name one split, as a small model of that driver's own roles; this
    schema passes each set role through as a plain ``{role: rank count}``
    dict. ``aiida-koopmans`` turns that into yambo's own runcard
    ``<DRIVER>_ROLEs``/``<DRIVER>_CPU`` pair of strings, in yambo's own
    fixed per-driver order (its ``YAMBO_ROLE_DRIVERS`` table) — roles are
    matched by name, not position (yambo 5.3
    ``src/parallel/PARALLEL_structure.F``), so this schema's own field
    order carries no meaning for that translation. The three role models
    above still declare their fields in yambo's own order (``k``, ``eh``,
    ``t`` for Bethe-Salpeter; ``q``, ``g``, ``k``, ``c``, ``v`` for static
    screening; ``k``, ``c``, ``v`` for dipoles), checked against
    ``aiida-koopmans``'s own table by a canary test, so the two packages'
    role vocabularies cannot drift apart.

    A driver's role counts must multiply to ``ntasks``. yambo does not
    abort on a mismatch: it logs a warning and silently discards the named
    split, falling back to its own automatic distribution instead (yambo
    5.3 ``src/parallel/PARALLEL_global_defaults.F``). That silent fallback
    is rejected here instead, at parse time, so a mismatched split is never
    passed through unnoticed. A driver named without ``ntasks`` set cannot
    be checked this way, so it is rejected too. A driver left unset means
    yambo distributes that work over the ranks itself; a driver whose model
    omits a role in its own vocabulary is also fine — yambo appends any
    missing ``q``/``k`` role as 1 itself (``PARALLEL_get_user_structure.F``).

    Example::

        parallelization:
          yambo:
            ntasks: 4
            bethe_salpeter: {k: 2, eh: 2}   # becomes BS_CPU = "2 2", BS_ROLEs = "k eh"

    ``static_screening`` and ``dipoles`` take the same shape, over their own
    roles. Each role's count must also fit the system's own phase space (a
    ``k`` count no larger than the number of irreducible k-points, and so
    on) — a constraint this schema cannot check.
    """

    bethe_salpeter: BetheSalpeterRoles | None = Field(
        default=None,
        description="MPI role split for the Bethe-Salpeter kernel (yambo's ``BS_CPU``/"
        "``BS_ROLEs``).",
    )
    static_screening: StaticScreeningRoles | None = Field(
        default=None,
        description="MPI role split for the static screening / response function "
        "(yambo's ``X_and_IO_CPU``/``X_and_IO_ROLEs``).",
    )
    dipoles: DipoleRoles | None = Field(
        default=None,
        description="MPI role split for the dipole matrix elements (yambo's ``DIP_CPU``/"
        "``DIP_ROLEs``).",
    )

    @model_validator(mode="after")
    def check_role_splits(self) -> Self:
        """Validate each named driver's role counts against ``ntasks``."""
        for driver in _YAMBO_DRIVERS:
            roles: BaseModel | None = getattr(self, driver)
            if roles is None:
                continue
            if self.ntasks is None:
                raise ValueError(
                    f"'parallelization.yambo.{driver}' needs 'parallelization.yambo.ntasks' "
                    "set: yambo's role split must multiply out to the total rank count."
                )
            set_roles = roles.model_dump(exclude_none=True)
            product = 1
            for count in set_roles.values():
                product *= count
            if product != self.ntasks:
                raise ValueError(
                    f"'parallelization.yambo.{driver}' role counts {set_roles} multiply "
                    f"to {product}, not 'parallelization.yambo.ntasks' = {self.ntasks}; "
                    "yambo would silently discard this split and fall back to its own "
                    "distribution rather than use it."
                )
        return self


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
        splits pass through as their own keys, each a plain ``{role: rank
        count}`` dict — the shape ``aiida-koopmans``'s ``YamboParallelization``
        TypedDict carries (a strict extension of its ``CodeParallelization``,
        for the ``yambo`` entry only). ``aiida-koopmans`` does the translation
        into yambo's own ``*_CPU``/``*_ROLEs`` runcard strings; this schema
        never builds those strings itself.

        This is the ``ParallelizationDict`` shape ``aiida-koopmans`` expects.
        """
        mapping: dict[str, dict[str, object]] = {}
        for code in ALL_CODES:
            cfg = getattr(self, code)
            fields = cfg.model_dump(exclude_none=True, exclude={"walltime"}) if cfg else {}
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
