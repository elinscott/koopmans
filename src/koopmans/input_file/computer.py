"""Which AiiDA computer a calculation runs on, and its scheduler options.

The top-level ``computer`` block names the AiiDA computer label a run
submits to. A bare string (``computer: daint``) is shorthand for the block
form with only ``name`` set; the block additionally states a scheduler
account, queue, and default walltime. ``account``/``queue`` feed pw's
``metadata.options`` via
:func:`koopmans.aiida.conversion.code_parallelization`; ``walltime`` reaches
every code, either through that same function (pw) or as the fallback in
:meth:`koopmans.input_file.parallelization.ParallelizationInput.as_mapping`
(every other code) wherever the code's own
``parallelization.<code>.walltime`` is unset. A scheduler with no
account/queue concept (HyperQueue, the direct scheduler) refuses the ones it
cannot honour — see
:func:`koopmans.aiida.conversion.validate_computer_scheduler_support`.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator, Field

from koopmans.base import BaseModel
from koopmans.input_file._utils import Walltime

__all__ = ["ComputerConfig", "ComputerInput"]


class ComputerInput(BaseModel):
    """The AiiDA computer a calculation runs on, and its scheduler options."""

    name: str = Field(
        default="localhost",
        description="the AiiDA computer label the calculation submits to",
    )
    account: str | None = Field(
        default=None,
        description="scheduler account/project to charge (e.g. SLURM's `--account`); "
        "refused where the computer's scheduler has no such concept (HyperQueue, the "
        "direct scheduler)",
    )
    queue: str | None = Field(
        default=None,
        description="scheduler queue/partition to submit to (e.g. SLURM's `--partition`); "
        "refused where the computer's scheduler has no such concept (HyperQueue, the "
        "direct scheduler)",
    )
    walltime: Walltime = Field(
        default=None,
        description="default wallclock limit for every step (`2h`, `90m`, `1d12h`, "
        "`HH:MM:SS`, or any pydantic-native duration); `parallelization.<code>.walltime` "
        "overrides it for that one code. Has no effect on the direct scheduler, which "
        "enforces no wallclock limit",
    )


def _coerce_bare_label(value: Any) -> Any:
    """Accept a bare computer label (``daint``) in place of the block form."""
    if isinstance(value, str):
        return {"name": value}
    return value


ComputerConfig = Annotated[ComputerInput, BeforeValidator(_coerce_bare_label)]
