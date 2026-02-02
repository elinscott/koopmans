"""Debugging utilities for koopmans."""

import sys
import traceback
from types import TracebackType

import ipdb


def _pdb_exception_hook(
    exception_type: type[BaseException],
    exception_value: BaseException,
    exception_traceback: TracebackType | None,
) -> None:
    traceback.print_exception(exception_type, exception_value, exception_traceback)
    ipdb.post_mortem(exception_traceback)


def enable_pdb() -> None:
    """Enable ipdb debugger to catch unhandled exceptions."""
    sys.tracebacklimit = None
    sys.excepthook = _pdb_exception_hook
