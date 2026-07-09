"""Utility functions for AiiDA integration."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiida.orm import Node, ProcessNode

__all__ = ["get_node_label", "suppress_aiida_logging", "suppress_stdout"]


def get_node_label(node: ProcessNode, include_code: bool = True) -> str:
    """Extract a meaningful label for a ProcessNode.

    Traverses up the caller chain to find a descriptive label, skipping
    generic labels like "iteration_XX". For CalcJobNodes, optionally
    includes the code label (e.g., "pw", "ph").

    :param node: The ProcessNode to label.
    :param include_code: Whether to include the code label for CalcJobNodes.
    :return: A descriptive label string.
    """
    from aiida.common.links import LinkType
    from aiida.orm import CalcJobNode

    # Get the call link label by traversing up the caller chain
    call_link_label = None
    current: Node | None = node
    while current is not None:
        try:
            link_triple = current.base.links.get_incoming(
                link_type=(LinkType.CALL_CALC, LinkType.CALL_WORK)
            ).one()
            label = link_triple.link_label
            # Skip generic labels like "iteration_XX" or "task_XX"
            if not re.match(r"^(iteration|task)_\d+$", label):
                call_link_label = label
                break
            current = link_triple.node
        except ValueError:
            break

    # For CalcJobNodes, get the code label
    code_label = None
    if include_code and isinstance(node, CalcJobNode):
        try:
            code = node.inputs.code
            code_label = code.label.split("@")[0] if code else None
        except (AttributeError, KeyError, TypeError):
            pass

    # Build the label
    parts = []
    if code_label:
        parts.append(code_label)
    if call_link_label:
        parts.append(call_link_label)

    return "-".join(parts) if parts else "unknown"


@contextmanager
def suppress_aiida_logging() -> Iterator[None]:
    """Context manager to suppress AiiDA's logging output.

    Useful for suppressing verbose output during dump operations or workgraph execution.
    """
    aiida_logger = logging.getLogger("aiida")
    original_level = aiida_logger.level
    aiida_logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        aiida_logger.setLevel(original_level)


@contextmanager
def suppress_stdout() -> Iterator[None]:
    """Context manager to suppress stdout output.

    Useful for suppressing print statements from third-party libraries.
    """
    import os
    import sys

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout
