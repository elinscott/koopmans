"""AiiDA integration for Koopmans.

This package provides utilities for converting Koopmans input to AiiDA data nodes
and setting up the AiiDA environment.
"""

from koopmans.aiida.conversion import convert_koopmans_input_for_builder
from koopmans.aiida.setup import ensure_pseudo_family_installed

__all__ = [
    "convert_koopmans_input_for_builder",
    "ensure_pseudo_family_installed",
]
