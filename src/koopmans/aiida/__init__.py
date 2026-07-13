"""AiiDA integration for Koopmans.

This package provides utilities for converting Koopmans input to AiiDA data nodes
and setting up the AiiDA environment.
"""

from koopmans.aiida.setup.pseudos import ensure_pseudo_family_installed

__all__ = [
    "ensure_pseudo_family_installed",
]
