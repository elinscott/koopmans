"""Probe run in a config-less subprocess by ``test_configless_imports``.

Not collected by pytest (no ``test_`` prefix); executed by path with an
empty ``AIIDA_PATH`` so any module-level import that loads the AiiDA
configuration fails the subprocess.
"""

from __future__ import annotations

import sys

#: Modules a config-less context must be able to import: the input-file
#: schema (docs builds render it) and the dispatcher package (the CLI
#: imports it before any profile exists).
CONFIGLESS_MODULES = (
    "koopmans.input_file",
    "koopmans.aiida.workflows",
)


def main() -> None:
    """Import the guarded modules and reject a workgraph-reaching closure."""
    for name in CONFIGLESS_MODULES:
        __import__(name)
    bad = sorted(m for m in sys.modules if m.startswith("aiida_workgraph") or "pythonjob" in m)
    if bad:
        raise AssertionError(f"import closure pulls {bad}")


if __name__ == "__main__":
    main()
