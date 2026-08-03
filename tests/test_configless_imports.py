"""Import-closure guard: the user-facing modules import without an AiiDA config.

Mirrors aiida-koopmans' ``test_vocabulary_imports``: aiida-pythonjob loads
the AiiDA configuration at import time (aiidateam/aiida-pythonjob#84), so
any module-level import whose closure reaches it makes the package
unimportable on a machine with no configuration — a docs build, a fresh
install before ``koopmans install``, or a bare CI runner.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

#: Modules a config-less context must be able to import: the input-file
#: schema (docs builds render it) and the dispatcher package (the CLI
#: imports it before any profile exists).
CONFIGLESS_MODULES = (
    "koopmans.input_file",
    "koopmans.aiida.workflows",
)


def test_imports_without_aiida_configuration(tmp_path: Path) -> None:
    """Import the user-facing modules in a subprocess with no AiiDA config.

    ``AIIDA_PATH`` points at an empty directory, so an import that loads
    the AiiDA configuration fails the subprocess; the sys.modules check
    additionally rejects a closure that pulls aiida-workgraph or
    aiida-pythonjob without happening to touch the configuration.
    """
    code = "\n".join(
        [
            "import sys",
            f"for name in {CONFIGLESS_MODULES!r}:",
            "    __import__(name)",
            "bad = sorted(",
            "    m for m in sys.modules",
            "    if m.startswith('aiida_workgraph') or 'pythonjob' in m",
            ")",
            "assert not bad, f'import closure pulls {bad}'",
        ]
    )
    env = dict(os.environ, AIIDA_PATH=str(tmp_path))
    result = subprocess.run(  # noqa: S603 -- fixed argv: this interpreter + a literal
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
