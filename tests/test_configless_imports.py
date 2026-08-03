"""Import-closure guard: the user-facing modules import without an AiiDA config.

Mirrors aiida-koopmans' ``test_vocabulary_imports``: aiida-pythonjob loads
the AiiDA configuration at import time (aiidateam/aiida-pythonjob#84), so
any module-level import whose closure reaches it makes the package
unimportable on a machine with no configuration — a docs build, a fresh
install before ``koopmans install``, or a bare CI runner.

The check runs :mod:`tests.configless_probe` in a subprocess. In-process
importlib cannot give this guarantee: the test process has already loaded
an AiiDA configuration (the test profile) and populated ``sys.modules``
with much of the import closure, and neither can be unloaded — the
configuration is a module-global singleton, and re-importing a cached
module skips the module-level code under test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_imports_without_aiida_configuration(tmp_path: Path) -> None:
    """Run the probe in a subprocess whose ``AIIDA_PATH`` is an empty directory.

    An import in the probe's closure that loads the AiiDA configuration —
    or pulls aiida-workgraph / aiida-pythonjob without happening to touch
    it — fails the subprocess.
    """
    probe = Path(__file__).parent / "configless_probe.py"
    env = dict(os.environ, AIIDA_PATH=str(tmp_path))
    result = subprocess.run(  # noqa: S603 -- fixed argv: this interpreter + our probe
        [sys.executable, str(probe)], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
