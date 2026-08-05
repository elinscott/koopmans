"""Dependency guard: the installed plumpy emits outputs without changing the process specification.

A specification is built once per process class and shared by every
instance of it, so a port added while emitting a namespaced output
validates the outputs of all later instances. In a daemon worker that had
already served such a process, a plain value under that name was then
rejected as not being a ``Mapping``, and the workgraph task that emitted
it failed on links that work in a fresh interpreter.

The fix is aiidateam/plumpy#351, carried on the ``patched`` branch of the
fork that ``[tool.uv.sources]`` points at. Patched and unpatched both
report version ``0.26.0``, so only behavior can tell them apart: this
guard fails if the source entry is dropped or plumpy is resolved from
PyPI before #351 is released.
"""

from __future__ import annotations

from typing import Any

from plumpy.process_spec import ProcessSpec
from plumpy.processes import Process


class _DynamicOutputProcess(Process):
    """Emit a namespaced output into a dynamic output namespace."""

    @classmethod
    def define(cls, spec: ProcessSpec) -> None:
        """Accept any integer output under any name."""
        super().define(spec)
        spec.outputs.valid_type = int

    def run(self) -> Any:
        """Emit one value one namespace deep."""
        self.out("alphas.filled", 1)


def test_emitting_a_namespaced_output_leaves_the_spec_unchanged() -> None:
    """Emit ``alphas.filled`` and check no ``alphas`` port appears on the shared spec."""
    before = set(_DynamicOutputProcess.spec().outputs.keys())

    process = _DynamicOutputProcess()
    process.execute()

    assert process.outputs == {"alphas": {"filled": 1}}
    assert set(_DynamicOutputProcess.spec().outputs.keys()) == before
