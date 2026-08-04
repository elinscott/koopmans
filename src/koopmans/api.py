"""Drive koopmans calculations from python.

``build`` / ``run`` / ``submit`` take a
:class:`~koopmans.input_file.KoopmansInput` and mirror the workgraph verbs
of the underlying AiiDA engine. A finished calculation is read back as a
plain dict of its outputs — deserialized to python / numpy values and keyed
by output socket name — either from ``run`` directly or from
:func:`outputs` given the calculation's integer id. Output socket names
are public API: renaming one is a user-breaking change, reviewed like a
schema keyword.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiida import orm
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput

__all__ = ["build", "outputs", "run", "submit"]


def build(koopmans_input: KoopmansInput) -> WorkGraph:
    """Materialize the calculation's workgraph without running it."""
    from koopmans.aiida.workflows import build_workgraph

    _ensure_profile()
    return build_workgraph(koopmans_input)


def run(koopmans_input: KoopmansInput) -> dict[str, Any]:
    """Run the calculation to completion in this interpreter.

    Blocks until the calculation finishes and returns its outputs
    (:func:`outputs`); a calculation that fails raises instead.
    """
    node = launch(build(koopmans_input), blocking=True)
    _require_finished_ok(node)
    return _deserialized_outputs(node)


def submit(koopmans_input: KoopmansInput, *, wait: bool = False) -> int:
    """Hand the calculation to the daemon and return its integer id.

    With ``wait=True`` the call blocks until the daemon finishes the
    calculation. The id survives the python session; read the finished
    calculation back with :func:`outputs`.
    """
    node = launch(build(koopmans_input), blocking=False, wait=wait)
    pk = node.pk
    if pk is None:
        raise RuntimeError("The calculation was never stored, so it has no id.")
    return int(pk)


def outputs(pk: int) -> dict[str, Any]:
    """Return the outputs of the finished calculation ``pk``, deserialized.

    Keyed by output socket name, with nested namespaces as nested dicts
    and every value a plain python / numpy one. Remote-scratch and
    retrieved-file handles have no plain python analogue and are omitted.
    A calculation that is still running, or failed, raises instead.
    """
    from aiida import orm

    _ensure_profile()
    node = orm.load_node(pk)
    if not isinstance(node, orm.ProcessNode):
        raise ValueError(f"pk {pk} is not a calculation; it holds a {type(node).__name__}.")
    _require_finished_ok(node)
    return _deserialized_outputs(node)


def launch(workgraph: WorkGraph, *, blocking: bool, wait: bool = False) -> orm.ProcessNode:
    """Start ``workgraph`` — the single call site every verb goes through.

    If upstream's launch inversion lands (aiida-core#7261 /
    aiida-workgraph#768: ``engine.run(workgraph)`` replacing
    ``workgraph.run()``), this helper is the only place to migrate.
    """
    if blocking:
        workgraph.run()
    else:
        from koopmans.aiida.setup.daemon import ensure_daemon_running

        ensure_daemon_running()
        workgraph.submit(wait=wait)
    node: orm.ProcessNode = workgraph.process
    return node


def _ensure_profile() -> None:
    """Load the koopmans AiiDA profile unless one is already loaded."""
    from aiida.manage.configuration import get_profile

    from koopmans.aiida.setup.profile import load_koopmans_profile

    if get_profile() is None:
        load_koopmans_profile()


def _require_finished_ok(node: orm.ProcessNode) -> None:
    """Raise unless the calculation finished successfully."""
    if not node.is_terminated:
        raise RuntimeError(
            f"Calculation {node.pk} is still running: wait for it to finish "
            "(or submit with wait=True) before reading its outputs."
        )
    if not node.is_finished_ok:
        raise RuntimeError(
            f"Calculation {node.pk} failed (exit status {node.exit_status}); "
            "its outputs cannot be read."
        )


def _deserialized_outputs(node: orm.ProcessNode) -> dict[str, Any]:
    """Walk the node's outputs into a plain nested dict, keyed by socket name.

    Every output link deserializes through aiida-pythonjob; file and
    scratch handles (``RemoteData``, ``FolderData``) are skipped.
    """
    from aiida import orm
    from aiida.common.links import LinkType
    from aiida_pythonjob.data.deserializer import deserialize_to_raw_python_data

    deserialized: dict[str, Any] = {}
    links = node.base.links.get_outgoing(link_type=LinkType.RETURN).all()
    for triple in sorted(links, key=lambda t: t.link_label):
        if isinstance(triple.node, (orm.RemoteData, orm.FolderData)):
            continue
        cursor = deserialized
        *parents, leaf = triple.link_label.split("__")
        for part in parents:
            cursor = cursor.setdefault(part, {})
        cursor[leaf] = deserialize_to_raw_python_data(triple.node)
    return deserialized
