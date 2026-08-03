"""Drive koopmans calculations from python.

``build`` / ``run`` / ``submit`` take a
:class:`~koopmans.input_file.KoopmansInput` and mirror the workgraph verbs
of the underlying AiiDA engine; :class:`Results` reads a finished
calculation back in user vocabulary (energies, orbital energies, screening
parameters), so no AiiDA fluency is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from aiida import orm
    from aiida_workgraph import WorkGraph

    from koopmans.input_file import KoopmansInput

__all__ = ["Results", "build", "run", "submit"]


def build(koopmans_input: KoopmansInput) -> WorkGraph:
    """Materialize the calculation's workgraph without running it."""
    from koopmans.aiida.workflows import build_workgraph

    _ensure_profile()
    return build_workgraph(koopmans_input)


def run(koopmans_input: KoopmansInput) -> Results:
    """Run the calculation to completion in this interpreter.

    Blocks until the calculation finishes; the returned :class:`Results`
    is ready to read.
    """
    return _launch(build(koopmans_input), blocking=True)


def submit(koopmans_input: KoopmansInput, *, wait: bool = False) -> Results:
    """Hand the calculation to the daemon and return without blocking.

    With ``wait=True`` the call blocks until the daemon finishes the
    calculation. Otherwise the returned :class:`Results` is a handle on the
    running calculation: poll ``finished`` and read it once done.
    """
    return _launch(build(koopmans_input), blocking=False, wait=wait)


def _launch(workgraph: WorkGraph, *, blocking: bool, wait: bool = False) -> Results:
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
    return Results(workgraph.process)


def _ensure_profile() -> None:
    """Load the koopmans AiiDA profile unless one is already loaded."""
    from aiida.manage.configuration import get_profile

    from koopmans.aiida.setup.profile import load_koopmans_profile

    if get_profile() is None:
        load_koopmans_profile()


class Results:
    """A koopmans calculation's results, read back in user vocabulary.

    Returned by :func:`run` and :func:`submit`, or reconstructed later with
    :meth:`from_pk`. Every accessor returns plain python / numpy values.
    Energies are in eV. The accessors read the singlepoint (kcp.x DSCF)
    outputs; for the other tasks they raise ``NotImplementedError`` naming
    the gap.
    """

    def __init__(self, node: orm.ProcessNode) -> None:
        """Wrap the calculation's provenance node (internal; see ``from_pk``)."""
        self._node = node

    @classmethod
    def from_pk(cls, pk: int) -> Results:
        """Reconnect to an earlier calculation by the integer id ``pk``."""
        from aiida import orm

        _ensure_profile()
        node = orm.load_node(pk)
        if not isinstance(node, orm.ProcessNode):
            raise ValueError(f"pk {pk} is not a calculation; it holds a {type(node).__name__}.")
        return cls(node)

    @property
    def pk(self) -> int:
        """Integer id of the calculation; feed it to :meth:`from_pk`."""
        pk = self._node.pk
        if pk is None:
            raise RuntimeError("The calculation was never stored, so it has no id.")
        return int(pk)

    @property
    def finished(self) -> bool:
        """Whether the calculation has finished (successfully or not)."""
        return bool(self._node.is_terminated)

    @property
    def total_energy(self) -> float:
        """Total energy of the final KI calculation (eV)."""
        return float(self._parameters()["energy"])

    @property
    def homo_energy(self) -> float | None:
        """Energy of the highest occupied orbital (eV)."""
        homo = self._parameters()["homo_energy"]
        return float(homo) if homo is not None else None

    @property
    def lumo_energy(self) -> float | None:
        """Energy of the lowest unoccupied orbital (eV)."""
        lumo = self._parameters()["lumo_energy"]
        return float(lumo) if lumo is not None else None

    @property
    def ionization_potential(self) -> float | None:
        """Ionization potential: the negative of the HOMO energy (eV)."""
        homo = self.homo_energy
        return -homo if homo is not None else None

    @property
    def electron_affinity(self) -> float | None:
        """Electron affinity: the negative of the LUMO energy (eV)."""
        lumo = self.lumo_energy
        return -lumo if lumo is not None else None

    @property
    def orbital_energies(self) -> np.ndarray:
        """Orbital energies of the final KI calculation (eV).

        One row per spin channel, occupied and empty states together in
        ascending band order.
        """
        eigenvalues: np.ndarray = self._outputs().eigenvalues.get_array("eigenvalues")
        return eigenvalues

    @property
    def alphas(self) -> dict[str, dict[str, list[float]]]:
        """Screening parameters the final KI consumed, one per orbital.

        Keyed ``filled`` / ``empty``, then by spin channel (``none`` for an
        unpolarized calculation, ``up`` / ``down`` for a spin-polarized
        one); each value lists one alpha per orbital of that channel.
        """
        outputs = self._outputs()
        return {
            "filled": dict(outputs.alphas.filled.get_dict()),
            "empty": dict(outputs.alphas.empty.get_dict()),
        }

    def dump(self, path: str | Path) -> Path:
        """Write the calculation's files under ``path`` and return that path.

        Produces the same directory layout as ``koopmans run``: one folder
        per step, each with its ``inputs`` and ``outputs``.
        """
        from koopmans.aiida.dumping import dump_workgraph

        self._require_finished()
        return dump_workgraph(self._node, Path(path), overwrite=True)

    def _require_finished(self) -> None:
        """Raise unless the calculation finished successfully."""
        node = self._node
        if not node.is_terminated:
            raise RuntimeError(
                f"Calculation {node.pk} is still running: wait for it to finish "
                "(or submit with wait=True) before reading its results."
            )
        if not node.is_finished_ok:
            raise RuntimeError(
                f"Calculation {node.pk} failed (exit status {node.exit_status}); "
                "its results cannot be read."
            )

    def _outputs(self) -> Any:
        """Return the finished calculation's output sockets, guarded."""
        self._require_finished()
        outputs = self._node.outputs
        if "parameters" not in outputs:
            raise NotImplementedError(
                f"Results reads the singlepoint (kcp.x DSCF) outputs, which "
                f"calculation {self._node.pk} ({self._node.process_label}) does not "
                "provide; read this calculation's outputs from its AiiDA node for now."
            )
        return outputs

    def _parameters(self) -> dict[str, Any]:
        """Return the final KI's parsed output parameters."""
        parameters: dict[str, Any] = self._outputs().parameters.get_dict()
        return parameters
