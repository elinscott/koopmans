"""Shared test data, helper classes, and pytest fixtures for koopmans2.

Definitions live here; ``conftest.py`` just re-exports the fixtures so
pytest's collection machinery picks them up for every test module. Mirrors
the pattern used by the sibling ``aiida-koopmans2/tests/fixtures.py``.
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

# ----------------------------------------------------------------------
# Plain-data fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def tutorials_dir() -> Path:
    """Return the path to the tutorials directory shipped with the docs."""
    return Path(__file__).parent.parent / "docs" / "source" / "tutorials"


# ----------------------------------------------------------------------
# Broken-upstream-fixture overrides
# ----------------------------------------------------------------------
# The deprecated ``aiida.manage.tests.pytest_fixtures`` chain calls a
# removed ``Profile.clear_profile()`` during teardown; override with no-ops
# so tests that don't need an isolated DB aren't tripped. Tests that *do*
# need isolation should request ``aiida_profile_clean`` directly.


@pytest.fixture(scope="function")
def clear_database_after_test(aiida_profile: Any) -> Iterator[Any]:
    """Override the deprecated-and-broken upstream fixture with a no-op yield."""
    yield aiida_profile


@pytest.fixture(scope="function")
def clear_database(clear_database_after_test: Any) -> Iterator[None]:
    """Alias override for ``clear_database``."""
    yield


# ----------------------------------------------------------------------
# Codes + pseudos for dispatcher tests that build (but do not run) workgraphs
# ----------------------------------------------------------------------


@pytest.fixture
def localhost_computer(aiida_computer_local: Any) -> Any:
    """Return a computer whose label is literally ``localhost``.

    aiida-core's ``aiida_localhost`` fixture now suffixes its computer label
    with the pytest-xdist worker id (``localhost-master``, ...), but the
    dispatcher resolves codes as ``<name>@localhost`` — the label real
    profiles use — so the dummy codes must live on a literal one.
    """
    return aiida_computer_local(label="localhost")


@pytest.fixture
def localhost_code(localhost_computer: Any) -> Any:
    """Return a get-or-create factory for dummy codes on the literal ``localhost``.

    Unlike aiida-core's ``aiida_code_installed`` factory, the lookup matches
    label *and* computer — a same-labelled code another test left on a
    different computer (e.g. ``test_code_setup``'s ``pw`` on the suffixed
    ``aiida_localhost``) must not shadow the one the dispatcher resolves as
    ``<label>@localhost``.
    """
    from aiida.common.exceptions import NotExistent
    from aiida.orm import InstalledCode, load_code

    def factory(label: str, entry_point: str) -> Any:
        """Return the ``<label>@localhost`` code, creating it if absent."""
        try:
            return load_code(f"{label}@{localhost_computer.label}")
        except NotExistent:
            return InstalledCode(
                label=label,
                computer=localhost_computer,
                default_calc_job_plugin=entry_point,
                filepath_executable="/bin/true",
            ).store()

    return factory


@pytest.fixture
def installed_pw_code(localhost_code: Any) -> Any:
    """Register a dummy ``pw@localhost`` code so ``load_code`` succeeds."""
    return localhost_code("pw", "quantumespresso.pw")


@pytest.fixture
def installed_kcp_code(localhost_code: Any) -> Any:
    """Register a dummy ``kcp@localhost`` code so ``load_code`` succeeds."""
    return localhost_code("kcp", "koopmans.kcp")


@pytest.fixture
def installed_kcw_code(localhost_code: Any) -> Any:
    """Register a dummy ``kcw@localhost`` code so ``load_code`` succeeds."""
    return localhost_code("kcw", "koopmans.kcw_wann2kc")


@pytest.fixture
def installed_ph_code(localhost_code: Any) -> Any:
    """Register a dummy ``ph@localhost`` code so ``load_code`` succeeds."""
    return localhost_code("ph", "quantumespresso.ph")


@pytest.fixture
def installed_wannier_codes(localhost_code: Any) -> dict[str, Any]:
    """Register dummy ``wannier90`` / ``pw2wannier90`` codes for DFPT builds."""
    return {
        "wannier90": localhost_code("wannier90", "wannier90.wannier90"),
        "pw2wannier90": localhost_code("pw2wannier90", "quantumespresso.pw2wannier90"),
    }


@pytest.fixture
def installed_fold_codes(localhost_code: Any) -> dict[str, Any]:
    """Register dummy ``wann2kcp`` / ``merge_evc`` codes for the fold path."""
    return {
        "wann2kcp": localhost_code("wann2kcp", "koopmans.wann2kcp"),
        "merge_evc": localhost_code("merge_evc", "koopmans.merge_evc"),
    }


@pytest.fixture
def fake_sg15_pseudo_family(aiida_profile: Any) -> Any:
    """Install a minimal fake ``SG15/1.2/PBE/SR`` family (O and Si pseudos).

    This prevents ``ensure_pseudo_family_installed`` from hitting the network
    when the dispatcher builds the workgraph. Uses synthetic UPF streams —
    enough for ``UpfData`` validation, not physically meaningful pseudos.
    """
    from aiida.common.exceptions import NotExistent
    from aiida_pseudo.data.pseudo.upf import UpfData
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    label = "SG15/1.2/PBE/SR"
    try:
        return PseudoPotentialFamily.collection.get(label=label)
    except NotExistent:
        pass

    family = PseudoPotentialFamily(label=label, description="fake SG15 family for tests")
    family.store()
    for element, z_valence in (("O", 6.0), ("Si", 4.0)):
        content = (
            f'<UPF version="2.0.1"><PP_HEADER\nelement="{element}"\n'
            f'z_valence="{z_valence}"\n/></UPF>\n'
        )
        upf = UpfData(io.BytesIO(content.encode("utf-8")), filename=f"{element}.upf")
        family.add_nodes([upf.store()])
    return family


@pytest.fixture
def fake_sg15_cutoffs_family(aiida_profile: Any) -> Any:
    """Install a minimal fake ``SG15/1.0/PBE/SR`` cutoffs family (O and Si).

    Workgraph builders that call ``get_builder_from_protocol`` eagerly at
    build time (e.g. the ``dft_eps`` chain) need a family the aiida-qe
    protocol machinery accepts: SSSP, PseudoDojo, or a
    ``CutoffsPseudoPotentialFamily`` with recommended cutoffs — the plain
    ``PseudoPotentialFamily`` of ``fake_sg15_pseudo_family`` is not found by
    its query. Uses a different version label so both fixtures can coexist
    in one session profile.
    """
    from aiida.common.exceptions import NotExistent
    from aiida_pseudo.data.pseudo.upf import UpfData
    from aiida_pseudo.groups.family import CutoffsPseudoPotentialFamily

    label = "SG15/1.0/PBE/SR"
    try:
        return CutoffsPseudoPotentialFamily.collection.get(label=label)
    except NotExistent:
        pass

    family = CutoffsPseudoPotentialFamily(label=label)
    family.store()
    pseudos = []
    for element, z_valence in (("O", 6.0), ("Si", 4.0)):
        content = (
            f'<UPF version="2.0.1"><PP_HEADER\nelement="{element}"\n'
            f'z_valence="{z_valence}"\n/></UPF>\n'
        )
        upf = UpfData(io.BytesIO(content.encode("utf-8")), filename=f"{element}.upf")
        pseudos.append(upf.store())
    family.add_nodes(pseudos)
    family.set_cutoffs(
        {element: {"cutoff_wfc": 30.0, "cutoff_rho": 240.0} for element in ("O", "Si")},
        stringency="normal",
    )
    return family


# ----------------------------------------------------------------------
# WorkGraph → stable dict for snapshot regressions
# ----------------------------------------------------------------------


_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_SCRUB_KEYS = {
    "uuid",
    "pk",
    "id",
    "ctime",
    "mtime",
    "identifier",
    "remote_path",
    "computer",
    # Machine-specific / version-drift fields that aren't structural.
    "file_path",
    "package_version",
    "platform_version",
    "hash",
    "process",  # pyyaml block scalar of 'null\n...\n' varies by version
}


def _scrub(value: Any) -> Any:  # noqa: C901
    """Recursively replace non-deterministic fields (UUIDs, PKs, paths) with placeholders.

    Also turns AiiDA ``Node`` instances into stable ``<class:key>``
    placeholders (for ``Dict`` nodes we include the dict contents, with
    UUIDs scrubbed) so YAML serialisation of the WorkGraph dict doesn't
    trip over live ORM objects.

    Mirrors the intent of aiida-qe's ``serialize_builder``: diff structure,
    not volatile run metadata.
    """
    try:
        from aiida import orm
    except ImportError:  # pragma: no cover — aiida is always present here
        orm = None  # type: ignore[assignment]

    # Unwrap node_graph's TaggedValue (``wrapt.ObjectProxy``) so downstream
    # isinstance checks see the real underlying type.
    wrapped = getattr(value, "__wrapped__", None)
    if wrapped is not None and wrapped is not value:
        return _scrub(wrapped)

    if orm is not None:
        if isinstance(value, orm.Dict):
            return {"__aiida_dict__": _scrub(value.get_dict())}  # type: ignore[no-untyped-call]
        if isinstance(value, orm.StructureData):
            return {"__aiida_structure__": value.get_formula()}  # type: ignore[no-untyped-call]
        if isinstance(value, orm.AbstractCode):
            return {"__aiida_code__": value.full_label}
        if isinstance(value, orm.Node):
            return {"__aiida_node__": type(value).__name__}

    if isinstance(value, dict):
        return {
            key: (f"<scrubbed:{key}>" if key in _SCRUB_KEYS else _scrub(val))
            for key, val in value.items()
        }
    if isinstance(value, list | tuple):
        scrubbed = [_scrub(v) for v in value]
        # ``WorkGraph.to_dict()`` collects some namespaces by iterating
        # an unordered dict, so two runs of the same build can emit the
        # same list of port names in different orders. Sort lists of
        # plain strings to make the snapshot stable; lists holding
        # structured items (dicts, sub-lists) are left in place — their
        # order tends to carry semantic meaning (e.g. socket-connection
        # order).
        if scrubbed and all(isinstance(v, str) for v in scrubbed):
            scrubbed.sort()
        return scrubbed
    if isinstance(value, str):
        return _UUID_RE.sub("<uuid>", value)
    # node-graph socket objects (``SocketAny`` etc.) sometimes appear in
    # the serialised workgraph payload when one ``@task.graph``'s output
    # is wired into another's input. YAML can't represent them; collapse
    # to a stable placeholder so ``data_regression`` works.
    type_name = type(value).__name__
    if type_name.startswith("Socket"):
        return f"<{type_name}>"
    return value


@pytest.fixture
def serialize_workgraph() -> Callable[..., dict[str, Any]]:
    """Return a callable that serializes a ``WorkGraph`` into a stable dict.

    The returned dict records the task name list, the task count, and a
    scrubbed version of ``WorkGraph.to_dict()`` (UUIDs / PKs / paths
    replaced with stable placeholders). Suitable for ``data_regression``.
    """
    from aiida_workgraph import WorkGraph

    def _serialize(workgraph: WorkGraph) -> dict[str, Any]:
        raw = workgraph.to_dict()
        task_names = sorted(workgraph.get_task_names())
        return {
            "workgraph_name": workgraph.name,
            "task_names": task_names,
            "n_tasks": len(task_names),
            "raw": _scrub(raw),
        }

    return _serialize
