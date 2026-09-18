"""``koopmans fetch``: writing the results tree for an already-submitted run.

Exercised end to end against a real process node built with
:func:`tests.fixtures.make_process` -- the same "no broad AiiDA mocking"
approach ``tests/test_cli_submit_status.py`` uses for ``status``/``attach``
-- so ``dump_workgraph`` runs for real against the database, not a stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from koopmans.aiida.anchor import AnchorEntry, append_anchor_entry
from koopmans.aiida.setup.profile import PROFILE_NAME
from koopmans.cli import cli
from tests.fixtures import make_process, skip_profile_loading


def _anchor(
    tmp_path: Path, node: Any, name: str = "si.run.yaml", input_name: str = "si.yaml"
) -> Path:
    """Record ``node`` in a fresh anchor file, returning the anchor's path."""
    anchor_path = tmp_path / name
    append_anchor_entry(
        anchor_path,
        AnchorEntry(
            uuid=node.uuid,
            pk=node.pk,
            input=input_name,
            profile=PROFILE_NAME,
            submitted="2026-08-11T12:00:00+00:00",
        ),
    )
    return anchor_path


class TestFetch:
    """``koopmans fetch``: write the results tree a `koopmans run` would have written."""

    def test_a_finished_run_writes_the_tree_next_to_the_input_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aiida_profile_clean: Any
    ) -> None:
        """The dump lands at `<input stem>`, beside the input file the anchor names."""
        skip_profile_loading(monkeypatch)
        node = make_process(process_label="WorkGraph<Tiny>")
        anchor_path = _anchor(tmp_path, node)

        result = CliRunner().invoke(cli, ["fetch", str(anchor_path)])

        assert result.exit_code == 0, result.output
        dump_path = tmp_path / "si"
        assert dump_path.is_dir()
        assert f"Wrote {dump_path}" in result.output

    def test_a_running_calculation_writes_a_partial_tree_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aiida_profile_clean: Any
    ) -> None:
        """A calculation still running writes what has finished, with a notice."""
        skip_profile_loading(monkeypatch)
        node = make_process(process_label="WorkGraph<Tiny>", process_state="running")
        anchor_path = _anchor(tmp_path, node)

        result = CliRunner().invoke(cli, ["fetch", str(anchor_path)])

        assert result.exit_code == 0, result.output
        dump_path = tmp_path / "si"
        assert dump_path.is_dir()
        assert "has not finished yet" in result.output

    def test_a_failed_root_exits_nonzero_and_names_the_step(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aiida_profile_clean: Any
    ) -> None:
        """A failed root process still gets its tree written, and the failure is shown.

        Reuses `render_process_once`, the same rendering `koopmans status`
        uses, so the failing step is named the same way there.
        """
        skip_profile_loading(monkeypatch)
        node = make_process(
            process_label="WorkGraph<Tiny>", exit_status=1, exit_message="the graph failed"
        )
        anchor_path = _anchor(tmp_path, node)

        result = CliRunner().invoke(cli, ["fetch", str(anchor_path)])

        assert result.exit_code != 0
        assert "the graph failed" in result.output
        dump_path = tmp_path / "si"
        assert dump_path.is_dir()
        assert f"Wrote {dump_path}" in result.output

    def test_a_refetch_overwrites_the_existing_tree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aiida_profile_clean: Any
    ) -> None:
        """Fetching twice leaves one tree behind, not stale files from the first."""
        skip_profile_loading(monkeypatch)
        node = make_process(process_label="WorkGraph<Tiny>")
        anchor_path = _anchor(tmp_path, node)

        dump_path = tmp_path / "si"
        dump_path.mkdir()
        stale_file = dump_path / "stale-from-a-previous-fetch.txt"
        stale_file.write_text("leftover")

        result = CliRunner().invoke(cli, ["fetch", str(anchor_path)])

        assert result.exit_code == 0, result.output
        assert dump_path.is_dir()
        assert not stale_file.exists()

    def test_uuid_with_no_run_file_dumps_beside_the_process_label(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aiida_profile_clean: Any
    ) -> None:
        """`--uuid` with no run file writes into `./<process label>`, in the cwd."""
        skip_profile_loading(monkeypatch)
        monkeypatch.chdir(tmp_path)
        node = make_process(process_label="WorkGraph<Direct>")

        result = CliRunner().invoke(cli, ["fetch", "--uuid", node.uuid])

        assert result.exit_code == 0, result.output
        dump_path = tmp_path / "WorkGraph<Direct>"
        assert dump_path.is_dir()
        assert f"Wrote {dump_path}" in result.output

    def test_pk_with_no_run_file_and_no_process_label_dumps_beside_the_pk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aiida_profile_clean: Any
    ) -> None:
        """A process with no label falls back to its pk, still inside the cwd."""
        skip_profile_loading(monkeypatch)
        monkeypatch.chdir(tmp_path)
        node = make_process()

        result = CliRunner().invoke(cli, ["fetch", "--pk", str(node.pk)])

        assert result.exit_code == 0, result.output
        dump_path = tmp_path / str(node.pk)
        assert dump_path.is_dir()

    def test_target_resolution_errors_match_status(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No target and no run file fails identically to `koopmans status`."""
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path))

        result = CliRunner().invoke(cli, ["fetch"])

        assert result.exit_code != 0
        assert "run.yaml" in result.output


class TestAttachDumps:
    """`koopmans attach` writes the tree once the target has terminated."""

    def test_attaching_to_an_already_finished_run_writes_the_tree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aiida_profile_clean: Any
    ) -> None:
        """An already-terminated target is dumped immediately, no watch loop needed."""
        skip_profile_loading(monkeypatch)
        node = make_process(process_label="WorkGraph<Tiny>")
        anchor_path = _anchor(tmp_path, node)

        result = CliRunner().invoke(cli, ["attach", str(anchor_path)])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "si").is_dir()
