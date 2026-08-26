"""``koopmans submit``/``status``/``attach``: the CLI layer over anchor files.

``submit`` is exercised up to the daemon hand-off: ``read_input_file`` and
``build_workgraph`` run for real (a minimal ``dft_bands`` input against
throwaway pw.x/pseudo fixtures), and only ``koopmans.api.launch`` — the
one function that hands a workgraph to the daemon — is stubbed, per the
project's "no broad AiiDA mocking" rule. ``status``/``attach`` are
exercised end to end against a real, already-terminated process built
with :func:`tests.fixtures.make_process`, so what's under test is the
whole path from a run file (or ``--uuid``/``--pk``) to the printed output
and exit code. Neither command's live-daemon behaviour (a real submit, or
``attach`` catching a still-running process) is covered here — see the PR
description.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from koopmans.aiida.anchor import AnchorEntry, append_anchor_entry, read_anchor_entries
from koopmans.aiida.setup.profile import PROFILE_NAME
from koopmans.cli import cli
from tests.fixtures import make_process, skip_profile_loading, write_koopmans_input


class FakeProcessNode:
    """The minimal shape ``koopmans.api.launch`` hands back: pk, uuid, label."""

    def __init__(
        self,
        pk: int = 999,
        uuid: str = "fake-process-uuid",
        label: str = "WorkGraph<DftBands>",
    ) -> None:
        """Set the pk, uuid, and process label ``koopmans.cli.submit`` reads back."""
        self.pk = pk
        self.uuid = uuid
        self.process_label = label


class TestSubmit:
    """The command up to (not including) the real daemon hand-off."""

    def _invoke(
        self,
        monkeypatch: pytest.MonkeyPatch,
        input_path: Path,
        fake_node: FakeProcessNode,
    ) -> tuple[Any, dict[str, Any]]:
        """Invoke ``submit``, returning the CLI result and the launch() kwargs it saw."""
        skip_profile_loading(monkeypatch)
        captured: dict[str, Any] = {}

        def _fake_launch(wg: Any, *, blocking: bool, wait: bool = False) -> FakeProcessNode:
            captured["blocking"] = blocking
            captured["wait"] = wait
            return fake_node

        monkeypatch.setattr("koopmans.api.launch", _fake_launch)
        result = CliRunner().invoke(cli, ["submit", str(input_path)])
        return result, captured

    def test_a_successful_submission_hands_off_without_blocking(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """The submitted process runs asynchronously: blocking=False, wait=False."""
        input_path = write_koopmans_input(tmp_path)
        fake_node = FakeProcessNode()

        result, captured = self._invoke(monkeypatch, input_path, fake_node)

        assert result.exit_code == 0, result.output
        assert captured == {"blocking": False, "wait": False}

    def test_the_output_is_one_clean_line(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """The user sees one line; the identifiers live in the run file."""
        input_path = write_koopmans_input(tmp_path)
        fake_node = FakeProcessNode(pk=42, uuid="abc-123", label="WorkGraph<DftBands>")

        result, _ = self._invoke(monkeypatch, input_path, fake_node)

        assert "Workflow submitted" in result.output
        assert "abc-123" not in result.output
        assert "42" not in result.output

    def test_the_anchor_entry_matches_the_submitted_node(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """The written entry's fields match the (stubbed) submitted node."""
        input_path = write_koopmans_input(tmp_path)
        fake_node = FakeProcessNode(pk=42, uuid="abc-123")

        self._invoke(monkeypatch, input_path, fake_node)

        anchor_path = tmp_path / "si.run.yaml"
        entries = read_anchor_entries(anchor_path)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.uuid == "abc-123"
        assert entry.pk == 42
        assert entry.input == "si.yaml"
        assert entry.profile == PROFILE_NAME
        # Round-trips as a real timestamp; raises otherwise.
        datetime.fromisoformat(entry.submitted)

    def test_a_failed_anchor_write_still_hands_back_the_recovery_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """The daemon already has the job; a run-file write failure must not hide it.

        A read-only directory makes ``append_anchor_entry`` raise
        ``OSError`` after the (stubbed) launch has already "happened".
        The message hands back the whole ``koopmans status --uuid``
        command, which is what finds the process again; the pk names
        nothing the user can type and is absent.
        """
        import os
        import stat

        input_path = write_koopmans_input(tmp_path)
        fake_node = FakeProcessNode(pk=4242, uuid="recoverable-uuid")
        skip_profile_loading(monkeypatch)
        monkeypatch.setattr("koopmans.api.launch", lambda wg, *, blocking, wait=False: fake_node)

        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
        try:
            result = CliRunner().invoke(cli, ["submit", str(input_path)])
        finally:
            os.chmod(tmp_path, stat.S_IRWXU)

        assert result.exit_code != 0
        assert "koopmans status --uuid recoverable-uuid" in result.output
        assert "4242" not in result.output
        assert not (tmp_path / "si.run.yaml").exists()

    def test_a_resubmission_appends_rather_than_overwrites(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """Two submissions from the same input file leave both entries behind."""
        input_path = write_koopmans_input(tmp_path)

        self._invoke(monkeypatch, input_path, FakeProcessNode(pk=1, uuid="first"))
        self._invoke(monkeypatch, input_path, FakeProcessNode(pk=2, uuid="second"))

        entries = read_anchor_entries(tmp_path / "si.run.yaml")
        assert [e.uuid for e in entries] == ["first", "second"]


class TestStatus:
    """``koopmans status``: one-shot render plus exit code."""

    def test_a_successful_run_exits_zero_and_shows_its_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aiida_profile_clean: Any
    ) -> None:
        """A finished-ok run's anchor file resolves to a zero-exit, rendered status."""
        skip_profile_loading(monkeypatch)
        node = make_process(process_label="WorkGraph<Tiny>")
        append_anchor_entry(
            tmp_path / "si.run.yaml",
            AnchorEntry(
                uuid=node.uuid,
                pk=node.pk,
                input="si.yaml",
                profile=PROFILE_NAME,
                submitted="2026-08-11T12:00:00+00:00",
            ),
        )

        result = CliRunner().invoke(cli, ["status", str(tmp_path / "si.run.yaml")])

        assert result.exit_code == 0, result.output
        assert "Tiny" in result.output
        assert "completed successfully" in result.output

    def test_a_failed_run_exits_nonzero_and_shows_the_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aiida_profile_clean: Any
    ) -> None:
        """The root process failing is both reported and reflected in the exit code."""
        skip_profile_loading(monkeypatch)
        node = make_process(
            process_label="WorkGraph<Tiny>", exit_status=1, exit_message="the graph failed"
        )
        anchor_path = tmp_path / "si.run.yaml"
        append_anchor_entry(
            anchor_path,
            AnchorEntry(
                uuid=node.uuid,
                pk=node.pk,
                input="si.yaml",
                profile=PROFILE_NAME,
                submitted="2026-08-11T12:00:00+00:00",
            ),
        )

        result = CliRunner().invoke(cli, ["status", str(anchor_path)])

        assert result.exit_code != 0
        assert "the graph failed" in result.output

    def test_direct_uuid_bypasses_the_run_file(
        self, monkeypatch: pytest.MonkeyPatch, aiida_profile_clean: Any
    ) -> None:
        """--uuid loads the node directly; no anchor file is needed at all."""
        skip_profile_loading(monkeypatch)
        node = make_process(process_label="WorkGraph<Direct>")

        result = CliRunner().invoke(cli, ["status", "--uuid", node.uuid])

        assert result.exit_code == 0, result.output
        assert "Direct" in result.output

    def test_direct_pk_bypasses_the_run_file(
        self, monkeypatch: pytest.MonkeyPatch, aiida_profile_clean: Any
    ) -> None:
        """--pk loads the node directly; no anchor file is needed at all."""
        skip_profile_loading(monkeypatch)
        node = make_process(process_label="WorkGraph<ByPk>")

        result = CliRunner().invoke(cli, ["status", "--pk", str(node.pk)])

        assert result.exit_code == 0, result.output
        # "ByPk" is not a name the display table knows, so it is shown
        # exactly as written rather than guessed at.
        assert "ByPk" in result.output

    def test_a_target_that_is_not_a_process_is_named_as_the_user_named_it(
        self, monkeypatch: pytest.MonkeyPatch, aiida_profile_clean: Any
    ) -> None:
        """The rejection quotes the identifier given, not the node's pk.

        A run file records a uuid, and ``--uuid`` takes one, so a pk read
        off the node names it by something the user never typed and
        cannot type back.
        """
        from aiida import orm

        skip_profile_loading(monkeypatch)
        data = orm.Dict(dict={"not": "a process"})  # type: ignore[no-untyped-call]
        data.store()

        result = CliRunner().invoke(cli, ["status", "--uuid", data.uuid])

        assert result.exit_code != 0
        assert f"'{data.uuid}' is not a calculation" in result.output
        assert "Dict" in result.output
        assert f"Node {data.pk}" not in result.output

    def test_a_deleted_node_is_a_clean_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aiida_profile_clean: Any
    ) -> None:
        """The anchor names a node that no longer exists in the profile."""
        from aiida.tools import delete_nodes

        skip_profile_loading(monkeypatch)
        node = make_process(process_label="WorkGraph<Tiny>")
        node_pk, node_uuid = node.pk, node.uuid
        anchor_path = tmp_path / "si.run.yaml"
        append_anchor_entry(
            anchor_path,
            AnchorEntry(
                uuid=node_uuid,
                pk=node_pk,
                input="si.yaml",
                profile=PROFILE_NAME,
                submitted="2026-08-11T12:00:00+00:00",
            ),
        )
        delete_nodes([node_pk], dry_run=False)

        result = CliRunner().invoke(cli, ["status", str(anchor_path)])

        assert result.exit_code != 0
        assert node_uuid in result.output
        assert "deleted" in result.output

    def test_an_entry_for_a_foreign_profile_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A run file naming a different profile than this install's own is refused.

        ``AnchorEntry.profile`` is typed ``Literal[PROFILE_NAME]``, so
        pydantic itself rejects a foreign value at validation — there is
        no separate guard to test, only that its error reaches the user
        naming both the value found and the one required.
        """
        anchor_path = tmp_path / "si.run.yaml"
        anchor_path.write_text(
            yaml.safe_dump(
                [
                    {
                        "uuid": "some-uuid",
                        "pk": 1,
                        "input": "si.yaml",
                        "profile": "someone-elses-profile",
                        "submitted": "2026-08-11T12:00:00+00:00",
                    }
                ]
            )
        )

        result = CliRunner().invoke(cli, ["status", str(anchor_path)])

        assert result.exit_code != 0
        assert "someone-elses-profile" in result.output
        assert PROFILE_NAME in result.output

    def test_no_target_and_no_run_file_is_a_clean_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No AiiDA profile is touched — the error comes from resolution alone."""
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path))

        result = CliRunner().invoke(cli, ["status"])

        assert result.exit_code != 0
        assert "run.yaml" in result.output


class TestAttach:
    """``koopmans attach``: degrades to the status render for a terminated process."""

    def test_an_already_terminated_process_renders_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aiida_profile_clean: Any
    ) -> None:
        """A terminated target is shown once, exactly as `status` would show it."""
        skip_profile_loading(monkeypatch)
        node = make_process(process_label="WorkGraph<Tiny>")
        anchor_path = tmp_path / "si.run.yaml"
        append_anchor_entry(
            anchor_path,
            AnchorEntry(
                uuid=node.uuid,
                pk=node.pk,
                input="si.yaml",
                profile=PROFILE_NAME,
                submitted="2026-08-11T12:00:00+00:00",
            ),
        )

        result = CliRunner().invoke(cli, ["attach", str(anchor_path)])

        assert result.exit_code == 0, result.output
        assert "Tiny" in result.output
        assert "completed successfully" in result.output
