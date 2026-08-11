"""Anchor-file read/write and status/attach target resolution.

Pure filesystem and YAML logic — no AiiDA profile touched, matching
``resolve_target``'s own contract of only raising ``ValueError`` on
ambiguity or a missing file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from koopmans.aiida.anchor import (
    AnchorEntry,
    ResolvedTarget,
    anchor_path_for_input,
    append_anchor_entry,
    newest_anchor_entry,
    read_anchor_entries,
    resolve_target,
)


def _entry(uuid: str = "uuid-1", pk: int = 1, input_name: str = "si.yaml") -> AnchorEntry:
    return {
        "uuid": uuid,
        "pk": pk,
        "input": input_name,
        "profile": "koopmans",
        "submitted": "2026-08-11T12:00:00+00:00",
    }


class TestAnchorPathForInput:
    """The anchor file sits next to the input file, named off its stem."""

    @pytest.mark.parametrize(
        ("input_name", "expected_name"),
        [
            ("si.yaml", "si.run.yaml"),
            ("si.json", "si.run.yaml"),
            ("si", "si.run.yaml"),
        ],
    )
    def test_the_extension_is_replaced_regardless_of_input_format(
        self, tmp_path: Path, input_name: str, expected_name: str
    ) -> None:
        """A JSON or extensionless input still gets a `.run.yaml` anchor."""
        assert anchor_path_for_input(tmp_path / input_name) == tmp_path / expected_name

    def test_a_nested_input_keeps_its_directory(self, tmp_path: Path) -> None:
        """The anchor is a sibling, not dropped into the cwd."""
        nested = tmp_path / "runs" / "si.yaml"
        assert anchor_path_for_input(nested) == tmp_path / "runs" / "si.run.yaml"


class TestReadAndAppend:
    """Entries accumulate; nothing already recorded is ever overwritten."""

    def test_reading_a_missing_file_is_an_empty_list(self, tmp_path: Path) -> None:
        """No anchor file yet is read as no submissions, not an error."""
        assert read_anchor_entries(tmp_path / "missing.run.yaml") == []

    def test_reading_an_empty_file_is_an_empty_list(self, tmp_path: Path) -> None:
        """An anchor file with nothing but whitespace parses as no entries."""
        anchor = tmp_path / "si.run.yaml"
        anchor.write_text("")
        assert read_anchor_entries(anchor) == []

    def test_a_non_list_anchor_file_is_rejected(self, tmp_path: Path) -> None:
        """A hand-edited anchor file that lost its list shape fails loudly."""
        anchor = tmp_path / "si.run.yaml"
        anchor.write_text(yaml.safe_dump({"uuid": "not-a-list"}))
        with pytest.raises(ValueError, match="does not hold a list"):
            read_anchor_entries(anchor)

    def test_append_creates_the_file(self, tmp_path: Path) -> None:
        """Appending to a nonexistent anchor file creates it."""
        anchor = tmp_path / "si.run.yaml"
        append_anchor_entry(anchor, _entry())
        assert read_anchor_entries(anchor) == [_entry()]

    def test_append_keeps_earlier_entries_and_orders_newest_last(self, tmp_path: Path) -> None:
        """A resubmission appends; it does not replace the record of the first run."""
        anchor = tmp_path / "si.run.yaml"
        append_anchor_entry(anchor, _entry(uuid="uuid-1", pk=1))
        append_anchor_entry(anchor, _entry(uuid="uuid-2", pk=2))

        entries = read_anchor_entries(anchor)

        assert [e["uuid"] for e in entries] == ["uuid-1", "uuid-2"]

    def test_newest_entry_is_the_last_one_appended(self, tmp_path: Path) -> None:
        """``newest_anchor_entry`` is the last-appended entry, not the first."""
        anchor = tmp_path / "si.run.yaml"
        append_anchor_entry(anchor, _entry(uuid="uuid-1", pk=1))
        append_anchor_entry(anchor, _entry(uuid="uuid-2", pk=2))

        assert newest_anchor_entry(anchor)["uuid"] == "uuid-2"

    def test_newest_entry_on_an_empty_anchor_raises(self, tmp_path: Path) -> None:
        """An anchor file with an empty list has no newest entry to return."""
        anchor = tmp_path / "si.run.yaml"
        anchor.write_text(yaml.safe_dump([]))
        with pytest.raises(ValueError, match="no recorded submissions"):
            newest_anchor_entry(anchor)


class TestResolveTarget:
    """Turning a status/attach argument into a process identity."""

    def test_uuid_option_wins_outright(self, tmp_path: Path) -> None:
        """--uuid is honoured even with a target and no anchor file present."""
        resolved = resolve_target("si.yaml", uuid="direct-uuid", pk=None, cwd=tmp_path)
        assert resolved == ResolvedTarget(uuid="direct-uuid", pk=None)

    def test_pk_option_wins_outright(self, tmp_path: Path) -> None:
        """--pk is honoured even with no target and no anchor file present."""
        resolved = resolve_target(None, uuid=None, pk=7, cwd=tmp_path)
        assert resolved == ResolvedTarget(uuid=None, pk=7)

    def test_uuid_and_pk_together_is_rejected(self, tmp_path: Path) -> None:
        """Passing both flags at once is ambiguous, not a silent precedence choice."""
        with pytest.raises(ValueError, match="only one of --uuid or --pk"):
            resolve_target(None, uuid="u", pk=1, cwd=tmp_path)

    def test_no_target_and_no_anchor_file_is_an_error(self, tmp_path: Path) -> None:
        """With nothing to resolve against, the error points at `koopmans submit`."""
        with pytest.raises(ValueError, match=r"No \*\.run\.yaml file found"):
            resolve_target(None, uuid=None, pk=None, cwd=tmp_path)

    def test_no_target_and_one_anchor_file_uses_its_newest_entry(self, tmp_path: Path) -> None:
        """The single anchor file in cwd is found without being named."""
        anchor = tmp_path / "si.run.yaml"
        append_anchor_entry(anchor, _entry(uuid="uuid-1", pk=1))
        append_anchor_entry(anchor, _entry(uuid="uuid-2", pk=2))

        resolved = resolve_target(None, uuid=None, pk=None, cwd=tmp_path)

        assert resolved == ResolvedTarget(uuid="uuid-2", pk=2)

    def test_no_target_and_several_anchor_files_is_an_error_listing_them(
        self, tmp_path: Path
    ) -> None:
        """Ambiguity between run files is reported, not guessed at."""
        append_anchor_entry(tmp_path / "si.run.yaml", _entry())
        append_anchor_entry(tmp_path / "zno.run.yaml", _entry())

        with pytest.raises(ValueError) as excinfo:
            resolve_target(None, uuid=None, pk=None, cwd=tmp_path)

        assert "si.run.yaml" in str(excinfo.value)
        assert "zno.run.yaml" in str(excinfo.value)

    def test_a_run_file_target_uses_its_newest_entry(self, tmp_path: Path) -> None:
        """Naming the anchor file directly resolves to its newest submission."""
        anchor = tmp_path / "si.run.yaml"
        append_anchor_entry(anchor, _entry(uuid="uuid-1", pk=1))
        append_anchor_entry(anchor, _entry(uuid="uuid-2", pk=2))

        resolved = resolve_target("si.run.yaml", uuid=None, pk=None, cwd=tmp_path)

        assert resolved == ResolvedTarget(uuid="uuid-2", pk=2)

    def test_an_input_file_target_reads_its_sibling_anchor(self, tmp_path: Path) -> None:
        """Naming the input file resolves through its `.run.yaml` sibling."""
        append_anchor_entry(tmp_path / "si.run.yaml", _entry(uuid="uuid-1", pk=1))

        resolved = resolve_target("si.yaml", uuid=None, pk=None, cwd=tmp_path)

        assert resolved == ResolvedTarget(uuid="uuid-1", pk=1)

    def test_an_input_file_target_with_no_anchor_yet_is_an_error(self, tmp_path: Path) -> None:
        """An input file that was never submitted has no sibling anchor to read."""
        with pytest.raises(ValueError, match="No run file found"):
            resolve_target("si.yaml", uuid=None, pk=None, cwd=tmp_path)

    def test_a_target_given_as_a_path_is_read_relative_to_itself_not_cwd(
        self, tmp_path: Path
    ) -> None:
        """A target given with its own directory does not need to match cwd."""
        subdir = tmp_path / "runs"
        subdir.mkdir()
        append_anchor_entry(subdir / "si.run.yaml", _entry(uuid="uuid-1", pk=1))

        resolved = resolve_target(str(subdir / "si.yaml"), uuid=None, pk=None, cwd=tmp_path)

        assert resolved == ResolvedTarget(uuid="uuid-1", pk=1)
