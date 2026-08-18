"""What ``koopmans run`` prints once the calculation is over.

The run itself is not what is under test here: ``read_input_file`` and
``build_workgraph`` run for real (a minimal ``dft_bands`` input against
throwaway pw.x/pseudo fixtures), while the two steps that need a daemon
or a real dump — ``run_with_progress`` and ``dump_workgraph`` — are
stubbed at the CLI boundary, the same way ``tests/test_cli_submit_status``
stubs ``koopmans.api.launch``. The process the stub hands back is a real
stored node carrying a real ``model`` ``Dict`` output, so
``trained_model_output`` runs against the database rather than a fake.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from click.testing import CliRunner

import koopmans.cli as cli_mod
from koopmans.cli import cli
from tests.fixtures import make_process, skip_profile_loading, write_koopmans_input

if TYPE_CHECKING:
    from pathlib import Path


def _process_with_model_output(model: dict[str, Any] | None) -> Any:
    """Return a finished process node, carrying a ``model`` Dict if given."""
    from aiida import orm
    from aiida.common.links import LinkType

    node = make_process(process_label="WorkGraph<Trained>")
    if model is not None:
        data = orm.Dict(dict=model)  # type: ignore[no-untyped-call]
        data.store()
        data.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label="model")
    return node


def _run(
    monkeypatch: pytest.MonkeyPatch, input_path: Path, process: Any
) -> tuple[Any, dict[str, Any]]:
    """Invoke ``koopmans run``, returning the CLI result and the dump kwargs."""
    skip_profile_loading(monkeypatch)
    captured: dict[str, Any] = {}

    def _fake_run_with_progress(wg: Any, refresh_interval: float = 2.0) -> None:
        wg.process = process

    def _fake_dump(process: Any, output_path: Path, overwrite: bool = True) -> Path:
        captured["output_path"] = output_path
        return output_path

    monkeypatch.setattr(cli_mod, "run_with_progress", _fake_run_with_progress)
    monkeypatch.setattr(cli_mod, "dump_workgraph", _fake_dump)
    result = CliRunner().invoke(cli, ["run", str(input_path)])
    return result, captured


class TestTrainedModelMessage:
    """The closing line of a ``ml: {mode: train}`` run."""

    def test_it_names_the_written_file_and_the_keyword_that_reads_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """The user is handed a path and the input-file line that reuses it."""
        input_path = write_koopmans_input(tmp_path)
        process = _process_with_model_output({"correction": "ki"})

        result, captured = _run(monkeypatch, input_path, process)

        assert result.exit_code == 0, result.output
        # The path named is the one the dump actually wrote into.
        assert captured["output_path"] == tmp_path / "si"
        assert str(tmp_path / "si" / "model.json") in result.output
        assert "ml: {model_file: si/model.json}" in result.output

    def test_it_names_no_node_identifier(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """Neither the model node's pk nor its uuid reaches the terminal.

        Both identify the model as well as the file does, and neither is
        anything a reader without AiiDA can use. The line is pinned
        whole, so an identifier put back on it fails here.
        """
        input_path = write_koopmans_input(tmp_path)
        process = _process_with_model_output({"correction": "ki"})
        model_node = process.outputs.model

        result, _ = _run(monkeypatch, input_path, process)

        assert model_node.uuid not in result.output
        assert result.output.strip().splitlines()[-1] == (
            f"Trained model written to {tmp_path / 'si' / 'model.json'} — reuse it from "
            "an input file beside si.yaml with `ml: {model_file: si/model.json}`."
        )

    def test_a_run_that_trained_nothing_says_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        aiida_profile_clean: Any,
        installed_pw_code: Any,
        fake_sg15_pseudo_family: Any,
    ) -> None:
        """A run with no ``model`` output prints no model line at all."""
        input_path = write_koopmans_input(tmp_path)
        process = _process_with_model_output(None)

        result, _ = _run(monkeypatch, input_path, process)

        assert result.exit_code == 0, result.output
        assert "model.json" not in result.output
