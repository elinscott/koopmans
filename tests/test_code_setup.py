"""Tests for AiiDA code registration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest


def _workflows_source() -> str:
    """Concatenate the workflows package's module sources for call-site scans."""
    import koopmans.aiida.workflows as workflows

    return "".join(
        path.read_text() for path in sorted(Path(workflows.__file__).parent.glob("*.py"))
    )


class TestExecutableCoverage:
    """Every code the dispatcher can load must be registrable."""

    def test_dispatched_codes_have_executable_entries(self) -> None:
        """Each ``load_code`` executable is a key in ``QE_EXECUTABLES``."""
        from koopmans.aiida.setup.codes import QE_EXECUTABLES

        source = _workflows_source()
        executables = set(re.findall(r'(?<!\.)\bload_code\(\s*"[^"]+"\s*,\s*"([^"]+)"', source))

        assert executables, "regex matched no load_code call sites"
        # Codes registered outside the QE install scan (their second argument
        # is a human-readable name, not an executable on PATH).
        non_qe = {name for name in executables if not name.endswith(".x")}
        assert non_qe <= {"julia (Wannier.jl)"}, f"unexpected non-QE load sites: {non_qe}"
        missing = (executables - non_qe) - set(QE_EXECUTABLES)
        assert not missing, f"dispatcher loads {missing} with no QE_EXECUTABLES entry"

    def test_dispatched_labels_are_registered(self) -> None:
        """Each ``load_code`` label is a label the installer registers."""
        from koopmans.aiida.setup.codes import code_specs

        source = _workflows_source()
        labels = set(re.findall(r'(?<!\.)\bload_code\(\s*"([^"]+)"\s*,\s*"[^"]+"', source))

        assert labels, "regex matched no load_code call sites"
        # ``wannierjl`` is registered by its own plugin helper, not the QE scan.
        missing = labels - set(code_specs()) - {"wannierjl"}
        assert not missing, f"dispatcher loads {missing} with no registration entry"

    def test_no_non_literal_load_code_calls(self) -> None:
        """Every ``load_code`` call passes two string literals.

        The coverage test above only sees literal arguments; a call built
        from variables would escape it and could load an unregistered code.
        """
        source = _workflows_source()
        all_calls = len(re.findall(r"(?<!\.)\bload_code\((?!\s*self)", source))
        literal_calls = len(re.findall(r'load_code\(\s*"[^"]+"\s*,\s*"[^"]+"', source))
        definitions = len(re.findall(r"def load_code\(", source))

        assert all_calls - definitions == literal_calls, (
            "a load_code call site uses non-literal arguments and escapes "
            "the executable-coverage test"
        )

    def test_load_code_labels_match_executables(self) -> None:
        """Each ``load_code`` pair matches how that label is registered.

        Registration pairs a label with an executable, so a call site that
        names a different executable for a label would pass the coverage
        test yet load a code built from the wrong binary.
        """
        from koopmans.aiida.setup.codes import code_specs

        source = _workflows_source()
        pairs = re.findall(r'(?<!\.)\bload_code\(\s*"([^"]+)"\s*,\s*"([^"]+)"', source)
        specs = code_specs()

        assert pairs, "regex matched no load_code call sites"
        mismatched = [
            (label, executable)
            for label, executable in pairs
            if executable.endswith(".x") and specs.get(label, (None,))[0] != executable
        ]
        assert not mismatched, f"label/executable mismatch at call sites: {mismatched}"


class TestForcedCodeReinstall:
    """Forced reinstalls must not collide on the retired ``<label>_old`` label."""

    def test_repeated_force_reinstall_uniquifies_old_labels(
        self, aiida_profile_clean: Any, aiida_localhost: Any, tmp_path: Any
    ) -> None:
        """Two forced reinstalls retire two codes with distinct labels."""
        from aiida import orm

        from koopmans.aiida.setup.codes import setup_code

        exe = tmp_path / "pw.x"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)

        for _ in range(3):
            setup_code("pw.x", str(exe), "quantumespresso.pw", aiida_localhost, force=True)

        labels = {
            code.label
            for (code,) in orm.QueryBuilder().append(orm.InstalledCode).iterall()
            if code.label.startswith("pw")
        }
        assert labels == {"pw", "pw_old", "pw_old2"}


class TestLabelKeyedRegistration:
    """The install scan is keyed by code label, not executable name."""

    def test_registered_labels_split_from_missing(
        self, aiida_profile_clean: Any, aiida_localhost: Any, tmp_path: Any
    ) -> None:
        """An already-registered label lands in existing, the rest in to-find."""
        from koopmans.aiida.setup.codes import get_codes_to_register, setup_code

        exe = tmp_path / "pw.x"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        setup_code("pw.x", str(exe), "quantumespresso.pw", aiida_localhost)

        existing, to_find = get_codes_to_register(aiida_localhost)
        assert existing == ["pw"]
        assert "pw" not in to_find
        assert to_find["pw2wannier90"] == ("pw2wannier90.x", "quantumespresso.pw2wannier90")

    def test_explicit_path_wins_over_the_scan(
        self, aiida_profile_clean: Any, aiida_localhost: Any, tmp_path: Any
    ) -> None:
        """An explicit per-label path registers that binary; absent ones report missing."""
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import scan_and_register_codes

        exe = tmp_path / "special_pw2wannier90.x"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)

        found, missing = scan_and_register_codes(
            {
                "pw2wannier90": ("pw2wannier90.x", "quantumespresso.pw2wannier90"),
                "kcp": ("definitely_not_on_path.x", "koopmans.kcp"),
            },
            aiida_localhost,
            explicit_codes={"pw2wannier90": str(exe)},
        )
        assert found == ["pw2wannier90"]
        assert missing == ["kcp"]
        code = load_code(f"pw2wannier90@{aiida_localhost.label}")
        assert str(code.filepath_executable) == str(exe)


class TestVariantCodes:
    """A variant build is registered from an explicit path, never from PATH."""

    def test_path_scan_leaves_a_variant_unregistered(
        self, aiida_profile_clean: Any, aiida_localhost: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """A stock binary on PATH does not get registered under the variant label.

        The variant's executable name is the stock one, so a scan that
        honoured PATH here would register a binary that cannot run the mode
        the label promises.
        """
        from koopmans.aiida.setup import codes as codes_module

        exe = tmp_path / "pw2wannier90.x"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        monkeypatch.setattr(codes_module, "find_executable", lambda name: str(exe))

        found, missing = codes_module.scan_and_register_codes(
            {"pw2wannier90_decompose": codes_module.VARIANT_CODES["pw2wannier90_decompose"]},
            aiida_localhost,
        )
        assert found == []
        assert missing == ["pw2wannier90_decompose"]

    def test_explicit_path_registers_the_variant(
        self, aiida_profile_clean: Any, aiida_localhost: Any, tmp_path: Any
    ) -> None:
        """``--code pw2wannier90_decompose=<path>`` registers that binary."""
        from aiida.orm import load_code

        from koopmans.aiida.setup.codes import VARIANT_CODES, scan_and_register_codes

        exe = tmp_path / "decompose_pw2wannier90.x"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)

        found, missing = scan_and_register_codes(
            {"pw2wannier90_decompose": VARIANT_CODES["pw2wannier90_decompose"]},
            aiida_localhost,
            explicit_codes={"pw2wannier90_decompose": str(exe)},
        )
        assert found == ["pw2wannier90_decompose"]
        assert missing == []
        code = load_code(f"pw2wannier90_decompose@{aiida_localhost.label}")
        assert str(code.filepath_executable) == str(exe)
        assert code.default_calc_job_plugin == "koopmans.pw2wannier_decompose"

    def test_unknown_code_label_is_rejected(
        self, aiida_profile_clean: Any, aiida_localhost: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """A mistyped ``--code`` label stops the install instead of doing nothing."""
        import click

        from koopmans.aiida.setup import orchestrate

        monkeypatch.setattr(orchestrate, "get_localhost_computer", lambda **_: aiida_localhost)

        with pytest.raises(click.ClickException, match="pw2wannier_decompose"):
            orchestrate.setup_computers(
                explicit_codes={"pw2wannier_decompose": str(tmp_path / "pw2wannier90.x")}
            )
