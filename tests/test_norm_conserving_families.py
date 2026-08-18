"""A family that is not norm-conserving is refused, on its headers not its label."""

from __future__ import annotations

from typing import Any

import pytest

from koopmans.input_file import KoopmansInput
from tests.fixtures import silicon_pw_input


def _dispatch(label: str) -> Any:
    """Run the dispatch-time checks against a family label."""
    from koopmans.aiida.workflows import prepare_common_inputs

    inp = KoopmansInput.model_validate(
        silicon_pw_input(pseudo_library=label, calculator_parameters={"ecutwfc": 20.0})
    )
    return prepare_common_inputs(inp, ["scf", "bands"])


class TestTheHeaderDecides:
    """The check reads each pseudopotential, not the family's label."""

    @pytest.mark.parametrize(
        ("fixture", "label", "pseudo_type"),
        [
            ("fake_ultrasoft_family", "MyPseudos/ultrasoft", "US"),
            ("fake_paw_family", "MyPseudos/paw", "PAW"),
        ],
    )
    def test_an_ultrasoft_or_paw_family_is_refused(
        self,
        aiida_profile_clean: Any,
        request: pytest.FixtureRequest,
        fixture: str,
        label: str,
        pseudo_type: str,
    ) -> None:
        """Both kinds koopmans cannot use are caught, and the message names them.

        Nothing in either label says ``US`` or ``PAW``, so a check reading the
        label would pass both; these families are exactly the shape
        ``aiida-pseudo install family`` produces from a downloaded directory.
        """
        request.getfixturevalue(fixture)

        with pytest.raises(ValueError) as excinfo:
            _dispatch(label)

        message = str(excinfo.value)
        assert "not norm-conserving" in message
        assert f"Si ({pseudo_type})" in message
        assert "workflow.pseudo_library" in message

    def test_a_bare_coulomb_family_is_accepted(
        self, aiida_profile_clean: Any, fake_coulomb_family: Any
    ) -> None:
        """A bare Coulomb potential ("1/r") passes the check.

        kcp.x and kcw.x synthesise its local potential and treat it like a
        local-only norm-conserving potential, so there is nothing to refuse.
        """
        _, pseudo_family, _ = _dispatch("MyPseudos/coulomb")
        assert pseudo_family == "MyPseudos/coulomb"

    def test_a_family_declaring_itself_nc_is_accepted(
        self, aiida_profile_clean: Any, fake_declared_nc_family: Any
    ) -> None:
        """The positive control: the same fixture machinery, one header value apart.

        Without this a check that refused every family would pass the two
        cases above.
        """
        structure, pseudo_family, _ = _dispatch("MyPseudos/nc")
        assert pseudo_family == "MyPseudos/nc"
        assert structure.get_kind_names() == ["Si"]

    def test_a_header_that_says_nothing_is_accepted(
        self, aiida_profile_clean: Any, fake_sg15_cutoffs_family: Any
    ) -> None:
        """A header without the attribute cannot be judged, so it does not block.

        Real generators write it, but a hand-trimmed or unparseable header
        must not stop a legitimate run: the check refuses only on positive
        evidence. Pinned so that turning it into a hard requirement is a
        deliberate change rather than a silent one.
        """
        _, pseudo_family, _ = _dispatch("SG15/1.0/PBE/SR")
        assert pseudo_family == "SG15/1.0/PBE/SR"


class _Pseudo:
    """The one thing the reader asks of a pseudopotential node."""

    def __init__(self, content: str) -> None:
        self._content = content

    def get_content(self) -> str:
        """Return the stub's text."""
        return self._content


class TestTheCheckReadsRealHeaders:
    """The reader is exercised against UPF text, not only through the dispatcher."""

    @pytest.mark.parametrize(
        ("pseudo_type", "expected"),
        [("NC", None), ("US", "US"), ("PAW", "PAW"), (None, None)],
    )
    def test_each_header_kind_reads_back(
        self, pseudo_type: str | None, expected: str | None
    ) -> None:
        """``NC`` and a missing attribute pass; ``US`` and ``PAW`` are named.

        Covers the reader without a profile, so a failure here separates a
        parsing bug from a wiring one.
        """
        from koopmans.aiida.setup.pseudos._norm_conserving import _pseudo_type
        from tests.fixtures import fake_upf_content

        content = fake_upf_content("Si", 4.0, pseudo_type=pseudo_type)
        assert _pseudo_type(_Pseudo(content)) == expected

    @pytest.mark.parametrize(
        ("fixture", "expected"),
        [
            ("UPF_V2_NORM_CONSERVING_HEADER", None),
            ("UPF_V2_ULTRASOFT_HEADER", "USPP"),
            ("UPF_V2_PAW_HEADER", "PAW"),
            ("UPF_V1_ULTRASOFT_HEADER", "US"),
            ("UPF_V2_ULTRASOFT_WITH_NAMELIST", "USPP"),
            ("UPF_V2_FLAGGED_BUT_UNNAMED", "US"),
        ],
    )
    def test_headers_transcribed_from_real_pseudopotentials(
        self, fixture: str, expected: str | None
    ) -> None:
        """Each real-world spelling the synthetic stream does not cover.

        PSlibrary writes ``is_ultrasoft="true"`` where SG15 writes ``"F"``; a
        PAW file sets the ultrasoft flag too, so flag order decides what it is
        called; the v1 layout is a fixed-format block rather than attributes;
        and a PP_INFO carrying a Fortran namelist makes the file invalid XML
        while leaving the header perfectly readable. The last states no
        ``pseudo_type`` at all, which is the only case the flags decide.
        """
        from koopmans.aiida.setup.pseudos import _norm_conserving
        from tests import fixtures

        assert _norm_conserving._pseudo_type(_Pseudo(getattr(fixtures, fixture))) == expected

    def test_a_v1_header_spelling_paw_is_refused(self) -> None:
        """A v1 PAW header is refused, as its v2 counterpart is.

        The v1 type line takes ``US``, ``PAW``, ``NC`` or ``1/r`` (Quantum
        ESPRESSO's own ``upflib/read_upf_v1.f90``), so a v1 file can say PAW
        and koopmans cannot use it. Reading it needs upf-tools to report that
        line's PAW as ``is_paw``, the key this check reads before the
        ultrasoft one.
        """
        from koopmans.aiida.setup.pseudos._norm_conserving import _pseudo_type
        from tests import fixtures

        assert _pseudo_type(_Pseudo(fixtures.UPF_V1_PAW_HEADER)) == "PAW"

    def test_a_file_whose_body_cannot_be_read_is_still_classified(self) -> None:
        """A pseudopotential truncated below its header is refused, not admitted.

        The first assertion is what makes the second mean anything: reading
        the whole file raises on this stream, so a check that had to parse the
        file to reach its header would report "cannot tell" and let an
        ultrasoft pseudopotential through.
        """
        from upf_tools import UPFDict

        from koopmans.aiida.setup.pseudos._norm_conserving import _pseudo_type
        from tests import fixtures

        stream = fixtures.UPF_V2_ULTRASOFT_WITH_UNREADABLE_BODY

        with pytest.raises(SyntaxError):
            UPFDict.from_str(stream)
        assert _pseudo_type(_Pseudo(stream)) == "USPP"

    def test_an_unparseable_stream_does_not_raise(self) -> None:
        """A file that is not UPF at all reads as "cannot tell", not as a crash."""
        from koopmans.aiida.setup.pseudos._norm_conserving import _pseudo_type

        assert _pseudo_type(_Pseudo("this is not a pseudopotential")) is None

    def test_a_v1_header_too_short_to_hold_a_type_is_not_guessed(self) -> None:
        """A truncated v1 block reads as "cannot tell" rather than as its second line.

        Indexing a fixed-format block is only safe while the block has the
        lines; without this a short header would refuse on an element symbol.
        """
        from koopmans.aiida.setup.pseudos._norm_conserving import _pseudo_type

        truncated = "<PP_HEADER>\n   0    Version Number\n  C     Element\n</PP_HEADER>\n"
        assert _pseudo_type(_Pseudo(truncated)) is None
