"""Read a pseudopotential's own header to tell norm-conserving from the rest.

Koopmans functionals are defined for norm-conserving pseudopotentials, and
``kcp.x`` and ``kcw.x`` accept no other kind. The label a family carries says
nothing about this once the family is one the user installed themselves, so
the answer comes from the UPF headers.

Neither ``aiida-pseudo`` nor ``aiida-core`` parses the field: ``UpfData``
reads the element and z_valence, and ``aiida.orm.nodes.data.upf.parse_upf``
the version and element. Both layouts are read here instead.
"""

from __future__ import annotations

import re
from typing import Protocol


class _ReadableFile(Protocol):
    """What this module needs of a pseudopotential node: its bytes as text."""

    def get_content(self) -> str:
        """Return the file's text."""
        ...


# UPF ``pseudo_type`` values that are not norm-conserving. "NC" and "SL"
# (semilocal) are; "US" is what a v1 header carries, where there are no
# flags to fall back on; "USPP" is PSlibrary's v2 spelling. A bare Coulomb
# potential ("1/r") passes: both kcp.x and kcw.x synthesise its local
# potential and treat it like any local-only norm-conserving potential
# (CPV/src/pseudopot_sub.f90, upflib/vloc_mod.f90).
_NOT_NORM_CONSERVING = {"US", "USPP", "PAW"}

# UPF v2 writes the header as XML attributes. Only the first 4 kB after the
# tag is searched, which covers the longest real header and keeps a stray
# match in the body out of it.
_HEADER_V2 = re.compile(r"<PP_HEADER\b")
_ATTRIBUTE = r'{}\s*=\s*"([^"]*)"'
_HEADER_SCAN = 4096

# UPF v1 writes a fixed-format block instead, whose third line is the type
# followed by its prose name:
#
#     <PP_HEADER>
#        0                   Version Number
#       C                    Element
#        US                  Ultrasoft pseudopotential
_HEADER_V1 = re.compile(r"<PP_HEADER>(?P<block>.*?)</PP_HEADER>", re.DOTALL)
_V1_TYPE_LINE = 2

# UPF booleans are written as T/F, true/false or .true./.false. depending on
# the generator; PSlibrary writes "true" where SG15 writes "F".
_TRUE = {"t", "true", ".true."}


def non_norm_conserving_kinds(pseudos: dict[str, _ReadableFile]) -> dict[str, str]:
    """Return the kinds whose pseudopotential is demonstrably not norm-conserving.

    Maps kind name to the offending ``pseudo_type``. A header that states
    nothing about its type is left out: the check refuses on positive
    evidence, so an unreadable or minimal header never blocks a run.
    """
    offenders: dict[str, str] = {}
    for kind, pseudo in sorted(pseudos.items()):
        pseudo_type = _pseudo_type(pseudo)
        if pseudo_type is not None:
            offenders[kind] = pseudo_type
    return offenders


def _pseudo_type(pseudo: _ReadableFile) -> str | None:
    """Return the pseudopotential's type if its header says it is not norm-conserving.

    ``None`` when the header calls it norm-conserving, says nothing, or
    cannot be read at all.
    """
    try:
        content = pseudo.get_content()
    except Exception:
        # Any read failure means "cannot tell", which is not grounds to refuse.
        return None

    declared = _read_v2_header(content)
    if declared is None:
        declared = _read_v1_header(content)
    if declared is None:
        return None

    pseudo_type, is_ultrasoft, is_paw = declared
    if is_paw:
        return pseudo_type or "PAW"
    if is_ultrasoft:
        return pseudo_type or "US"
    if pseudo_type is not None and pseudo_type.strip().upper() in _NOT_NORM_CONSERVING:
        return pseudo_type
    return None


def _read_v2_header(content: str) -> tuple[str | None, bool, bool] | None:
    """Return ``(pseudo_type, is_ultrasoft, is_paw)`` from an XML-attribute header.

    ``None`` when the file carries no such header, which is what sends a v1
    file on to :func:`_read_v1_header`.
    """
    match = _HEADER_V2.search(content)
    if match is None:
        return None

    window = content[match.end() : match.end() + _HEADER_SCAN]
    pseudo_type = _attribute(window, "pseudo_type")
    is_ultrasoft = _flag(_attribute(window, "is_ultrasoft"))
    is_paw = _flag(_attribute(window, "is_paw"))

    if pseudo_type is None and not is_ultrasoft and not is_paw:
        return None
    return pseudo_type, is_ultrasoft, is_paw


def _read_v1_header(content: str) -> tuple[str | None, bool, bool] | None:
    """Return ``(pseudo_type, False, False)`` from a fixed-format v1 header.

    The type is the first word of the block's third line. ``None`` when there
    is no such block or it is too short to hold one.
    """
    match = _HEADER_V1.search(content)
    if match is None:
        return None

    lines = [line for line in match.group("block").splitlines() if line.strip()]
    if len(lines) <= _V1_TYPE_LINE:
        return None

    words = lines[_V1_TYPE_LINE].split()
    if not words:
        return None
    return words[0], False, False


def _attribute(window: str, name: str) -> str | None:
    """Return a quoted XML attribute's value, or ``None`` if it is absent."""
    match = re.search(_ATTRIBUTE.format(re.escape(name)), window)
    return match.group(1) if match else None


def _flag(value: str | None) -> bool:
    """Read a UPF boolean, which generators spell T, true or .true.."""
    return value is not None and value.strip().lower() in _TRUE
