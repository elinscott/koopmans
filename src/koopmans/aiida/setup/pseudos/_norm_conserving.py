"""Read a pseudopotential's own header to tell norm-conserving from the rest.

Koopmans functionals are defined for norm-conserving pseudopotentials, and
``kcp.x`` and ``kcw.x`` accept no other kind. The label a family carries says
nothing about this once the family is one the user installed themselves, so
the answer comes from the UPF headers.

Neither ``aiida-pseudo`` nor ``aiida-core`` parses the field: ``UpfData``
reads the element and z_valence, and ``aiida.orm.nodes.data.upf.parse_upf``
the version and element. ``upf_tools.header_from_str`` reads the header of
either UPF layout and stops there, so a body no parser can read still yields
a verdict.
"""

from __future__ import annotations

import warnings
from typing import Any, Protocol

from upf_tools import header_from_str


class _ReadableFile(Protocol):
    """What this module needs of a pseudopotential node: its bytes as text."""

    def get_content(self) -> str:
        """Return the file's text."""
        ...


# UPF ``pseudo_type`` values that are not norm-conserving. "NC" and "SL"
# (semilocal) are; "US" and PSlibrary's "USPP" are the two spellings a v2
# header gives an ultrasoft pseudopotential, and a v1 header names no type at
# all, being judged on its ``is_ultrasoft`` flag instead. A bare Coulomb
# potential ("1/r") passes: both kcp.x and kcw.x synthesise its local
# potential and treat it like any local-only norm-conserving potential
# (CPV/src/pseudopot_sub.f90, upflib/vloc_mod.f90).
_NOT_NORM_CONSERVING = {"US", "USPP", "PAW"}

# UPF booleans are written as T/F, true/false or .true./.false. depending on
# the generator; upf-tools reads the first two spellings as booleans and hands
# back any other as the string it found.
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

    A v2 header names its ``pseudo_type``; a v1 header does not, and reaches
    here as the flags ``is_paw`` and ``is_ultrasoft``. The flags decide only
    where no name is given, which is why a v2 file naming itself ``PAW`` is
    reported as PAW even though it raises the ultrasoft flag too.
    """
    try:
        with warnings.catch_warnings():
            # A UPF v1 file states no version, so upf-tools warns that it
            # could not determine one for every single one of them.
            warnings.simplefilter("ignore", UserWarning)
            header = header_from_str(pseudo.get_content())
    except Exception:
        # Any read failure means "cannot tell", which is not grounds to refuse.
        return None

    declared = header.get("pseudo_type")
    name = "" if declared is None else str(declared).strip()
    if name:
        return name if name.upper() in _NOT_NORM_CONSERVING else None
    if _flag(header.get("is_paw")):
        return "PAW"
    if _flag(header.get("is_ultrasoft")):
        return "US"
    return None


def _flag(value: Any) -> bool:
    """Read a UPF boolean, which generators spell T, true or .true.."""
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() in _TRUE
