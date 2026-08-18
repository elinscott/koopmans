"""Validate a release version before `just tag-release` tags it.

Checks that the version is a final PEP 440 release (no dev/pre/post/local
segment), strictly newer than the latest existing `vX.Y.Z` tag (if any),
and matches the version CITATION.cff declares. Exits non-zero with an
explanatory message on failure.
"""

import re
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version


def main() -> None:
    """Validate ``sys.argv[1]`` against the latest tag in ``sys.argv[2]``."""
    version_str = sys.argv[1]
    latest_tag = sys.argv[2] if len(sys.argv) > 2 else ""

    if version_str[:1] in ("v", "V"):
        sys.exit(
            f"Pass the version without a leading 'v': use "
            f"'{version_str[1:]}', not '{version_str}'. tag-release adds "
            "the 'v' prefix itself."
        )

    try:
        version = Version(version_str)
    except InvalidVersion as exc:
        sys.exit(f"{version_str} is not a valid PEP 440 version: {exc}")

    if version.is_devrelease or version.is_prerelease or version.is_postrelease or version.local:
        sys.exit(
            f"{version_str} is not a final release version — a release tag "
            "cannot carry a dev, pre-release, post-release, or local segment."
        )

    if latest_tag and version <= Version(latest_tag):
        sys.exit(f"{version_str} is not newer than the latest tag v{latest_tag}.")

    citation = Path("CITATION.cff")
    match = re.search(r"^version:\s*(\S+)\s*$", citation.read_text(), re.MULTILINE)
    if match is None:
        sys.exit(f"CITATION.cff declares no version; add `version: {version_str}` before tagging.")
    if Version(match.group(1)) != version:
        sys.exit(
            f"CITATION.cff declares version {match.group(1)}, not "
            f"{version_str} — update it (and commit) before tagging, so the "
            "citation matches the release."
        )


if __name__ == "__main__":
    main()
