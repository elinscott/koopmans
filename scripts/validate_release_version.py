"""Validate a release version before `just tag-release` tags it.

Checks that the version is valid PEP 440 and strictly newer than the latest
existing `vX.Y.Z` tag (if any). Exits non-zero with an explanatory message
on failure.
"""

import sys

from packaging.version import InvalidVersion, Version


def main() -> None:
    """Validate ``sys.argv[1]`` against the latest tag in ``sys.argv[2]``."""
    version_str = sys.argv[1]
    latest_tag = sys.argv[2] if len(sys.argv) > 2 else ""

    try:
        version = Version(version_str)
    except InvalidVersion as exc:
        sys.exit(f"{version_str} is not a valid PEP 440 version: {exc}")

    if latest_tag and version <= Version(latest_tag):
        sys.exit(f"{version_str} is not newer than the latest tag v{latest_tag}.")


if __name__ == "__main__":
    main()
