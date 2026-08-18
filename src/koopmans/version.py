"""Version information for :mod:`koopmans`.

Run with ``python -m koopmans.version``
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = [
    "VERSION",
    "get_version",
]

try:
    VERSION = version("koopmans")
except PackageNotFoundError:
    # Package metadata is missing (e.g. running from a source checkout that
    # was never installed). Match hatch-vcs's own git-less fallback.
    VERSION = "0.0.0+unknown"


def get_version() -> str:
    """Get the :mod:`koopmans` version string."""
    return VERSION


if __name__ == "__main__":
    print(get_version())  # noqa:T201
