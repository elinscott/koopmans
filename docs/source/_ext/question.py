"""Provide the ``question`` directive used by the tutorials."""

from __future__ import annotations

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective

TEMPLATE = """\
.. admonition:: Question
    :class: question

{question}

    .. dropdown:: Show answer

{answer}
"""


def _indent(lines: list[str], width: int) -> str:
    """Indent non-blank lines by ``width`` spaces, leaving blank ones empty."""
    padding = " " * width
    return "\n".join(padding + line if line.strip() else "" for line in lines)


class Question(SphinxDirective):
    """Pose a question, with its answer folded away behind a dropdown.

    The argument is the question, the body its answer::

        .. question:: Why is the simulation cell larger than the molecule?

            The cell is a box of vacuum that keeps the periodic images apart.
    """

    required_arguments = 1
    final_argument_whitespace = True
    has_content = True

    def run(self) -> list[nodes.Node]:
        """Expand the directive into an admonition wrapping a dropdown."""
        text = TEMPLATE.format(
            question=_indent(self.arguments[0].splitlines(), 4),
            answer=_indent(list(self.content), 8),
        )
        return self.parse_text_to_nodes(text, offset=self.content_offset)


def setup(app: Sphinx) -> dict[str, object]:
    """Register the directive."""
    app.add_directive("question", Question)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
