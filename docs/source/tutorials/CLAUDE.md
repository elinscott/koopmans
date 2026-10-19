# Writing a tutorial

Applies to every page under this directory. The repo-wide standard in the
root `CLAUDE.md` (Writing) still holds; this adds what is specific to a page
a reader arrives at cold.

A tutorial shows before it tells: the command the reader types, what they
see, then why. Write for someone arriving cold with the domain basics, and
answer their questions in the order they arise.

- Let a figure carry the result where one can, rather than a table or a
  paragraph restating it. Quote a number once, in the prose, when the
  argument depends on it; otherwise the axes are enough.
- Every number comes from a run of the shipped input, read off that run's
  own outputs — never hand-rolled, estimated, or taken from a note.
- A number's precision follows its job. One that points at a feature on a
  figure takes the precision the reader gets from the axes ("about 3.5 eV",
  "roughly doubles"), since a reader checks it by eye. One the argument
  computes with takes the decimals the arithmetic needs. Program output is
  quoted verbatim.
- Explain a mechanism with its logic signposted: each sentence is cause →
  consequence with the connective written ("so … until …", "therefore");
  contrasting cases stand in stated opposition ("In contrast, …");
  describe each case by its effect on the quantity the reader cares about,
  not by a mechanical picture; name the general lesson, not only the
  instance.
- No shorthand that assumes the reader already knows. An analogy is not an
  explanation unless it informs more than the plain statement — if the
  plain statement takes two sentences, write them. No "schematically" or
  "effectively": say what is omitted or approximated.
- An aside that answers a reviewer's question but not a first-time
  reader's does not belong on the page; the finding goes in an issue.
- Never document a quirk, workaround, or per-route caveat: fix it, file
  it, or patch the code it describes.
- Name the value the page settles on where it is used ("Set the converged
  value, 13.28, as …"), never "this value".
- Results are labelled correction@base (`KI@LDA`), with the base
  functional dashed in overlays; a `question` prompt stays on one line so
  it reads as the question, not the answer.
