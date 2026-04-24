---
name: regression-test
description: Set up a regression test that runs the same tutorial JSON through legacy `koopmans` and new `koopmans2`, then diffs the outputs. Invoke as `/regression-test <path-to-json>` (e.g. `/regression-test tutorials/tutorial_2/si.json`).
---

# regression-test

Create a regression test fixture pinning koopmans2 behaviour to legacy koopmans behaviour for a given tutorial input.

## Arguments

- `$1` — path (absolute or relative to `/home/linsco_e/code/koopmans/`) to a tutorial JSON input. Required.

If missing, list the tutorial JSONs under `/home/linsco_e/code/koopmans/tutorials/` and ask the user to pick one.

## Procedure

Delegate most of the work to the `aiida-test-author` agent. The skill's job is to set up the inputs and hand off cleanly.

1. **Validate the JSON:** confirm the dispatcher in `koopmans2` supports the `task` it declares. If not, stop and tell the user which task needs to be ported first (`/port-workflow <...>`).
2. **Generate the legacy reference** (one-time, cached):
   - Run the legacy workflow via `koopmans` CLI in a temp directory with QE available.
   - If QE is not available locally, skip execution and mark the reference as "construction-only" — the test will assert workgraph shape, not numerics. Tell the user this is what happened.
   - Serialize the legacy outputs (energies, bands, spreads, alpha parameters, DOS arrays) to a JSON reference at `koopmans2/tests/regression/references/<slug>.json`.
3. **Write the test file** at `koopmans2/tests/regression/test_<slug>.py`:
   - Parse the JSON with `KoopmansInput`.
   - Call `build_workgraph(koopmans_input)`.
   - Either assert graph shape (tasks, ports, wiring), or — behind `@pytest.mark.qe` — execute and diff against the reference with documented tolerances.
4. **Diff tolerances** (defaults, override only with justification):
   - Energies: `atol=1e-5 eV`.
   - Band energies: `atol=1e-4 eV`.
   - Wannier spreads: `atol=1e-3 Å²`.
   - Alpha screening parameters: `atol=1e-3`.
   - Numerical arrays: `numpy.testing.assert_allclose`, ignore transient keys (walltime, paths, PKs).
5. **Record the tolerance rationale** in a docstring on the test function.
6. **Summarize for the user**: which tutorial, which mode (construction-only / execution), reference size, and what tolerances were set.

## Sanity checks before you call it done

- Reference JSON is deterministic across re-runs of the legacy code.
- Test runs in <30s unless marked `@pytest.mark.qe`.
- Test name includes the tutorial number and the physics step (`test_tutorial_2_si_dft_bands`).
- No real HPC credentials or shared profiles touched.
