---
name: port-calculator
description: Port one legacy koopmans CalculatorExt subclass to AiiDA — preferring an upstream WorkChain wrapper and only writing a new CalcJob when nothing upstream fits. Invoke as `/port-calculator <ClassName>` (e.g. `/port-calculator KCWCalculator`).
---

# port-calculator

Port a single `CalculatorExt` subclass from `koopmans/src/koopmans/calculators/` to AiiDA.

## Arguments

- `$1` — the legacy class name (e.g. `PWCalculator`, `Wannier90Calculator`, `KCWCalculator`, `KoopmansCPCalculator`, `Wann2KCPCalculator`, `PhCalculator`, `ProjwfcCalculator`, `Pw2Wannier90Calculator`). Required.

If missing, list candidates from `/home/linsco_e/code/koopmans/src/koopmans/calculators/`.

## Procedure

1. **Invoke the `qe-plugin-scout` agent** first. Give it the QE binary the calculator wraps (e.g. "kcw.x", "projwfc.x"). Do not write any code until the scout returns.
2. **Branch on the scout's recommendation:**
   - **Upstream WorkChain exists** → the port is thin: just wrap it as `task(UpstreamWorkChain)` inside a `@task.graph` in `aiida-koopmans2/src/aiida_koopmans/workgraphs/<step>.py`. Use `get_builder_from_protocol` if available. Delegate to `koopmans-porter` for the write.
   - **Only an upstream CalcJob exists, no WorkChain** → write a minimal `@task.graph` that wraps the CalcJob with whatever input setup the legacy calculator did. Delegate to `koopmans-porter`.
   - **Nothing upstream** (expected for `kcp.x`, `kcw.x`, `wann2kc[p]`) → a new CalcJob + Parser lives in `aiida-koopmans2/src/aiida_koopmans/calculations/<tool>.py` and `parsers/<tool>.py`, registered via `pyproject.toml` entry points under `aiida.calculations` and `aiida.parsers`. Delegate to `koopmans-porter`, then have `workgraph-author` review the `@task.graph` wrapper.
3. **Delete the `diff*` template placeholders** if this is the first real CalcJob/Parser being added to `aiida-koopmans2` — confirm with the user first.
4. **Read the legacy calculator fully** before writing — its `__init__`, `calculate`, and any `set_file_reference` / `link` logic encode physics-relevant file wiring. Translate the wiring intent (not the mechanism) into builder inputs and `parent_folder` chains.
5. **Drop legacy infrastructure baggage**: `HasDirectory`, dill pickling, ASE `Calculator` parent, `CalculatorExt.link()` file symlinking. AiiDA handles this.
6. **Preserve input translation** — the legacy settings dict → QE namelist mapping. In the new world this happens in `koopmans2/src/koopmans/aiida/conversion.py` (or in `get_builder_from_protocol` overrides). Delegate ASE-side conversion to `ase-bridge` if structure/kpoints are involved.
7. **Tests via `aiida-test-author`**: a minimal CalcJob construction test (no execution) at minimum; execution test behind `@pytest.mark.qe` if the binary is available.
8. **Summarize for the user**: which path (WorkChain wrap / CalcJob wrap / new CalcJob) was chosen, why, what files changed, what tests were added.

## Sanity checks before you call it done

- No new CalcJob was written without the scout confirming nothing upstream fits.
- `pyproject.toml` entry points updated if a new CalcJob/Parser was added.
- `ruff check` passes in both repos.
- Legacy repo untouched.
