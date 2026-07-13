---
name: map-legacy
description: Report where the concepts in a legacy koopmans file should live in the new AiiDA layout — reconnaissance only, no code changes. Invoke as `/map-legacy <path>` (e.g. `/map-legacy src/koopmans/workflows/_koopmans_dscf.py`).
---

# map-legacy

Read a file from the legacy `koopmans` package and produce a mapping report: for each significant concept (class, function, data structure), where does it belong in `koopmans2` or `aiida-koopmans2`?

**This skill makes no code changes.** It's for planning the next port.

## Arguments

- `$1` — path to a file in `/home/linsco_e/code/koopmans/`. Accepts absolute paths or paths relative to the legacy repo root. Required.

## Procedure

1. **Read the file fully** (and any module it clearly depends on, e.g. its imports from sibling modules within `koopmans/`).
2. **For each top-level class/function, classify it** into one of:
   - Physics/workflow logic → `aiida-koopmans2/src/aiida_koopmans/workgraphs/<step>.py` as `@task.graph`.
   - Calculator wrapper → either wrap upstream (scout first) or new CalcJob in `aiida-koopmans2/src/aiida_koopmans/calculations/`.
   - Settings / parameter schema → Pydantic model in `koopmans2/src/koopmans/input_file/`.
   - Domain data (Band, ProjectionBlock, etc.) → `orm.Data` subclass in `aiida-koopmans2/src/aiida_koopmans/data/`.
   - Input parsing / dispatch → `koopmans2/src/koopmans/input_file/` and `koopmans2/src/koopmans/aiida/workflows.py`.
   - ASE↔AiiDA conversion → `koopmans2/src/koopmans/aiida/conversion.py` (via `ase-bridge`).
   - **Drop** (infrastructure replaced by AiiDA): dill pickling, `HasDirectory`, engine, subprocess handling, file symlinking, `Status` enum.
3. **Check what's already ported.** Grep `koopmans2/` and `aiida-koopmans2/` for the legacy symbol names. Flag duplicates.
4. **Produce a report** in this format:

```
Legacy file: src/koopmans/workflows/_koopmans_dscf.py (lines: 1932)

| Legacy symbol | Kind | New location | Status |
|---|---|---|---|
| KoopmansDSCFWorkflow | Workflow class | aiida-koopmans2/workgraphs/koopmans_dscf.py as @task.graph | not started |
| _initialize_alpha | method (→ calcfunction) | aiida-koopmans2/workgraphs/koopmans_dscf.py helper | not started |
| Band.alpha | data attr | aiida-koopmans2/data/band.py | not started |
| DROP: self._pickle_inputs | infra | — | — |
| ... | ... | ... | ... |

Unported sub-workflows this file instantiates:
- WannierizeWorkflow (partially ported — see aiida-koopmans2/workgraphs/wannier90.py)
- DFTPWWorkflow (not ported)

Recommended next skill calls:
- /port-workflow WannierizeWorkflow   (to complete the dependency)
- /port-workflow KoopmansDSCFWorkflow  (this file)
```

5. **Do not write any code.** If the user asks you to port after seeing the map, they invoke `/port-workflow` or `/port-calculator` explicitly.

## Delegation

- For very large files (>1000 LOC), have `Explore` agents survey the file first rather than reading it all into context yourself.
