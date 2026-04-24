---
name: port-workflow
description: Port one legacy koopmans Workflow class into a `@task.graph` builder in `aiida-koopmans2/` plus a dispatcher branch in `koopmans2/`. Invoke as `/port-workflow <ClassName>` (e.g. `/port-workflow WannierizeWorkflow`).
---

# port-workflow

Port a single `Workflow` subclass from the legacy `koopmans` package to the AiiDA rewrite.

## Arguments

- `$1` — the legacy class name (e.g. `WannierizeWorkflow`, `KoopmansDSCFWorkflow`, `DFTBandsWorkflow`, `TrajectoryWorkflow`). Required.

If missing, ask the user which class to port and list candidates from `/home/linsco_e/code/koopmans/src/koopmans/workflows/`.

## Procedure

1. **Read `koopmans2/CLAUDE.md`** if you haven't already this session — it has the mapping table and architectural rules.
2. **Locate and read the legacy class.** It's in `/home/linsco_e/code/koopmans/src/koopmans/workflows/`. Read the whole file plus any sub-workflows it instantiates.
3. **Check what's already done.** Read `koopmans2/src/koopmans/aiida/workflows.py` and `aiida-koopmans2/src/aiida_koopmans/workgraphs/` — the task may be partially ported.
4. **For each QE calculator the workflow uses, delegate to the `qe-plugin-scout` agent** in a single batched message if there are multiple. Collect the recommendations before writing code.
5. **Delegate the actual porting to the `koopmans-porter` agent.** Pass it:
   - The legacy class name and file path.
   - The scout's findings.
   - Any relevant existing workgraphs it should compose with.
6. **Review the produced `@task.graph` with `workgraph-author`** if the wiring is non-trivial (chained SCF→NSCF→X, optimizer loops, conditional branches).
7. **Ensure the dispatcher is updated** in `koopmans2/src/koopmans/aiida/workflows.py`:
   - `Task` enum entry (if new) in `koopmans2/src/koopmans/input_file/workflow.py`.
   - `load_codes_for_task` branch.
   - `_build_<task>_workgraph` helper.
   - `build_workgraph` dispatch branch.
8. **Add a regression test via `aiida-test-author`.** Construction-only (no QE execution) is fine for the first pass; flag to the user whether an execution-based test should follow.
9. **Summarize for the user**: what was ported, what was deferred, what new dependencies (if any), and which tutorial JSONs can now be dispatched.

## Sanity checks before you call it done

- `ruff check` passes in both repos.
- The dispatcher branches don't raise `NotImplementedError` for the new task.
- A unit test that builds the workgraph from a representative JSON passes.
- No edits landed in `/home/linsco_e/code/koopmans/` — that repo is read-only.
