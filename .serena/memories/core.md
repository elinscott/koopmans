# koopmans2 core

User-facing package of the AiiDA-based rewrite of the legacy `koopmans` package (Koopmans spectral functionals with Quantum ESPRESSO).

## Three-repo layout (siblings under ~/code/)
- `koopmans/` — legacy ASE implementation. READ-ONLY reference for physics/workflow logic/tutorial inputs.
- `koopmans2/` — this repo: CLI, Pydantic input schema, AiiDA profile/code setup, dispatcher.
- `aiida-koopmans2/` — AiiDA plugin: `@task.graph` builders, KC-specific CalcJobs/Parsers.
Dependency direction: koopmans2 → aiida-koopmans (never reverse).

## Source map (src/koopmans/)
- `cli.py`, `__main__.py` — CLI entry (`koopmans` script).
- `input_file/` — Pydantic input schema (~95% ported). Parsing in `__init__.py`; `Task` enum in `workflow.py`.
- `aiida/workflows/` — dispatcher package: `Task` enum → WorkGraph assembly, one route module per task. MUST stay thin; real logic lives in aiida-koopmans2 workgraphs.
- `aiida/conversion.py` — ONLY place Pydantic models touch AiiDA ORM (`atoms_input_to_structure`, `input_to_pw_parameters`, …).
- `aiida/setup.py` — profile/code/pseudo-family setup (`ensure_pseudo_family_installed`).

## Invariants
- No new CalcJobs unless upstream (`aiida-quantumespresso`, `aiida-wannier90-workflows`) lacks an equivalent.
- No WorkChain subclasses; composition via `@task.graph` (in aiida-koopmans2).
- No dill/pickle checkpoints — AiiDA provenance replaces them.
- Legacy→new conversion mapping table lives in `CLAUDE.md` (repo root) — read it before porting.

Tech/deps: `mem:tech_stack`. Commands: `mem:suggested_commands`. Style: `mem:conventions`. Done-criteria: `mem:task_completion`.
