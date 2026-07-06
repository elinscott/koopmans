# koopmans2

AiiDA-based rewrite of the original `koopmans` package (ASE-based Koopmans spectral functional calculations with Quantum ESPRESSO).

## The three-repo layout

All three live as sibling directories under `/home/linsco_e/code/`:

- **`koopmans/`** — the legacy, ASE-based implementation. Read-only source of truth for physics, workflow logic, and tutorial inputs. **Do not add features here.** Consult it to understand *what* the new code needs to do.
- **`koopmans2/`** (this repo) — user-facing package. Owns: CLI, Pydantic input file schema (`input_file/`), AiiDA profile/code setup, and the dispatcher that turns a `KoopmansInput` into a `WorkGraph` (`aiida/workflows.py`).
- **`aiida-koopmans2/`** — the AiiDA plugin. Owns: `@task.graph` builders that wrap upstream WorkChains (`PwBandsTaskViaBuilder`, `Wannier90TaskViaBuilder`, …) and, when unavoidable, new CalcJobs/Parsers for QE tools not covered upstream.

`koopmans2` depends on `aiida-koopmans`. Not the reverse. During development both (and `aiida-quantumespresso`, `aiida-workgraph`) are editable local installs — see `[tool.uv.sources]` in `pyproject.toml`.

## Conversion mapping (legacy → new home)

| Legacy (`koopmans/src/koopmans/...`) | New home |
|---|---|
| `workflows/_dft.py`, `_wannierize.py`, `_koopmans_dscf.py`, … | `aiida-koopmans2/src/aiida_koopmans/workgraphs/*.py` as `@task.graph` builders |
| `calculators/_pw.py`, `_wannier90.py`, … (ASE calculators) | Prefer reusing upstream (`aiida-quantumespresso`, `aiida-wannier90-workflows`). New CalcJob **only** if no upstream equivalent exists (likely for `kcp.x`, `kcw.x`, `wann2kc[p]`). |
| `settings/*.py` (ASE-flavoured settings dicts) | `koopmans2/src/koopmans/input_file/*.py` (Pydantic models). Mostly done. |
| `io/_json.py`, `Workflow._fromjsondct` | `koopmans2/src/koopmans/input_file/__init__.py` (parsing) + `aiida/workflows.py` (dispatch) |
| `bands.py` (`Band`, `Bands`), `projections.py` (`ProjectionBlock`) | AiiDA `orm.Data` subclasses in `aiida-koopmans2/src/aiida_koopmans/data/` (to be created; register via entry points) |
| `engines/` (`LocalhostEngine`) | Replaced by AiiDA daemon + `aiida-workgraph`; no port |
| `processes/_process.py` dill checkpointing | Replaced by AiiDA provenance; no port |
| `cli/main.py` | `koopmans2/src/koopmans/cli.py` |
| `ase_koopmans.Atoms` as the central structure object | `orm.StructureData`, converted via `aiida/conversion.py` |

When in doubt, run `/map-legacy <file>` to get a current mapping report.

## Architectural rules

1. **Do not define new CalcJobs unless upstream has no equivalent.** First check `aiida-quantumespresso` (`PwBaseWorkChain`, `PwBandsWorkChain`, `PdosWorkChain`, `PhBaseWorkChain`, …) and `aiida-wannier90-workflows` (`Wannier90WorkChain`, `Wannier90OptimizeWorkChain`). Delegate that check to the `qe-plugin-scout` agent.
2. **Workflows live in `aiida-koopmans2` as `@task.graph` functions, not `WorkChain` subclasses.** WorkChain-as-task wrapping is fine (`task(PwBaseWorkChain)`), but composition is via `@task.graph`.
3. **Task outputs are `TypedDict`s** (see `workgraphs/pw.py` `ScfBandsOutputs`, `ScfNscfOutputs`). Wire downstream inputs as `outputs["remote_folder"]`, not attribute access.
4. **Builder → dict conversion uses `aiida_workgraph.utils.get_dict_from_builder`** before calling the wrapped task. Strip `clean_workdir` when chaining.
5. **`koopmans2/aiida/workflows.py` stays thin.** It dispatches on `Task` enum and assembles the right `@task.graph` with codes + overrides. All real logic belongs in `aiida-koopmans2/workgraphs/`.
6. **Input translation is centralized in `koopmans2/aiida/conversion.py`.** Functions like `atoms_input_to_structure`, `input_to_pw_parameters` are the only place Pydantic models touch AiiDA ORM.
7. **No dill, no pickle checkpoints.** Provenance comes from AiiDA's database.

## Canonical patterns

**Adding a new workflow task** (expanding the dispatcher):

1. Add the `@task.graph` builder in `aiida-koopmans2/src/aiida_koopmans/workgraphs/<step>.py`.
2. Add a `TypedDict` for its outputs at the top of that module.
3. Expose any new codes through `load_codes_for_task` in [aiida/workflows.py](src/koopmans/aiida/workflows.py).
4. Add a `_build_<task>_workgraph` helper and wire it into `build_workgraph`.
5. Add a regression test driven by a tutorial JSON (see `/regression-test`).

**Reference implementation to mirror:** [aiida-koopmans2/src/aiida_koopmans/workgraphs/pw.py](../aiida-koopmans2/src/aiida_koopmans/workgraphs/pw.py) (SCF+NSCF chaining shows the wiring pattern cleanly).

## Specialized agents

Loaded from `.claude/agents/`. Delegate aggressively — porting work is context-heavy:

- **koopmans-porter** — ports a legacy class/workflow into the new layout end-to-end.
- **workgraph-author** — narrow expert on `@task.graph` + TypedDict patterns.
- **qe-plugin-scout** — read-only; reports which upstream WorkChain covers a given QE step.
- **aiida-test-author** — writes AiiDA-fixture-based tests and regression harnesses.
- **ase-bridge** — owns ASE↔AiiDA conversions (Atoms, kpoints, bandpaths, projections, pseudos).

## Serena instances

Three serena MCP servers, one per repo (see `.mcp.json`). Tool paths are relative to each instance's repo root:

- `mcp__serena__*` — `koopmans2/` (this repo).
- `mcp__serena-aiida__*` — `../aiida-koopmans2/`.
- `mcp__serena-legacy__*` — `../koopmans/` (use read-only tools; the repo must never be edited).

Upstream packages (`aiida-quantumespresso`, `aiida-workgraph`, …) are not indexed — use Read/Grep there.

## Skills

Invoked via `/<name>`:

- `/port-workflow <ClassName>` — port one legacy workflow.
- `/port-calculator <ClassName>` — port one legacy calculator (scouting upstream first).
- `/regression-test <tutorial.json>` — set up a koopmans-vs-koopmans2 regression test.
- `/map-legacy <path>` — report where a legacy file's concepts map in the new layout.
- `/wire-input-field <calc> <field>` — trace a Pydantic input field into the builder overrides.

## Testing conventions

- `pytest` + AiiDA profile fixtures (convention TBD as test coverage grows — prefer `pytest-aiida` / `aiida.manage.tests.pytest_fixtures`).
- Regression tests live in `tests/regression/` and each consumes a tutorial JSON from `../koopmans/tutorials/` to stay in sync with the legacy reference.
- Don't mock AiiDA. Use a throwaway profile.

## Current status (update as work progresses)

- Input file parsing (`input_file/`): ~95% ported.
- Dispatcher (`aiida/workflows.py`): covers `DFT_BANDS`, `WANNIERIZE`, `SINGLEPOINT` (DSCF via kcp.x with KI/KIPZ, molecular KS-init and periodic Wannier-init routes; DFPT via kcw.x), `TRAJECTORY` (ML train/test/predict, `self_hartree` descriptor, multi-snapshot xyz input via `atomic_positions: {snapshots: file.xyz}`), `DFT_EPS` (ph.x dielectric), and `UI` (unfold-and-interpolate, pure python).
- Spin: `workflow.spin` takes aiida-quantumespresso's `SpinType` (`none`/`collinear`/`non_collinear`/`spin_orbit`). DFPT supports all four regimes (collinear fans out per channel; noncollinear runs the spinor chain — QE reference `KCW/examples/example05.1`); the kcp.x streams support `none`/`collinear` only.
- Periodic DSCF (mlwfs/projwfs): wannierize → fold-to-supercell (wann2kcp.x + merge_evc.x) → Wannier-seeded kcp.x init; supercell-image orbital grouping approximated via a defaulted `orbital_groups_self_hartree_tol=1e-4` (constructive grouping not ported). Fold path has construction-level tests only — needs a live QE smoke test.
- DFPT supports multi-block manifolds per spin channel (u/hr/centres/u_dis merging in `aiida_koopmans/wannier_merge.py`; tutorial: `docs/source/tutorials/zno.json`, translated to the new schema — legacy `base_functional`/`engine` blocks have no schema equivalent yet).
- Known gaps (raise `NotImplementedError` with pointers in `aiida/workflows.py`): corrections `PKIPZ`/`NONE`/`ALL`; `init_orbitals='pz'`; `fix_spin_contamination`; gamma-only/molecular DFPT; `eps_inf='auto'` for DSCF (wired for DFPT); `orbital_density` descriptor (planned via wannier90 `feature/decompose`); UI inside singlepoints (occ/emp × spin fan-out + smooth-wannierization); `convergence` task; constructive supercell-image orbital grouping (1e-4 self-Hartree stopgap in place).
- Branches (PR-per-change flow, user pushes/opens PRs): `core_functionality` → `spin-dfpt` → siblings {`dft-eps`, `ui-task`, `dscf-mlwf-init`} → `dfpt-multiblock` (on dft-eps), `ml-completion` (on dscf-mlwf-init); `wave-2` = working superset.
