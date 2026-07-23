# koopmans2

AiiDA-based rewrite of the original `koopmans` package (ASE-based Koopmans spectral functional calculations with Quantum ESPRESSO).

## The three-repo layout

All three live as sibling directories under `/home/linsco_e/code/`:

- **`koopmans/`** — the legacy, ASE-based implementation. Read-only source of truth for physics, workflow logic, and tutorial inputs. **Do not add features here.** Consult it to understand *what* the new code needs to do.
- **`koopmans2/`** (this repo) — user-facing package. Owns: CLI, Pydantic input file schema (`input_file/`), AiiDA profile/code setup, and the dispatcher that turns a `KoopmansInput` into a `WorkGraph` (`aiida/workflows.py`).
- **`aiida-koopmans2/`** — the AiiDA plugin. Owns: `@task.graph` builders that wrap upstream WorkChains (`RunPwBands`, `Wannierize`, …) and, when unavoidable, new CalcJobs/Parsers for QE tools not covered upstream.

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

## Code standards

1. **No "legacy" in production code.** Comments and docstrings state constraints and behaviour, never provenance ("ported from", "legacy parity"). Commit messages and PR bodies may reference legacy freely.
2. **Thread parsed outputs; never re-parse files.** If an upstream parser emits the value (e.g. wannier90 `output_parameters`), expose/thread that socket — even when it means widening an interface. Raw-file access is reserved for data no parser provides (e.g. the u/hr/centres merge inputs).
3. **No duplication.** Before writing any helper, search `variational_orbitals.py`, `projections.py`, `occupations.py`, `wannier_merge.py`, `ml_helpers.py`, `types.py` (ak2) and `conversion.py` (k2); extend in place. New modules only for genuinely new orchestration. Same rule for tests: shared fixtures and builders live in `tests/fixtures.py` (re-exported via `conftest.py`) — never define a fixture module-locally that a sibling module already has or could share.
4. **Structural authority over conventions.** Band order, manifold membership, block identity travel as explicit lists/fields from the caller — never derived from label prefixes or key-name conventions.
5. **Consistent naming families.** New symbols join their module's family (`KcwScreenStep` → `GroupedKcwScreening`, not `GroupedDFPTScreening`). Short user-facing keywords (`spin`, not `spin_treatment`). Docstrings in the imperative mood (mechanically enforced: ruff D401 under the pep257 convention).
6. **Docs decouple orthogonal choices.** Never present a default pairing (e.g. grouping criterion ↔ screening method) as an equivalence; say once that defaults reflect what is wired up, and let the dispatcher reject the rest explicitly.
7. **Explicit failure over silent ignore.** An input that cannot take effect raises `NotImplementedError`/`ValueError` naming the gap — no keyword is silently dropped.
8. **Adversarial pass before merge.** Every PR gets a reviewer-agent pass; load-bearing claims (mechanisms, parity, orderings) get skeptic verification or are graded honestly (reproduced / code-read / theory) in the PR body. Claims graded below reproduced are phrased as such.
9. **PR descriptions are short.** A summary line, a bulleted change list, a validation/testing section — nothing else. No implementation narratives, no session-local details (database PKs, scratch paths, agent names); those live in the conversation, not the record.
10. **Squash-merge messages in 50/72** (subject ≤50 chars including the `(#N)`, body wrapped at 72), symptom-not-mechanism, enumerations as bullet lists.
11. **US spelling in prose** (Wannierize, initialize, normalize, behavior). Exempt: upstream keyword and file names keep their canonical form (`guiding_centres`, `*_centres.xyz` are wannier90's own spelling).
12. **Graph-layout changes need a cross-repo CI pairing.** k2's CI clones the same-named aiida-koopmans branch; an ak2-only PR that changes task names, sockets, or graph shapes must push a same-named k2 branch (even if empty of changes) so the pairing actually runs — otherwise k2 main goes silently red at the ak2 merge.
13. **`OMP_NUM_THREADS=1` on every QE code.** The GNU builds link threaded OpenBLAS; under mpirun each rank spawns its own BLAS threads and oversubscribes the hq allocation. Neither repo sets it yet — until the code-registration fix lands, export it in the codes' `prepend_text`.

## Current status (update as work progresses)

- Input file parsing (`input_file/`): ~95% ported.
- Dispatcher (`aiida/workflows.py`): covers `DFT_BANDS`, `WANNIERIZE`, `SINGLEPOINT` (DSCF via kcp.x with KI/KIPZ, molecular KS-init and periodic Wannier-init routes; DFPT via kcw.x), `TRAJECTORY` (ML train/test, `self_hartree` descriptor), `DFT_EPS` (ph.x dielectric), and `UI` (unfold-and-interpolate, pure python).
- Spin: `workflow.spin` takes aiida-quantumespresso's `SpinType` (`none`/`collinear`/`non_collinear`/`spin_orbit`). DFPT supports all four regimes (collinear fans out per channel; noncollinear runs the spinor chain — QE reference `KCW/examples/example05.1`); the kcp.x streams support `none`/`collinear` only.
- Periodic DSCF (mlwfs/projwfs): wannierize → fold-to-supercell (wann2kcp.x + merge_evc.x) → Wannier-seeded kcp.x init; supercell-image orbital grouping approximated via the defaulted `group_orbitals_by='self_hartree'` / `group_orbitals_tol=1e-4` (constructive grouping not ported). Fold path has construction-level tests only — needs a live QE smoke test.
- Multi-block DFPT manifolds: supported end-to-end (per-block wannierize → block-diagonal u/hr merge, concatenated centres, identity-extended u_dis → kcw.x). Live-validated on the ZnO tutorial vs legacy: KS bands ≤0.3 meV, KI occupied ≤3 meV. KI **empty** bands scatter up to ~0.6 eV between codes — genuine MLWF multi-minima for the 2-WF disentangled manifold, gauge-dependence question for the koopmans team (issue to be opened), not a merge bug.
- Orbital grouping: `group_orbitals_by`/`group_orbitals_tol`. DSCF groups by self-Hartree (kcp.x metric); DFPT groups by wannier90 spread with a per-representative `SCREEN.i_orb` fan-out + alpha broadcast. Criteria and methods are independent in principle; unwired combinations raise. kcw.x's internal `check_spread` shortcut is separate (ak2 graph input, on by default).
- Explicit k-paths resolve against the cell's own Bravais lattice (ASE vocabulary, position-insensitive); seekpath only serves automatic paths.
- aiida-core: builds from `../aiida-core` at current upstream main (post-v2.8, includes the workgraph-dump MRO fix); aiida-shell rides its git master via `[tool.uv.sources]` until a post-`Code.Model` release exists. CI/RTD clone pin: `4c81e9d6`. NOTE: `uv run --project` re-syncs the venv whenever `../aiida-core` changes — keep that checkout where the CI pin points.
- ML descriptors: `self_hartree` wired; `orbital_density` fully built on pw2wannier90 `wan_mode='decompose'` (new ak2 CalcJob+parser, legacy-comparable cross-power, decompose math reproduced to machine precision on live Si) but **gated** pending a live per-block WF-to-alpha alignment regression; flipping the guard is one line.
- Known gaps (raise `NotImplementedError` with pointers in `aiida/workflows.py`): corrections `PKIPZ`/`NONE`/`ALL`; `init_orbitals='pz'`; `fix_spin_contamination`; gamma-only/molecular DFPT; `eps_inf='auto'` for DSCF (wired for DFPT); `ml:predict`; multi-snapshot trajectory input; UI inside singlepoints; `convergence` task; parallelization entries for non-pw codes (schema accepts, dispatcher raises).
- Branch state: everything above is merged to `main` in both repos; no open PRs. PR-per-change flow via the `pr` remotes (elinscott forks); merges are the user's.
