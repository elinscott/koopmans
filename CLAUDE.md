# koopmans2

AiiDA-based rewrite of the original `koopmans` package (ASE-based Koopmans spectral functional calculations with Quantum ESPRESSO).

## The three-repo layout

All three live as sibling directories under `/home/linsco_e/code/`:

- **`koopmans/`** — the legacy, ASE-based implementation. Read-only source of truth for physics, workflow logic, and tutorial inputs. **Do not add features here.** Consult it to understand *what* the new code needs to do.
- **`koopmans2/`** (this repo) — user-facing package. Owns: CLI, Pydantic input file schema (`input_file/`), AiiDA profile/code setup, and the dispatcher that turns a `KoopmansInput` into a `WorkGraph` (`aiida/workflows/`).
- **`aiida-koopmans2/`** — the AiiDA plugin. Owns: `@task.graph` builders that wrap upstream WorkChains (`RunPwBands`, `Wannierize`, …) and, when unavoidable, new CalcJobs/Parsers for QE tools not covered upstream.

`koopmans2` depends on `aiida-koopmans`. Not the reverse. During development both (and `aiida-quantumespresso`, `aiida-workgraph`) are editable local installs — see `[tool.uv.sources]` in `pyproject.toml`.

## Conversion mapping (legacy → new home)

| Legacy (`koopmans/src/koopmans/...`) | New home |
|---|---|
| `workflows/_dft.py`, `_wannierize.py`, `_koopmans_dscf.py`, … | `aiida-koopmans2/src/aiida_koopmans/workgraphs/*.py` as `@task.graph` builders |
| `calculators/_pw.py`, `_wannier90.py`, … (ASE calculators) | Prefer reusing upstream (`aiida-quantumespresso`, `aiida-wannier90-workflows`). New CalcJob **only** if no upstream equivalent exists (likely for `kcp.x`, `kcw.x`, `wann2kc[p]`). |
| `settings/*.py` (ASE-flavoured settings dicts) | `koopmans2/src/koopmans/input_file/*.py` (Pydantic models). Mostly done. |
| `io/_json.py`, `Workflow._fromjsondct` | `koopmans2/src/koopmans/input_file/__init__.py` (parsing) + `aiida/workflows/` (dispatch) |
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
5. **`koopmans2/aiida/workflows/` stays thin.** The package `__init__` dispatches on `Task` enum (translating plugin errors into input-file advice via `advice_for`); each route builder lives in its own module (`dft`, `eps`, `wannierize`, `dscf`, `dfpt`, `trajectory`), with `blocks`, `grouping` and `projectors` as shared helpers. All real logic belongs in `aiida-koopmans2/workgraphs/`.
6. **Input translation is centralized in `koopmans2/aiida/conversion.py`.** Functions like `atoms_input_to_structure`, `input_to_pw_parameters` are the only place Pydantic models touch AiiDA ORM.
7. **No dill, no pickle checkpoints.** Provenance comes from AiiDA's database.

## Canonical patterns

**Adding a new workflow task** (expanding the dispatcher):

1. Add the `@task.graph` builder in `aiida-koopmans2/src/aiida_koopmans/workgraphs/<step>.py`.
2. Add a `TypedDict` for its outputs at the top of that module.
3. Expose any new codes through `load_codes_for_task` in [aiida/workflows/](src/koopmans/aiida/workflows/__init__.py).
4. Add a `build_<task>_workgraph` module under `aiida/workflows/` and wire it into `build_workgraph`.
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

1. **Thread parsed outputs; never re-parse files.** If an upstream parser emits the value (e.g. wannier90 `output_parameters`), expose/thread that socket — even when it means widening an interface. Raw-file access is reserved for data no parser provides (e.g. the u/hr/centres merge inputs).
2. **No duplication.** Before writing any helper, search `variational_orbitals.py`, `projections.py`, `occupations.py`, `workgraphs/utils/wannier_merge.py`, `workgraphs/ml/helpers.py`, `spin.py`, `ml.py`, `functionals.py`, `screening.py`, `parallelization.py` (ak2) and `conversion.py` (k2); extend in place. New modules only for genuinely new orchestration. Same rule for tests: shared fixtures and builders live in `tests/fixtures.py` (re-exported via `conftest.py`) — never define a fixture module-locally that a sibling module already has or could share.
3. **Structural authority over conventions.** Band order, manifold membership, block identity travel as explicit lists/fields from the caller — never derived from label prefixes or key-name conventions.
4. **Consistent naming families.** New symbols join their module's family (`KcwScreenStep` → `GroupedKcwScreening`, not `GroupedDFPTScreening`). Short user-facing keywords (`spin`, not `spin_treatment`).
5. **Explicit failure over silent ignore.** An input that cannot take effect raises `NotImplementedError`/`ValueError` naming the gap — no keyword is silently dropped.
6. **Adversarial pass before merge.** Every PR gets a reviewer-agent pass; load-bearing claims (mechanisms, parity, orderings) get skeptic verification or are graded honestly in the PR body (see Writing).
7. **Graph-layout changes need a cross-repo CI pairing.** k2's CI clones the same-named aiida-koopmans branch; an ak2-only PR that changes task names, sockets, or graph shapes must push a same-named k2 branch (even if empty of changes) so the pairing actually runs — otherwise k2 main goes silently red at the ak2 merge.
8. **Per-rank thread pinning is a computer-level default.** The GNU builds link threaded OpenBLAS; under mpirun each rank spawns its own BLAS threads and oversubscribes the hq allocation. The localhost computer's `prepend_text` carries `THREAD_PIN_PREPEND` (`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` = 1), set in `aiida/setup/computer.py` on both create and migrate (computers are mutable; code nodes are not). A per-code `omp` in the `parallelization` block raises the count for a given calculation via `metadata.options.prepend_text`, which aiida-core assembles after the computer prepend and so overrides it.
9. **A workaround for a dependency is a candidate bug report**, most often for aiida-workgraph or node-graph, which are young enough that our use finds their edges first. An annotation shaped for the framework rather than the contract, a value coerced to survive a serializer, a socket restructured to get past a validator: say what the defect is, which package it lives in, and what the workaround costs — then stop. Patching upstream is the maintainer's call; when taken, it is a branch off their main, cherry-picked onto our fork's `patched` (which CI clones), plus an upstream pull request. A workaround that stays says what it works around.

## Writing

One standard for everything we write: docstrings, comments, error messages,
PR bodies, commit messages, issues. Orwell's rules — short word, cut what
can be cut, active voice, no stale figure of speech — with one carve-out:
domain terms that carry a precise meaning (Wannier function,
disentanglement, socket) and upstream keyword names stay. What is forbidden
is *coined* jargon: "a pool-carrying block enters the split chain" makes
the reader learn two terms before they learn the fact. US spelling in prose
(Wannierize, behavior); upstream names keep their own (`guiding_centres`).

**Docstrings, comments, error messages.** A docstring says what the thing
does and what must hold for it to work.

- State the rule, not a picture of it: `dis_froz_max < min_k E(num_wann +
  1, k)`, not "the window is kept inside the block's own manifold".
- Say what the function does, not where its result is used.
- Document this object's contract. Another module's behaviour, and how it
  came to be this way, both go stale silently.
- No design justification and no restating the signature: both belong in
  the pull request.
- No redundant emphasis ("it is important to note", "and they must not be
  conflated").
- Imperative summary line, one line, full stop (ruff D401).
- Error messages add one rule: tell the reader what to change, in their
  vocabulary.

**PR bodies, commit messages, issues.** Public text explains; the diff
shows. One goal governs everything here: the body is a digestible
summary of what the PR does or solves, and how — an outsider gets the
whole story from it and opens the diff only for mechanics. Every rule
below serves that goal; where they conflict, clarity wins.

- Open with what the PR achieves, in plain terms; mechanics, private
  helper names and call-site detail stay in the diff.
- `### Problem / ### Changes / ### Testing` is the default shape, not a
  form: rename, add or drop headings as the change calls for.
- Bullets lend themselves to clarity when each carries one idea,
  briefly: several sentences in one bullet is a paragraph in disguise,
  and two changes joined by a semicolon are two bullets.
- Problem is a scenario an outsider can picture — never session
  codenames, database PKs or scratch paths. Testing says what each
  check discriminates, never bare pass counts.
- Grade claims (reproduced / code-read / theory) and assert only the
  reproduced ones.
- Worked examples stand alone: write the snippet a stranger could paste.
- Check for staleness before publishing.
- Squash messages in 50/72: subject ≤50 including `(#N)`, body wrapped at
  72, opening with one sentence pairing symptom and fix, then bullets.
- No Claude session URLs; the Co-Authored-By trailer stays.

**Documentation** decouples orthogonal choices: never present a default
pairing (grouping criterion ↔ screening method) as an equivalence.

## Current status (update as work progresses)

- Input file parsing (`input_file/`): ~95% ported.
- Automated block splitting (`block_wannierization_threshold`): the `WANNIERIZE` task routes through `aiida-koopmans2/workgraphs/auto_wannierize.py` — pw.x bands run, runtime gap/occ-boundary group detection, Wannier.jl parallel-transport split (via the `aiida-wannierjl` plugin; cubic b-vector fallback included), per-group preprocessing-free re-wannierization, block-diagonal u/hr/centres merge. Scope: explicit projections, spin='none'. Implicit/auto projections and `_u_dis.mat` merging of disentangled parent blocks are follow-ups. Needs a `wannierjl@localhost` code (julia; `aiida_wannierjl.helpers`).
- Dispatcher (`aiida/workflows/`): covers `DFT_BANDS`, `WANNIERIZE`, `SINGLEPOINT` (DSCF via kcp.x with KI/KIPZ, molecular KS-init and periodic Wannier-init routes; DFPT via kcw.x), `TRAJECTORY` (ML via `ml: {mode: train|test|predict}`, `self_hartree` descriptor; predict swaps each snapshot's Delta-SCF refinement for a trained-model prediction off the trial KI's self-Hartrees, feeding the final KI via the `initial_alphas`-style injection; test additionally runs a second final KI at model-predicted alphas per snapshot and reports the per-orbital alpha deltas and final-KI eigenvalue max/RMS under the evaluation's `alpha_and_eigenvalue_deltas`; multi-snapshot via `atoms.snapshots`, a multi-frame xyz mutually exclusive with explicit `atomic_positions`, resolved against the input file's directory), and `DFT_EPS` (ph.x dielectric).
- Spin: `workflow.spin` takes aiida-quantumespresso's `SpinType` (`none`/`collinear`/`non_collinear`/`spin_orbit`). DFPT supports all four regimes (collinear fans out per channel; noncollinear runs the spinor chain — QE reference `KCW/examples/example05.1`); the kcp.x streams support `none`/`collinear` only.
- Periodic DSCF (mlwfs/projwfs): wannierize → fold-to-supercell (wann2kcp.x + merge_evc.x) → Wannier-seeded kcp.x init; supercell-image orbital grouping approximated via the defaulted `group_orbitals_by='self_hartree'` / `group_orbitals_tol=1e-4` (constructive grouping not ported). Fold path has construction-level tests only — needs a live QE smoke test.
- Multi-block DFPT manifolds: supported end-to-end (per-block wannierize → block-diagonal u/hr merge, concatenated centres, identity-extended u_dis → kcw.x). Live-validated on the ZnO tutorial vs legacy: KS bands ≤0.3 meV, KI occupied ≤3 meV. KI **empty** bands scatter up to ~0.6 eV between codes — genuine MLWF multi-minima for the 2-WF disentangled manifold, gauge-dependence question for the koopmans team (issue to be opened), not a merge bug.
- Orbital grouping: `group_orbitals_by`/`group_orbitals_tol`. DSCF groups by self-Hartree (kcp.x metric); DFPT groups by wannier90 spread with a per-representative `SCREEN.i_orb` fan-out + alpha broadcast. Criteria and methods are independent in principle; unwired combinations raise. kcw.x's internal `check_spread` shortcut is separate (ak2 graph input, on by default).
- Explicit k-paths resolve against the cell's own Bravais lattice (ASE vocabulary, position-insensitive); seekpath only serves automatic paths.
- aiida-core: builds from `../aiida-core` at current upstream main (post-v2.8, includes the workgraph-dump MRO fix); aiida-shell rides its git master via `[tool.uv.sources]` until a post-`Code.Model` release exists. CI/RTD clone pin: `4c81e9d6`. NOTE: `uv run --project` re-syncs the venv whenever `../aiida-core` changes — keep that checkout where the CI pin points.
- aiida-quantumespresso: upstream main absorbed every `patched`-branch fix (pdos settings threading, PBC handling, hyperqueue default resources), so the `patched` branch is retired; CI/RTD clone upstream at the pinned sha `0a15b8ac`, and the local checkout tracks upstream main.
- ML descriptors: `self_hartree` and `power_spectrum` are wired in every `ml` mode (`train`/`test`/`predict`); under `predict`/`test` the decompose pass runs inside each snapshot's screening step, so it needs the Wannier-initialised route and a decompose-capable pw2wannier90.x. A trained model stamps `n_max`/`l_max`/`r_min`/`r_max` alongside `descriptor`/`correction`/`init_orbitals`, and prediction or scoring refuses a model whose stamps disagree. `power_spectrum` with a k-grid larger than 1x1x1 is refused on both sides — training trips the assemble-time length check, prediction a row-count check (rows are per primitive Wannier function, alphas per supercell orbital); pairing them needs a structural image-to-primitive map out of the fold, a follow-up. `orbital_density` fully built on pw2wannier90 `wan_mode='decompose'` (new ak2 CalcJob+parser, legacy-comparable cross-power, decompose math reproduced to machine precision on live Si) but **gated** pending a live per-block WF-to-alpha alignment regression; flipping the guard is one line.
- Known gaps (raise `NotImplementedError` with pointers in `aiida/workflows/`): corrections `PKIPZ`/`NONE`/`ALL`; `init_orbitals='pz'`; `fix_spin_contamination`; gamma-only/molecular DFPT; `eps_inf='auto'` for DSCF (wired for DFPT); UI inside singlepoints (the open item; the `ui-singlepoint` branch is kept as reference); `convergence` task.
- Standalone UI (unfold-and-interpolate) is unsupported by decision: `task: unfold_and_interpolate` is rejected at parse (no `Task` enum member), while the `unfold_and_interpolate:` input block and `UnfoldAndInterpolateConfig` stay for the future singlepoint use.
- Parallelization: a top-level per-code `parallelization` block (`input_file/parallelization.py`) sets each code's MPI ranks (`ntasks` → `metadata.options.resources`), k-point pools (`npool` → `-npool`), and pencil decomposition (`pd` → `-pd true`). Per-code flag support follows legacy `commands.py`: npool for pw/projwfc/kcw (kcw only on its wann2kc + screen steps, not ham), pd for pw/pw2wannier90/projwfc/kcw; the schema rejects unsupported flag/code combos. The dispatcher threads the whole mapping to every graph builder; the pw.x steps also ride the shared overrides. The legacy `workflow.npool` shorthand has been removed.
- Branch state: everything above is merged to `main` in both repos; the one open PR is aiida-wannierjl #2 (options-dict rebuild), awaiting that repo's wedged GitHub Actions. PR-per-change flow via the `pr` remotes; branches, commits, and PRs on the elinscott-owned repos are authored by the `elinsc-bot` machine account; merges are the user's.
