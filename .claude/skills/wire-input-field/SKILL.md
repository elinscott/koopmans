---
name: wire-input-field
description: Trace a single Pydantic input field (from `koopmans2/input_file/`) through `aiida/conversion.py` into the appropriate builder override so it actually takes effect in the downstream QE/Wannier calculation. Invoke as `/wire-input-field <calc> <field>` (e.g. `/wire-input-field pw ecutwfc`, `/wire-input-field w90 num_iter`, `/wire-input-field pw2wannier90 atom_proj_ext`).
---

# wire-input-field

A single input field (a Pydantic attribute declared in `koopmans2/src/koopmans/input_file/`) needs to reach the right QE input card / Wannier90 keyword / builder port at runtime. This skill traces or fixes the wiring end-to-end.

## Arguments

- `$1` — calculator namespace. One of: `workflow`, `pw`, `kcp`, `wannier90` (or `w90`), `pw2wannier90` (or `pw2w90`), `projwfc`, `kcw`, `ph`, `ui`.
- `$2` — field name as it appears on the Pydantic model (e.g. `ecutwfc`, `num_iter`, `atom_proj_ext`).

Both required.

## Procedure

1. **Locate the field** in `koopmans2/src/koopmans/input_file/<calc>.py`. Confirm its type, default, and any validators. If absent, stop and report: the field doesn't exist yet, the user needs to add it to the Pydantic model first.
2. **Locate the legacy wiring** for reference: search `/home/linsco_e/code/koopmans/src/koopmans/settings/_<calc>.py` and `/home/linsco_e/code/koopmans/src/koopmans/calculators/_<calc>.py`. Note which QE namelist / card or W90 keyword the legacy code writes it to.
3. **Locate the new wiring path**:
   - **Global flow:** `KoopmansInput` → `koopmans2/src/koopmans/aiida/workflows.py::_prepare_common_inputs` → `koopmans2/src/koopmans/aiida/conversion.py::input_to_pw_parameters` (or equivalent for other calculators) → `overrides` dict → `get_builder_from_protocol` → builder → `get_dict_from_builder` → `@task.graph` task inputs.
   - Identify where this field should slot into the `overrides` structure for its target WorkChain. For `pw.x` parameters, that's `overrides["<step>"]["pw"]["parameters"]["<NAMELIST>"][<KEY>]`. For Wannier90 parameters, `overrides["<step>"]["wannier90"]["parameters"]`. For builder-level settings (kpoints, pseudo), it's flatter.
4. **Check whether the wiring exists.** Grep for the field name in `koopmans2/src/koopmans/aiida/conversion.py` and in the corresponding `aiida-koopmans2/workgraphs/<step>.py`.
   - If wired: report the path (file:line → file:line → …) and stop. No code change.
   - If unwired: proceed to step 5.
5. **Write the wiring.** Minimal edit in `koopmans2/src/koopmans/aiida/conversion.py` (or the appropriate `_prepare_*` helper in `workflows.py`) to inject the field into the right `overrides` key. Follow the existing shape exactly — do not refactor the converter structure.
6. **Add a unit test** in `koopmans2/tests/unit/test_conversion.py` that constructs a `KoopmansInput` with the field set, runs the converter, and asserts the value appears at the expected overrides path.
7. **If the wiring requires translation** (e.g. a boolean field flips a QE namelist key to `"from_scratch"` vs `"restart"`), encode that translation in the converter — not in the workgraph — and document the mapping in a one-line comment.
8. **Summarize for the user**: the full path the value takes from Pydantic attribute to QE input, and whether any non-trivial translation happens along the way.

## What not to do

- **Don't bypass `conversion.py`** by threading the field as a new argument through `build_workgraph` into the workgraph. The converter is the single source of truth for input translation; don't fragment that.
- **Don't add the field to the workgraph's signature** just to expose it. Pydantic + overrides is the pattern.
- **Don't invent new overrides keys.** Match what `get_builder_from_protocol` expects upstream. If the upstream protocol doesn't accept it, the field needs to be applied post-builder via direct `data[...]` manipulation — flag this to the user; it's a smell.
