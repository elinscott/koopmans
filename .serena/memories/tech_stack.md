# Tech stack

- Python (requires ≥3.12 in practice; aiida-core 2.7 line), package name `koopmans`, version 0.0.1-dev.
- Build: `uv_build`; package manager: **uv** (always `uv run …`).
- Key deps: pydantic (input schema), aiida-core, aiida-workgraph, aiida-quantumespresso, node-graph, pyyaml, seekpath.
- `[tool.uv.sources]`: local editable installs of siblings — `../aiida-koopmans2` (as `aiida-koopmans`), `../aiida-quantumespresso`, `../aiida-workgraph`, `../aiida-core`, `../node-graph`, `../wannier90-input`, `../pydantic_espresso`. Changes in those repos take effect without reinstall.
- Tooling: pytest (marker `slow`), ruff (line-length 100), mypy (typed package, `py.typed`), bumpversion, cruft.