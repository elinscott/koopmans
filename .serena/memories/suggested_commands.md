# Suggested commands

All from repo root, always through uv:

- `uv run koopmans run <input.json>` — CLI entry point.
- `uv run pytest` / `uv run pytest -m "not slow"` — tests.
- `uv run ruff check .` / `uv run ruff format .` — lint/format.
- `uv run mypy src` — type check.
- `uv run verdi -p koopmans process list -a | tail -10` — recent AiiDA processes (profile `koopmans`).
- `uv run verdi -p koopmans process report <pk> | tail -50` — failure diagnosis (exception at bottom).
- `uv run verdi -p koopmans calcjob outputcat <pk> <file>` — retrieved outputs.

Session runs inside a `nono` sandbox — writes outside granted paths fail; don't work around, ask the user.