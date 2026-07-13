# Task completion checklist

1. `uv run ruff check .` (and `uv run ruff format .` if files touched)
2. `uv run mypy src`
3. `uv run pytest -m "not slow"` (full `uv run pytest` for regression-touching changes)
4. If the change spans `../aiida-koopmans2`, run its tests too from this venv: `uv run pytest ../aiida-koopmans2/tests`.