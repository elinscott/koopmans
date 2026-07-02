# Conventions

- ruff, line-length 100; isort via ruff; pydocstyle rules configured.
- Fully typed; mypy enforced; package ships `py.typed`.
- Pydantic models for all user input live in `src/koopmans/input_file/` — extend, never duplicate as dicts.
- Dispatcher pattern (`aiida/workflows.py`): one `_build_<task>_workgraph` helper per `Task` enum value + code loading via `load_codes_for_task`. Keep dispatch thin.
- Task outputs wired via dict access (`outputs["remote_folder"]`), never attribute access.
- Regression tests in `tests/regression/` consume tutorial JSONs from `../koopmans/tutorials/`; never mock AiiDA — throwaway profile fixtures.
- `@task`/`@task.graph` names must not start with underscore (AiiDA link-label restriction).