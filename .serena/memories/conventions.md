# Conventions

- ruff, line-length 100; isort via ruff; pydocstyle rules configured.
- Fully typed; mypy enforced; package ships `py.typed`.
- Pydantic models for all user input live in `src/koopmans/input_file/` — extend, never duplicate as dicts.
- Dispatcher pattern (`aiida/workflows/`): one `build_<task>_workgraph` route module per `Task` enum value + per-workflow code loading via `load_codes(<Workflow>Codes)` inside the route; the package `__init__` dispatches and attaches input-file advice to plugin errors. Keep dispatch thin.
- Task outputs wired via dict access (`outputs["remote_folder"]`), never attribute access.
- Regression tests in `tests/regression/` consume tutorial JSONs from `../koopmans/tutorials/`; never mock AiiDA — throwaway profile fixtures.
- `@task`/`@task.graph` names must not start with underscore (AiiDA link-label restriction).
