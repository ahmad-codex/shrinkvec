## What and why

<!-- What this changes and the decision it helps with. Link the issue if there is one. -->

## How it was tested

<!-- Commands run, new or changed tests, datasets you ran it on. -->

## Checklist

- [ ] `uv run ruff check src tests` and `uv run ruff format --check src tests` pass
- [ ] `uv run pytest -m "not crosscheck"` passes
- [ ] A new or changed codec has a test against its paper's numbers, and `pytest -m crosscheck` passes when a reference implementation exists
- [ ] Byte counts include every per-row float the codec stores
- [ ] No connection string, password or private data in code, tests or reports
- [ ] `docs/method.md` and `CHANGELOG.md` updated where behavior or numbers change
