# Contributing

Thanks for helping. Issues labelled `good first issue` say where to start in the code and when the
work is done; [ROADMAP.md](./ROADMAP.md) has the bigger picture. Comment on an issue before you
start, so two people do not do the same work.

## Setup

```bash
git clone https://github.com/ahmad-codex/shrinkvec && cd shrinkvec
uv sync
```

[uv](https://docs.astral.sh/uv/) installs the right Python and the dev tools. Plain `pip install -e
".[pg,datasets]"` works too.

## Checks

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -m "not crosscheck"           # unit tests, including the papers' numbers
uv run --group crosscheck pytest -m crosscheck   # against rabitqlib and turbovec
```

For the Postgres side, start a server with pgvector and VectorChord:

```bash
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres \
  ghcr.io/tensorchord/vchord-postgres:pg18-v1.1.1 -c shared_preload_libraries=vchord
export SHRINKVEC_DSN=postgresql://postgres:postgres@localhost:5432/postgres
uv run shrinkvec run --dataset glove-100 -n 20000 --queries 200
```

## What reviews check

- **Numbers you can trust.** A new codec comes with a test against the figures its paper publishes
  (distortion, bias, error bounds), and a cross-check when a reference implementation exists.
- **Honest byte counts.** Every per-row float a codec keeps is counted. Padding is counted.
- **Postgres stays untouched.** The tool reads your tables and writes only to its own `shrinkvec`
  schema.
- **Small, focused pull requests**, with `CHANGELOG.md` and `docs/method.md` updated when behavior
  or numbers change.

## Sharing results

Ran it on an embedding model that is not in the README? The "Share a result" issue form takes the
`results.md` file the report writes. Results on many models are the most useful contribution there
is.
