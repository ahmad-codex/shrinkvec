# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- [Python API page](./docs/python.md) for `run_lab`, `make_codec` and `run_pg`, with examples that
  run as a test.

## [0.1.0] - 2026-10-25

### Added

- Codec lab: exact recall@k and bytes per vector for float32, halfvec, int8, binary,
  TurboQuant (mse and prod, 1-8 bits), RaBitQ (1-bit and extended, 1-8 bits), and search-then-rerank
  pipelines. Ground truth in float64; `--repeats` reports mean and standard deviation over seeds.
- Tests against the figures the TurboQuant and RaBitQ papers publish, and cross-checks against
  rabitqlib and turbovec.
- Postgres runner: pgvector HNSW on `vector`, `halfvec` and `binary_quantize` with rerank, and
  VectorChord `vchordrq` (default, and codes only with `rerank_in_table`), measured for recall,
  p50 and p95 latency, size on disk and build time, in a scratch schema.
- Recommendation of the smallest option that meets a recall target, with the SQL to adopt it, and
  `--min-recall` for CI.
- Self-contained HTML report with charts, plus Markdown and JSON.
- Public datasets (DBpedia OpenAI embeddings, GloVe, SIFT) and loaders for `.npy`, `.fvecs`,
  `.parquet` and a Postgres table.
- One-command demo image with Postgres, pgvector, VectorChord and real OpenAI embeddings.

[Unreleased]: https://github.com/ahmad-codex/shrinkvec/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ahmad-codex/shrinkvec/releases/tag/v0.1.0
