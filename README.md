# shrinkvec

[![ci](https://github.com/ahmad-codex/shrinkvec/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ahmad-codex/shrinkvec/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/shrinkvec.svg)](https://pypi.org/project/shrinkvec/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**How small can your embeddings get before search gets worse?** shrinkvec measures every
compression option on *your* vectors, builds the real indexes in *your* Postgres, and tells you the
smallest one that keeps your recall, with the SQL to switch.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/report-dark.png">
  <img alt="shrinkvec report on 100,000 OpenAI embeddings: the recommended index with its SQL, and recall against bytes per vector for 13 compression options" src="docs/images/report-light.png">
</picture>

Float32 embeddings are big: 6 KB per row at 1536 dimensions, so 100 million rows is 600 GB before
any index. pgvector's `halfvec` and `binary_quantize`, VectorChord's RaBitQ and the new TurboQuant
extensions all promise to shrink that. How much recall each one costs depends on your embedding
model, and published numbers are mostly measured by the vendors on their own data. shrinkvec
answers it on your data, in a few minutes, with exact ground truth.

## Try it in one command

```bash
docker run --rm -p 8080:8080 ghcr.io/ahmad-codex/shrinkvec:demo
```

It starts Postgres with pgvector and VectorChord, loads 25,000 real OpenAI embeddings, measures 13
compression options and 5 index types, and serves the report at http://localhost:8080. It takes
about three minutes on a laptop.

## On your own data

```bash
pip install 'shrinkvec[pg]'
export SHRINKVEC_DSN=postgresql://user:password@host:5432/db

shrinkvec run --table public.documents --column embedding --target 0.95
```

It samples 100,000 rows (`-n` to change), holds out 1,000 of them as queries, computes the exact
neighbours, and then:

1. **Codec lab**: the recall of every compression option, scanning every code exactly, so the
   numbers measure the compression and not an index. No extension needed.
2. **Real indexes**: builds pgvector HNSW on `vector`, `halfvec` and `binary_quantize` with rerank,
   and VectorChord `vchordrq` (with and without the vectors in the index) when it is installed, in
   its own `shrinkvec` schema. It never writes to your tables. For each index it sweeps the search
   setting and records recall, p50/p95 latency, size on disk and build time.
3. **Recommendation**: the smallest index that keeps recall@10 at or above `--target`, and the
   exact `CREATE INDEX`, `SET` and query to use it.

Vectors in a file work too, without a database: `shrinkvec lab --file vectors.npy` (also `.fvecs`
and `.parquet`).

## Results on public data

The same 13 options on three datasets, scanning every code (no index), with 1,000 held-out
queries and exact ground truth. Codecs with a random rotation ran with 3 seeds; ± is the standard
deviation.

| Option | dbpedia-openai-1536<br>recall@10 | glove-100<br>recall@10 | sift-128<br>recall@10 | Bytes/vector searched (1536 d) |
|---|---:|---:|---:|---:|
| `float32` | 1.000 | 1.000 | 1.000 | 6,144 |
| `halfvec` | 0.999 | 0.999 | 1.000 | 3,072 |
| `int8` | 0.994 | 0.981 | 0.990 | 1,536 |
| `tq-mse-4` | 0.953 ± 0.001 | 0.855 ± 0.003 | 0.814 ± 0.001 | 772 |
| `tq-prod-4` | 0.817 ± 0.003 | 0.631 ± 0.003 | 0.339 ± 0.005 | 772 |
| `tq-mse-3` | 0.916 ± 0.002 | 0.737 ± 0.002 | 0.671 ± 0.003 | 580 |
| `tq-mse-2` | 0.850 ± 0.000 | 0.539 ± 0.003 | 0.456 ± 0.005 | 388 |
| `rabitq-4` | 0.972 ± 0.001 | 0.878 ± 0.001 | 0.858 ± 0.001 | 776 |
| `rabitq-1` | 0.834 ± 0.003 | 0.329 ± 0.003 | 0.252 ± 0.002 | 200 |
| `binary` | 0.707 | 0.136 | 0.022 | 192 |
| `tq-mse-3>tq-mse-8` | 0.996 ± 0.000 | 0.986 ± 0.000 | 0.983 ± 0.001 | 580 |
| `rabitq-1>float32` | 0.999 ± 0.000 | 0.736 ± 0.001 | 0.704 ± 0.006 | 200 |
| `binary>float32` | 0.992 | 0.348 | 0.055 | 192 |

| Index | Size on disk | First setting with recall@10 ≥ 0.95 | p50 there | Best recall@10 |
|---|---:|---|---:|---:|
| `hnsw-vector` | 781 MiB | hnsw.ef_search=80 (0.959) | 2.34 ms | 0.982 |
| `hnsw-halfvec` | 391 MiB | hnsw.ef_search=80 (0.957) | 2.31 ms | 0.978 |
| `hnsw-binary-rerank` | 48 MiB | candidates=100 (0.963) | 2.83 ms | 0.994 |
| `vchordrq` | 812 MiB | vchordrq.probes=40 (0.981) | 2.58 ms | 0.981 |
| `vchordrq-codes` | 31 MiB | vchordrq.probes=20 (0.957) | 1.69 ms | 0.985 |

| Index | Setting | recall@10 | p50 ms | p95 ms | Index size | Build |
|---|---|---:|---:|---:|---:|---:|
| `hnsw-vector` | hnsw.ef_search=40 | 0.941 | 1.74 | 2.23 | 781 MiB | 83 s |
| `hnsw-vector` | hnsw.ef_search=80 | 0.959 | 2.34 | 3.07 | 781 MiB | 83 s |
| `hnsw-vector` | hnsw.ef_search=160 | 0.972 | 3.55 | 4.51 | 781 MiB | 83 s |
| `hnsw-vector` | hnsw.ef_search=320 | 0.982 | 5.83 | 8.60 | 781 MiB | 83 s |
| `hnsw-halfvec` | hnsw.ef_search=40 | 0.940 | 1.73 | 2.27 | 391 MiB | 71 s |
| `hnsw-halfvec` | hnsw.ef_search=80 | 0.957 | 2.31 | 3.89 | 391 MiB | 71 s |
| `hnsw-halfvec` | hnsw.ef_search=160 | 0.969 | 3.71 | 5.41 | 391 MiB | 71 s |
| `hnsw-halfvec` | hnsw.ef_search=320 | 0.978 | 5.35 | 7.10 | 391 MiB | 71 s |
| `hnsw-binary-rerank` | candidates=100 | 0.963 | 2.83 | 4.97 | 48 MiB | 33 s |
| `hnsw-binary-rerank` | candidates=200 | 0.977 | 4.98 | 7.07 | 48 MiB | 33 s |
| `hnsw-binary-rerank` | candidates=400 | 0.986 | 7.65 | 9.80 | 48 MiB | 33 s |
| `hnsw-binary-rerank` | candidates=800 | 0.994 | 13.69 | 16.52 | 48 MiB | 33 s |
| `vchordrq` | vchordrq.probes=4 | 0.854 | 1.04 | 1.42 | 812 MiB | 14 s |
| `vchordrq` | vchordrq.probes=10 | 0.916 | 1.31 | 1.77 | 812 MiB | 14 s |
| `vchordrq` | vchordrq.probes=20 | 0.949 | 1.71 | 2.00 | 812 MiB | 14 s |
| `vchordrq` | vchordrq.probes=40 | 0.981 | 2.58 | 3.27 | 812 MiB | 14 s |
| `vchordrq-codes` | vchordrq.probes=4 | 0.853 | 1.07 | 1.26 | 31 MiB | 11 s |
| `vchordrq-codes` | vchordrq.probes=10 | 0.920 | 1.30 | 1.51 | 31 MiB | 11 s |
| `vchordrq-codes` | vchordrq.probes=20 | 0.957 | 1.69 | 2.01 | 31 MiB | 11 s |
| `vchordrq-codes` | vchordrq.probes=40 | 0.985 | 2.43 | 2.87 | 31 MiB | 11 s |

Table (the vectors themselves): 795 MiB. PostgreSQL 18.3, vector 0.8.2, vchord 1.1.1. Windows AMD64, 12 threads, Python 3.12.13.

Bytes are for 1536 dimensions. Full results, with the reports' charts as Markdown:
[DBpedia OpenAI](./docs/results/dbpedia-openai-1536/results.md) ·
[GloVe-100](./docs/results/glove-100/results.md) · [SIFT-128](./docs/results/sift-128/results.md).

What the numbers say:

- **The same compression does not transfer between models.** pgvector's `binary_quantize` with a
  rerank keeps 0.992 of the true top 10 on OpenAI embeddings, 0.348 on GloVe and 0.055 on SIFT,
  whose values are all positive, so a sign bit carries almost nothing. Vendor numbers on someone
  else's embeddings do not tell you what yours will do.
- **RaBitQ beats a plain sign bit at the same size.** 0.834 against 0.707 on OpenAI embeddings for 8
  more bytes per row, and at 4 bits it beats TurboQuant on all three datasets.
- **For search, spend TurboQuant's bits on the codes.** The paper's unbiased variant (`tq-prod`)
  gives up one code bit for a correction sketch; at the same 4 bits it loses 14 to 48 points of
  recall to the plain codes (`tq-mse`). Unbiased inner products do not make a better ranking.
- **Search small, rerank precise.** 3-bit codes reranked with 8-bit codes keep 0.983 or more on
  every dataset while searching 10.6 times less data than float32.

### In Postgres

DBpedia OpenAI embeddings, 100,000 rows, 1,000 queries one at a time from one client, each index
built with its defaults and then swept over its search setting. Postgres 18.3 with pgvector 0.8.2
and VectorChord 1.1.1 in Docker on a 6-core laptop (Intel i7-10750H).



<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/report-postgres-dark.png">
  <img alt="Recall against p50 latency for five Postgres indexes on 100,000 OpenAI embeddings" src="docs/images/report-postgres-light.png">
</picture>

The table itself (the float32 vectors) is 795 MiB. VectorChord keeps a second copy of every vector
inside its index by default; with `rerank_in_table = true` the index holds only RaBitQ codes and
reranks from the table, which here made it both the smallest and the fastest index that reaches
0.95. VectorChord warns that this mode costs latency when the table has to come from disk: in this
run the whole table fits in shared_buffers, so measure it on your own data size. Every setting, with
p95 and build times, is in [results.md](./docs/results/dbpedia-openai-1536/results.md).

## What it measures, and how it was checked

Every result is recall@10 against exact search in float64 over held-out queries. Every byte a codec
stores per row is counted, including norms, correction factors and padding.

The TurboQuant and RaBitQ codecs are written from their papers and checked against them in the
test suite: TurboQuant's distortion at 1 to 4 bits matches the paper (0.363, 0.117, 0.0345, 0.0095),
its prod estimator is unbiased, RaBitQ's estimator is unbiased and its error bound holds at the
rate the theory predicts. Recall is also cross-checked against the reference libraries on the same
vectors: within 0.008 of the official [rabitqlib](https://github.com/VectorDB-NTU/RaBitQ-Library)
at 1, 4 and 7 bits, and within 0.003 of [turbovec](https://github.com/RyanCodrai/turbovec) at 2 and
3 bits (0.014 higher at 4 bits). The details, and what the numbers do not cover, are in
[docs/method.md](./docs/method.md).

| Codec | Paper | In Postgres |
|---|---|---|
| `halfvec` | | pgvector |
| `binary` (+ rerank) | | pgvector `binary_quantize` |
| `int8` | | not in pgvector |
| `rabitq-1`, `rabitq-<b>` | [RaBitQ](https://arxiv.org/abs/2405.12497) (SIGMOD 2024), [extended](https://arxiv.org/abs/2409.09913) (SIGMOD 2025) | [VectorChord](https://github.com/tensorchord/VectorChord) |
| `tq-mse-<b>`, `tq-prod-<b>` | [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) | [pg_turbovec](https://github.com/gburd/pg_turbovec) |

Pipelines combine them: `rabitq-1>float32` searches the 1-bit codes and reranks the best 100 with
the full vectors; `tq-mse-3>tq-mse-8:200` reranks 200 with 8-bit codes.

## Use it in CI

Fail a build when a model or codec change drops recall below a floor:

```bash
shrinkvec lab --file eval/embeddings.npy --pipelines 'halfvec,rabitq-1>float32' --min-recall 0.97
```

The exit code is 1 when any listed option falls below the floor, and the report is written either
way.

## Commands

| Command | What it does |
|---|---|
| `shrinkvec lab` | Codec lab only, no database |
| `shrinkvec pg` | Real indexes only |
| `shrinkvec run` | Both, when `--dsn` is set, plus the recommendation |
| `shrinkvec datasets` | The public datasets |

Common options: `--dataset`, `--file`, `--table`/`--column`, `-n`, `--queries`, `-k`, `--target`,
`--min-recall`, `--pipelines`, `--indexes`, `--repeats`, `--seed`, `--out`. `shrinkvec <command>
--help` lists them all.

Every run writes `report.html` (self-contained, charts, light and dark), `results.md` and
`results.json`.

## Development

```bash
uv sync
uv run pytest -m "not crosscheck"
uv run --group crosscheck pytest -m crosscheck
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) and [ROADMAP.md](./ROADMAP.md). Next up: dimension
truncation for Matryoshka models, recall under filters, IVFFlat, pgvectorscale and pg_turbovec.
Results on other embedding models are welcome through the "Share a result" issue form.

## License

[MIT](./LICENSE)
