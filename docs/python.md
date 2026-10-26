# Python API

Everything the command line does is plain Python underneath, so you can run the lab from a notebook
on vectors you already have in memory. This page covers the three entry points:

- `shrinkvec.lab.run_lab`: recall and bytes per vector for every codec, no database.
- `shrinkvec.codecs.make_codec`: one codec by name, to encode and score vectors yourself.
- `shrinkvec.pg.run_pg`: real indexes in Postgres (needs `pip install 'shrinkvec[pg]'` and a server).

The examples run in order as one script, and `tests/test_docs.py` runs them on every CI build, so
they stay in step with the code.

## The codec lab: `run_lab`

```python
import numpy as np

from shrinkvec.lab import run_lab

# 2,100 clustered random vectors in 64 dimensions; hold out the last 100 as queries.
rng = np.random.default_rng(0)
centers = rng.standard_normal((20, 64))
vectors = (centers[rng.integers(0, 20, 2100)] * 0.6 + rng.standard_normal((2100, 64))).astype(np.float32)
base, queries = vectors[:2000], vectors[2000:]

results, truth = run_lab(
    base,
    queries,
    metric="cosine",
    pipelines=["float32", "halfvec", "int8", "rabitq-4", "rabitq-1", "rabitq-1>float32"],
    k=10,
    log=lambda *_: None,  # the default, print, shows progress
)

for r in results:
    print(f"{r.pipeline:<18} recall@10 {r.recall:.3f}  {r.search_bytes:4.0f} B/vector  {r.compression:4.1f}x")

assert results[0].recall == 1.0  # float32 scans the vectors themselves
assert truth.shape == (len(queries), 10)
```

It prints something like this (the exact numbers depend on your data):

```text
float32            recall@10 1.000   256 B/vector   1.0x
halfvec            recall@10 0.999   128 B/vector   2.0x
int8               recall@10 0.993    64 B/vector   4.0x
rabitq-4           recall@10 0.901    40 B/vector   6.4x
rabitq-1           recall@10 0.402    16 B/vector  16.0x
rabitq-1>float32   recall@10 0.938    16 B/vector  16.0x
```

`run_lab(base, queries, metric="cosine", pipelines=None, k=10, candidates=100, seed=42, truth=None,
repeats=1, log=print)` takes:

| Argument | Meaning |
|---|---|
| `base`, `queries` | 2-D arrays, one vector per row. Any float dtype; they are converted to float32. |
| `metric` | `"cosine"` (vectors are normalised first, like pgvector's `vector_cosine_ops`) or `"l2"`. |
| `pipelines` | Codec names, or `search>rerank` pipelines (see below). `None` runs the 13 options in `shrinkvec.lab.DEFAULT_PIPELINES`. |
| `k` | Neighbours per query; the result is recall@k. |
| `candidates` | How many results a pipeline reranks when its name does not say (`rabitq-1>float32:200` does). |
| `seed` | Seed for the random rotations. |
| `truth` | Exact neighbours you already have, shape `(len(queries), k)`. Computed in float64 when `None`. |
| `repeats` | Rebuild codecs that use a random rotation with seeds `seed`, `seed + 1`, ... and report the mean and standard deviation. |
| `log` | Called with one string per progress line. |

It returns `(results, truth)`: a list of `LabResult`, one per pipeline in the order given, and the
exact neighbours, which you can pass back as `truth=` to try more pipelines on the same vectors
without recomputing them. Each `LabResult` has:

| Field | Meaning |
|---|---|
| `pipeline`, `label` | The name you passed, and a readable description. |
| `recall`, `recall_std`, `repeats` | Mean recall@k, its standard deviation over seeds, and how many runs it is over. |
| `candidate_recall` | For a rerank pipeline, the share of the true top k that made it into the candidates (the ceiling for the rerank); `None` otherwise. |
| `search_bytes` | Bytes per vector that the search scans. |
| `total_bytes` | Bytes per vector stored, including the rerank codec. |
| `compression` | float32 bytes divided by `search_bytes`. |
| `encode_seconds` | Time to encode the base vectors. |

### Pipelines

A pipeline is `search[>rerank[:candidates]]`. The search codec scans every code; with a rerank, the
best `candidates` are reordered by the second codec. Codec names are `float32`, `halfvec`, `int8`,
`binary`, `tq-mse-<1-8>`, `tq-prod-<2-8>` and `rabitq-<1-8>`.

```python
more, _ = run_lab(
    base,
    queries,
    pipelines=["binary>float32", "tq-mse-3>tq-mse-8:200"],
    truth=truth,  # reuse the exact neighbours from above
    log=lambda *_: None,
)
for r in more:
    assert r.recall <= r.candidate_recall  # a rerank cannot find what the search missed
```

### Picking the smallest option

`shrinkvec.recommend.from_lab` picks what the command line recommends: the option with the fewest
bytes to search among those that reach the target recall.

```python
from shrinkvec.recommend import from_lab

recommendation = from_lab(results, target=0.95)
print(recommendation.reason)
assert recommendation.choice is not None and recommendation.choice["recall"] >= 0.95
```

`shrinkvec.lab.as_dicts(results)` turns the results into plain dicts, ready for `json.dump` or
`pandas.DataFrame`.

## One codec: `make_codec`

`make_codec(name)` returns a fresh `Codec` for any name above. Build it on the base vectors, prepare
the queries once, then score. Higher scores are closer.

```python
from shrinkvec.codecs import make_codec
from shrinkvec.exact import prepare, recall_at_k, top_k

codec = make_codec("tq-mse-4")
# Codecs expect vectors already prepared for the metric: float32, unit length for cosine.
codec.build(prepare(base, "cosine"), "cosine", np.random.default_rng(42))
state = codec.prepare(prepare(queries, "cosine"))

scores = codec.scores(state, 0, codec.n)  # every query against base rows [0, n)
found = top_k(scores, 10)
print(codec.label, f"{codec.bytes_per_vector():.0f} B/vector", f"recall@10 {recall_at_k(found, truth, 10):.3f}")

assert scores.shape == (len(queries), len(base))
```

Other methods: `codec.scores_rows(state, qi, ids)` scores query `qi` against chosen rows (this is how
reranking works), and `codec.shared_bytes` is the state shared by all rows, such as a rotation
matrix, which `bytes_per_vector()` does not include. For many vectors, score in slices of rows as
`shrinkvec.lab.scan` does, rather than all at once.

## Real indexes: `run_pg`

`run_pg` builds each index in Postgres, sweeps its search setting and measures recall, latency and
size. It needs `pip install 'shrinkvec[pg]'` and a server with pgvector (VectorChord too, for the
`vchordrq` options). It copies the vectors into its own `shrinkvec` schema and never writes to your
tables. This example is not run by the tests, since it needs a database.

<!-- not tested: needs Postgres -->
```python
import os

from shrinkvec.pg import run_pg
from shrinkvec.recommend import from_pg

points, env = run_pg(
    os.environ["SHRINKVEC_DSN"],  # postgresql://user:password@host:5432/db
    base,
    queries,
    truth,  # exact neighbours, from run_lab or shrinkvec.exact.exact_neighbors
    "cosine",
    k=10,
    options=["hnsw-vector", "hnsw-halfvec", "hnsw-binary-rerank"],  # None builds every available index
)
for p in points:
    print(p.option, p.setting, f"recall@10 {p.recall:.3f}", f"p50 {p.p50_ms:.2f} ms", f"{p.index_bytes:,} bytes")

recommendation = from_pg(points, 0.95, "public.documents", env["dims"], "cosine", env["rows"])
print(recommendation.reason)
print(recommendation.sql)  # CREATE INDEX, SET and the query, for your table
```

`run_pg(dsn, base, queries, truth, metric, k=10, options=None, keep=False, log=print)` returns
`(points, env)`. Each `PgPoint` is one index at one setting, with `option`, `label`, `knob`, `value`,
`setting`, `recall`, `p50_ms`, `p95_ms`, `qps`, `index_bytes`, `bytes_per_vector` and
`build_seconds`. `env` records the server version, the extensions, the rows, the dimensions and the
table size in bytes. The options are `hnsw-vector`, `hnsw-halfvec`, `hnsw-binary-rerank`, `vchordrq`
and `vchordrq-codes`; ones whose extension the server lacks are skipped. With `keep=True` the table
and indexes stay in the `shrinkvec` schema afterwards.
