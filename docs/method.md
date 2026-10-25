# How shrinkvec measures

Compression results are easy to get wrong in ways that look plausible. This page states exactly
what is measured, how each codec was checked against its paper and a reference implementation, and
what the numbers do not cover.

## Ground truth and recall

- **Queries are held out.** For your own vectors and for DBpedia, random rows are removed from the
  base and used as queries, so no query finds itself. GloVe and SIFT use their published query sets.
- **Exact neighbours in float64.** The true top-k is computed by brute force in float64, so float32
  rounding cannot reorder near ties. Ties go to the lower row id.
- **recall@k** is the mean, over queries, of |found top-k ∩ true top-k| / k.
- **Cosine** search runs on unit-length vectors, which ranks exactly like pgvector's
  `vector_cosine_ops`.
- **Deterministic.** Every random choice (rotations, sketches, query sampling) comes from `--seed`.
  With `--repeats N`, codecs that use a random rotation are rebuilt with N seeds and the table shows
  the mean and the standard deviation.

## The codec lab

Every option scans **all** codes. That isolates what the compression costs from what an index costs
(HNSW and IVF each lose some recall of their own). A pipeline such as `rabitq-1>float32` scans all
1-bit codes, keeps the best `--candidates` (100 by default) and reorders them with the second codec.

Bytes per vector include everything stored per row: codes plus any per-row floats. Shared state
(rotation matrices, codebooks) is reported separately and is not counted per row.

| Codec | What it stores per row | Reference |
|---|---|---|
| `float32` | d floats | pgvector `vector` |
| `halfvec` | d half floats | pgvector `halfvec` |
| `int8` | d bytes, min/max per dimension shared | scalar quantization |
| `binary` | d sign bits, compared by Hamming distance | pgvector `binary_quantize` |
| `tq-mse-b` | b bits per dimension + 1 float (decoded norm) | TurboQuant_mse, [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) |
| `tq-prod-b` | b-1 bits per dimension + d QJL sign bits + 1 float | TurboQuant_prod, same paper |
| `rabitq-b` | b bits per dimension (padded to 64) + 2 floats | RaBitQ ([2405.12497](https://arxiv.org/abs/2405.12497)), extended RaBitQ ([2409.09913](https://arxiv.org/abs/2409.09913)) |

Two implementation choices matter for the numbers:

- **TurboQuant mse scores against the decoded vector rescaled to unit length.** The decoded length
  varies by about 1% between rows, and leaving that noise in costs up to 11 points of recall@10 at
  2 bits. turbovec does the same; it costs one float per row and is counted.
- **RaBitQ pads the dimension to a multiple of 64**, as the reference library does. At 100
  dimensions that is 128 bits per vector at 1 bit, and it is counted.

The lab uses one RaBitQ centroid (the data mean) and an unquantized query. VectorChord uses many
centroids and a 4-bit query; the RaBitQ paper reports both effects as small next to the code itself.

## How the codecs were checked

All of these run in the test suite (`pytest`), against the numbers the papers publish. They use 2,000
random unit vectors at 1,536 dimensions:

| Check | Expected | Measured |
|---|---|---|
| TurboQuant codebook, 2 bits, ×√d | ±0.453, ±1.51 (paper) | ±0.4528, ±1.510 |
| TurboQuant_mse distortion, 1-4 bits | 0.36, 0.117, 0.03, 0.009 (paper) | 0.363, 0.1172, 0.0345, 0.0095 |
| TurboQuant_mse distortion, 8 bits | ≤ √3·π/2 · 4⁻⁸ = 4.15e-5 (paper bound) | 4.12e-5 |
| TurboQuant_prod inner product, 2 bits | unbiased (slope 1) | slope 0.994, mean error within noise |
| TurboQuant_mse inner product, 2 bits | biased towards 0 | slope 0.891 |
| RaBitQ ⟨ō, o⟩ at 1 bit | √(2/π) = 0.798 | 0.798 |
| RaBitQ estimator | unbiased | mean error within noise at 1, 2, 4 bits |
| RaBitQ error bound, ε₀ = 1.9 | exceeded with probability P(\|Z\| > 1.9) ≈ 5.7% | 5.6%, 6.1%, 4.7% at 1, 2, 4 bits |
| Extended RaBitQ code | maximises the cosine over the grid | equal to brute force over every code |

And against the reference implementations on the same vectors (`pytest -m crosscheck`, flat scan,
recall@10):

| Data | Codec | shrinkvec | Reference |
|---|---|---|---|
| GloVe-100, 100k | RaBitQ 1-bit | 0.362 to 0.366 (3 seeds) | 0.364 (rabitqlib 0.7.1) |
| GloVe-100, 100k | RaBitQ 4-bit | 0.884 to 0.891 | 0.883 |
| GloVe-100, 100k | RaBitQ 7-bit | 0.985 | 0.982 |
| DBpedia OpenAI, 100k | TurboQuant 2-bit | 0.848 | 0.846 (turbovec 1.1.2) |
| DBpedia OpenAI, 100k | TurboQuant 3-bit | 0.916 | 0.913 |
| DBpedia OpenAI, 100k | TurboQuant 4-bit | 0.953 | 0.939 |

turbovec stores 3-bit codes in 4-bit slots, so its 3-bit index costs as much as its 4-bit one.

## Postgres

`shrinkvec pg` copies the vectors into a scratch schema (`shrinkvec`), never into your tables.
For each index type the server supports:

1. Build the index and record the build time and `pg_relation_size`.
2. For each value of the search knob, run every query once to warm the cache, then once more timed
   from a single client, with prepared statements. Latency is the client-side wall time, including
   the round trip.
3. Compare the returned ids with the exact ground truth.
4. Drop the index.

| Index | Built as | Knob swept |
|---|---|---|
| `hnsw-vector` | pgvector HNSW, default m = 16, ef_construction = 64 | `hnsw.ef_search` 40, 80, 160, 320 |
| `hnsw-halfvec` | HNSW on `embedding::halfvec(d)` (expression index) | `hnsw.ef_search` |
| `hnsw-binary-rerank` | HNSW on `binary_quantize(embedding)::bit(d)`, rerank by the exact distance | candidates 100 to 800 |
| `vchordrq` | VectorChord IVF + RaBitQ, lists per VectorChord's guidance (none under 100k rows, N/500 to 2M); the index also keeps every vector for the rerank (VectorChord's default) | `vchordrq.probes` 2%, 5%, 10%, 20% of lists |
| `vchordrq-codes` | the same with `rerank_in_table = true`: the index keeps only the RaBitQ codes and reranks from the table | `vchordrq.probes` |

The session sets `maintenance_work_mem = 2GB`, `max_parallel_maintenance_workers = 4` and
`jit = off` where the server allows it.

## What the numbers do not cover

- **Throughput under load.** Latency is one client at a time. Concurrent load, filters and writes
  during search change the picture.
- **Cold cache.** Every query runs once before it is timed.
- **Your hardware.** Absolute latencies depend on the machine; the recall and size numbers do not.
- **Speed of the lab codecs.** The lab is NumPy and measures recall and size only. Real
  implementations use integer kernels and SIMD.
