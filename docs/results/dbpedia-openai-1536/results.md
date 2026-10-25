# shrinkvec: dbpedia-openai-1536

100,000 vectors, 1536 dimensions, cosine, 1,000 queries, recall@10.

**Recommendation (recall@10 >= 0.95):** VectorChord vchordrq, codes only (rerank_in_table), vchordrq.probes=20: recall@10 0.957, p50 1.69 ms, index 30.6 MiB (321 bytes per vector). The smallest index that meets 0.95.

```sql
CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
CREATE INDEX ON items USING vchordrq (embedding vector_cosine_ops) WITH (options = $$
rerank_in_table = true
residual_quantization = true
[build.internal]
lists = [200]
spherical_centroids = true
$$);
SET vchordrq.probes = '20';
SELECT id FROM items ORDER BY embedding <=> $1 LIMIT 10;
```

## Codec lab

| pipeline | recall@10 | B/vector searched | B/vector stored | vs float32 |
|---|---:|---:|---:|---:|
| `float32` | 1.000 | 6,144 | 6,144 | 1.0x |
| `halfvec` | 0.999 | 3,072 | 3,072 | 2.0x |
| `int8` | 0.994 | 1,536 | 1,536 | 4.0x |
| `tq-mse-4` | 0.953 | 772 | 772 | 8.0x |
| `tq-prod-4` | 0.817 | 772 | 772 | 8.0x |
| `tq-mse-3` | 0.916 | 580 | 580 | 10.6x |
| `tq-mse-2` | 0.850 | 388 | 388 | 15.8x |
| `rabitq-4` | 0.972 | 776 | 776 | 7.9x |
| `rabitq-1` | 0.834 | 200 | 200 | 30.7x |
| `binary` | 0.707 | 192 | 192 | 32.0x |
| `tq-mse-3>tq-mse-8` | 0.996 | 580 | 2,120 | 10.6x |
| `rabitq-1>float32` | 0.999 | 200 | 6,344 | 30.7x |
| `binary>float32` | 0.992 | 192 | 6,336 | 32.0x |

## Postgres

| index | setting | recall@10 | p50 ms | p95 ms | index MiB | build s |
|---|---|---:|---:|---:|---:|---:|
| `hnsw-vector` | hnsw.ef_search=40 | 0.941 | 1.74 | 2.23 | 781.3 | 83.1 |
| `hnsw-vector` | hnsw.ef_search=80 | 0.959 | 2.34 | 3.07 | 781.3 | 83.1 |
| `hnsw-vector` | hnsw.ef_search=160 | 0.972 | 3.55 | 4.51 | 781.3 | 83.1 |
| `hnsw-vector` | hnsw.ef_search=320 | 0.982 | 5.83 | 8.60 | 781.3 | 83.1 |
| `hnsw-halfvec` | hnsw.ef_search=40 | 0.940 | 1.73 | 2.27 | 390.6 | 70.6 |
| `hnsw-halfvec` | hnsw.ef_search=80 | 0.957 | 2.31 | 3.89 | 390.6 | 70.6 |
| `hnsw-halfvec` | hnsw.ef_search=160 | 0.969 | 3.71 | 5.41 | 390.6 | 70.6 |
| `hnsw-halfvec` | hnsw.ef_search=320 | 0.978 | 5.35 | 7.10 | 390.6 | 70.6 |
| `hnsw-binary-rerank` | candidates=100 | 0.963 | 2.83 | 4.97 | 47.9 | 32.6 |
| `hnsw-binary-rerank` | candidates=200 | 0.977 | 4.98 | 7.07 | 47.9 | 32.6 |
| `hnsw-binary-rerank` | candidates=400 | 0.986 | 7.65 | 9.80 | 47.9 | 32.6 |
| `hnsw-binary-rerank` | candidates=800 | 0.994 | 13.69 | 16.52 | 47.9 | 32.6 |
| `vchordrq` | vchordrq.probes=4 | 0.854 | 1.04 | 1.42 | 812.4 | 13.7 |
| `vchordrq` | vchordrq.probes=10 | 0.916 | 1.31 | 1.77 | 812.4 | 13.7 |
| `vchordrq` | vchordrq.probes=20 | 0.949 | 1.71 | 2.00 | 812.4 | 13.7 |
| `vchordrq` | vchordrq.probes=40 | 0.981 | 2.58 | 3.27 | 812.4 | 13.7 |
| `vchordrq-codes` | vchordrq.probes=4 | 0.853 | 1.07 | 1.26 | 30.6 | 10.6 |
| `vchordrq-codes` | vchordrq.probes=10 | 0.920 | 1.30 | 1.51 | 30.6 | 10.6 |
| `vchordrq-codes` | vchordrq.probes=20 | 0.957 | 1.69 | 2.01 | 30.6 | 10.6 |
| `vchordrq-codes` | vchordrq.probes=40 | 0.985 | 2.43 | 2.87 | 30.6 | 10.6 |

Reproduce: `shrinkvec run --dataset dbpedia-openai-1536 -n 100000 --repeats 3 --out docs/results/dbpedia-openai-1536`
