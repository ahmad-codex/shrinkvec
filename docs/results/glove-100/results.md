# shrinkvec: glove-100

1,183,514 vectors, 100 dimensions, cosine, 1,000 queries, recall@10.

**Recommendation (recall@10 >= 0.95):** tq-mse-3>tq-mse-8 keeps recall@10 at 0.986 with 42 bytes per vector to search (9.5x smaller than float32). The rerank reads more precise data for each candidate, so 146 bytes per vector stay stored; only the searched part has to be in memory. In Postgres: pg_turbovec.

## Codec lab

| pipeline | recall@10 | B/vector searched | B/vector stored | vs float32 |
|---|---:|---:|---:|---:|
| `float32` | 1.000 | 400 | 400 | 1.0x |
| `halfvec` | 0.999 | 200 | 200 | 2.0x |
| `int8` | 0.981 | 100 | 100 | 4.0x |
| `tq-mse-4` | 0.855 | 54 | 54 | 7.4x |
| `tq-prod-4` | 0.631 | 55 | 55 | 7.3x |
| `tq-mse-3` | 0.737 | 42 | 42 | 9.5x |
| `tq-mse-2` | 0.539 | 29 | 29 | 13.8x |
| `rabitq-4` | 0.878 | 72 | 72 | 5.6x |
| `rabitq-1` | 0.329 | 24 | 24 | 16.7x |
| `binary` | 0.136 | 13 | 13 | 30.8x |
| `tq-mse-3>tq-mse-8` | 0.986 | 42 | 146 | 9.5x |
| `rabitq-1>float32` | 0.736 | 24 | 424 | 16.7x |
| `binary>float32` | 0.348 | 13 | 413 | 30.8x |

Reproduce: `shrinkvec lab --dataset glove-100 --repeats 3 --out docs/results/glove-100`
