# shrinkvec: sift-128

1,000,000 vectors, 128 dimensions, l2, 1,000 queries, recall@10.

**Recommendation (recall@10 >= 0.95):** tq-mse-3>tq-mse-8 keeps recall@10 at 0.983 with 56 bytes per vector to search (9.1x smaller than float32). The rerank reads more precise data for each candidate, so 192 bytes per vector stay stored; only the searched part has to be in memory. In Postgres: pg_turbovec.

## Codec lab

| pipeline | recall@10 | B/vector searched | B/vector stored | vs float32 |
|---|---:|---:|---:|---:|
| `float32` | 1.000 | 512 | 512 | 1.0x |
| `halfvec` | 1.000 | 256 | 256 | 2.0x |
| `int8` | 0.990 | 128 | 128 | 4.0x |
| `tq-mse-4` | 0.814 | 72 | 72 | 7.1x |
| `tq-prod-4` | 0.339 | 72 | 72 | 7.1x |
| `tq-mse-3` | 0.671 | 56 | 56 | 9.1x |
| `tq-mse-2` | 0.456 | 40 | 40 | 12.8x |
| `rabitq-4` | 0.858 | 72 | 72 | 7.1x |
| `rabitq-1` | 0.252 | 24 | 24 | 21.3x |
| `binary` | 0.022 | 16 | 16 | 32.0x |
| `tq-mse-3>tq-mse-8` | 0.983 | 56 | 192 | 9.1x |
| `rabitq-1>float32` | 0.704 | 24 | 536 | 21.3x |
| `binary>float32` | 0.055 | 16 | 528 | 32.0x |

Reproduce: `shrinkvec lab --dataset sift-128 --repeats 3 --out docs/results/sift-128`
