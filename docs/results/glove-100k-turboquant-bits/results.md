# shrinkvec: glove-100

100,000 vectors, 100 dimensions, cosine, 1,000 queries, recall@10.

**Recommendation (recall@10 >= 0.95):** tq-mse-8 keeps recall@10 at 0.989 with 104 bytes per vector to search (3.8x smaller than float32). In Postgres: pg_turbovec.

## Codec lab

| pipeline | recall@10 | B/vector searched | B/vector stored | vs float32 |
|---|---:|---:|---:|---:|
| `tq-prod-9` | 0.983 | 117 | 117 | 3.4x |
| `tq-prod-5` | 0.806 | 67 | 67 | 6.0x |
| `tq-prod-4` | 0.667 | 55 | 55 | 7.3x |
| `tq-prod-3` | 0.478 | 42 | 42 | 9.5x |
| `tq-mse-8` | 0.989 | 104 | 104 | 3.8x |
| `tq-mse-5` | 0.925 | 67 | 67 | 6.0x |
| `tq-mse-4` | 0.867 | 54 | 54 | 7.4x |
| `tq-mse-3` | 0.753 | 42 | 42 | 9.5x |
| `tq-mse-2` | 0.559 | 29 | 29 | 13.8x |

Reproduce: `shrinkvec lab --dataset glove-100 -n 100000 --pipelines tq-prod-9,tq-prod-5,tq-prod-4,tq-prod-3,tq-mse-8,tq-mse-5,tq-mse-4,tq-mse-3,tq-mse-2 --out docs/results/glove-100k-turboquant-bits`
