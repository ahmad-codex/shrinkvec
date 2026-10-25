#!/bin/sh
# Regenerates every number in the README and docs/results/.
#
#   export SHRINKVEC_DSN=postgresql://postgres:postgres@localhost:5432/postgres   # pgvector + VectorChord
#   ./benchmarks/run.sh
#
# The Postgres server used for the published results:
#   docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres --shm-size=2g \
#     ghcr.io/tensorchord/vchord-postgres:pg18-v1.1.1 \
#     -c shared_preload_libraries=vchord -c shared_buffers=2GB -c max_parallel_maintenance_workers=4
set -eu

shrinkvec run --dataset dbpedia-openai-1536 -n 100000 --repeats 3 --out docs/results/dbpedia-openai-1536
shrinkvec lab --dataset glove-100 --repeats 3 --out docs/results/glove-100
shrinkvec lab --dataset sift-128 --repeats 3 --out docs/results/sift-128

# The correction bit: TurboQuant's prod estimator against all bits in the codes (docs, article).
shrinkvec lab --dataset glove-100 -n 100000 \
  --pipelines tq-prod-9,tq-prod-5,tq-prod-4,tq-prod-3,tq-mse-8,tq-mse-5,tq-mse-4,tq-mse-3,tq-mse-2 \
  --out docs/results/glove-100k-turboquant-bits
