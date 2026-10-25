#!/bin/sh
# Entry point of the demo image: start Postgres (pgvector + VectorChord), run shrinkvec on real
# OpenAI embeddings, then serve the report.
set -eu

export PGDATA=/var/lib/postgresql/data
if [ ! -s "$PGDATA/PG_VERSION" ]; then
  gosu postgres initdb --auth=trust --username=postgres >/dev/null
fi
gosu postgres pg_ctl -D "$PGDATA" -w -l /tmp/postgres.log \
  -o "-c listen_addresses=127.0.0.1 -c shared_preload_libraries=vchord -c shared_buffers=1GB \
      -c max_parallel_maintenance_workers=4 -c max_wal_size=4GB" start >/dev/null

echo
echo "  shrinkvec demo: ${SHRINKVEC_ROWS} OpenAI embeddings (1536 dims), PostgreSQL with pgvector and VectorChord"
echo "  The report will be at http://localhost:8080 when the run finishes (a few minutes)."
echo

shrinkvec run --file /demo/dbpedia-openai.npy --dsn "postgresql://postgres@127.0.0.1/postgres" \
  -n "$SHRINKVEC_ROWS" --queries "$SHRINKVEC_QUERIES" --target "${SHRINKVEC_TARGET:-0.95}" --out /report "$@"

[ "${SHRINKVEC_NO_SERVE:-}" = "1" ] && exit 0
echo
echo "  Report: http://localhost:8080   (Ctrl+C to stop)"
cd /report && exec python3 -m http.server 8080 --bind 0.0.0.0 >/dev/null 2>&1
