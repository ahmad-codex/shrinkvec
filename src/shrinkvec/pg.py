"""Real indexes in a real Postgres: recall, latency, size on disk and build time.

Works in its own schema on a copy of the vectors; it never changes your tables. Every option that
the server supports is built in turn, measured over a sweep of its search knob, and dropped.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from .exact import Metric, recall_at_k

SCHEMA = "shrinkvec"
WARMUP = 50


@dataclass
class Option:
    key: str
    label: str
    extension: str
    create: Callable[[str, int, Metric, int], str]
    query: Callable[[str, int, Metric], str]
    knob: str
    values: Callable[[int], list[int]]
    params: Callable[[np.ndarray, int, int], tuple] = lambda q, k, v: (q, k)
    settings: Callable[[int], list[str]] = field(default=lambda v: [])


@dataclass
class PgPoint:
    option: str
    label: str
    knob: str
    value: int
    setting: str
    recall: float
    p50_ms: float
    p95_ms: float
    qps: float
    index_bytes: int
    bytes_per_vector: float
    build_seconds: float


def _ops(metric: Metric) -> tuple[str, str]:
    return ("cosine", "<=>") if metric == "cosine" else ("l2", "<->")


def _vchord_lists(n: int) -> int:
    """VectorChord's guidance: no partitions under 100k rows, N/500 up to 2M, then 4 * sqrt(N)."""
    if n < 100_000:
        return 0
    return n // 500 if n < 2_000_000 else int(4 * math.sqrt(n))


def _vchord_create(rerank_in_table: bool) -> Callable[[str, int, Metric, int], str]:
    """vchordrq as VectorChord recommends it. By default the index keeps a copy of every vector for
    the rerank; with ``rerank_in_table`` it keeps only the RaBitQ codes and reads vectors from the table."""

    def create(t: str, d: int, metric: Metric, n: int) -> str:
        op, _ = _ops(metric)
        lists = _vchord_lists(n)
        options = ["rerank_in_table = true"] if rerank_in_table else []
        options += ["residual_quantization = true"] if metric == "cosine" else []
        options += ["[build.internal]", f"lists = [{lists}]" if lists else "lists = []"]
        if metric == "cosine" and lists:
            options.append("spherical_centroids = true")
        body = "\n".join(options)
        return f"CREATE INDEX {{idx}} ON {t} USING vchordrq (embedding vector_{op}_ops) WITH (options = $$\n{body}\n$$)"

    return create


def _vchord_option(key: str, label: str, rerank_in_table: bool) -> Option:
    return Option(
        key,
        label,
        "vchord",
        _vchord_create(rerank_in_table),
        lambda t, d, m: f"SELECT id FROM {t} ORDER BY embedding {_ops(m)[1]} %s LIMIT %s",
        "vchordrq.probes",
        lambda n: [max(1, _vchord_lists(n) * p // 100) for p in (2, 5, 10, 20)] if _vchord_lists(n) else [0],
        settings=lambda v: [f"SET vchordrq.probes = '{v}'"] if v else [],
    )


OPTIONS = [
    Option(
        "hnsw-vector",
        "pgvector HNSW on vector (float32)",
        "vector",
        lambda t, d, m, n: f"CREATE INDEX {{idx}} ON {t} USING hnsw (embedding vector_{_ops(m)[0]}_ops)",
        lambda t, d, m: f"SELECT id FROM {t} ORDER BY embedding {_ops(m)[1]} %s LIMIT %s",
        "hnsw.ef_search",
        lambda n: [40, 80, 160, 320],
        settings=lambda v: [f"SET hnsw.ef_search = {v}"],
    ),
    Option(
        "hnsw-halfvec",
        "pgvector HNSW on halfvec (expression index)",
        "vector",
        lambda t, d, m, n: (
            f"CREATE INDEX {{idx}} ON {t} USING hnsw ((embedding::halfvec({d})) halfvec_{_ops(m)[0]}_ops)"
        ),
        lambda t, d, m: f"SELECT id FROM {t} ORDER BY embedding::halfvec({d}) {_ops(m)[1]} %s::halfvec({d}) LIMIT %s",
        "hnsw.ef_search",
        lambda n: [40, 80, 160, 320],
        settings=lambda v: [f"SET hnsw.ef_search = {v}"],
    ),
    Option(
        "hnsw-binary-rerank",
        "pgvector HNSW on binary_quantize, rerank with vector",
        "vector",
        lambda t, d, m, n: (
            f"CREATE INDEX {{idx}} ON {t} USING hnsw ((binary_quantize(embedding)::bit({d})) bit_hamming_ops)"
        ),
        lambda t, d, m: (
            f"SELECT id FROM (SELECT id, embedding FROM {t} ORDER BY binary_quantize(embedding)::bit({d}) <~> "
            f"binary_quantize(%s::vector) LIMIT %s) c ORDER BY embedding {_ops(m)[1]} %s LIMIT %s"
        ),
        "candidates",
        lambda n: [100, 200, 400, 800],
        params=lambda q, k, v: (q, v, q, k),
        settings=lambda v: [f"SET hnsw.ef_search = {min(v, 1000)}"],
    ),
    _vchord_option("vchordrq", "VectorChord vchordrq (IVF + RaBitQ, vectors kept in the index)", False),
    _vchord_option("vchordrq-codes", "VectorChord vchordrq, codes only (rerank_in_table)", True),
]


def connect(dsn: str):
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as e:
        raise SystemExit("this needs psycopg and pgvector: pip install 'shrinkvec[pg]'") from e
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def available(conn) -> dict[str, str]:
    """Installed or installable extensions and their versions."""
    rows = conn.execute(
        "SELECT name, coalesce(installed_version, default_version) FROM pg_available_extensions "
        "WHERE name IN ('vector', 'vchord')"
    ).fetchall()
    return dict(rows)


def sample_table(dsn: str, table: str, column: str, n: int, seed: int) -> np.ndarray:
    """A reproducible random sample of a vector column. Read-only."""
    from psycopg import sql

    conn = connect(dsn)
    ident = sql.Identifier(*table.split("."))
    conn.execute("SELECT setseed(%s)", (((seed % 1000) / 1000),))
    rows = conn.execute(
        sql.SQL("SELECT {c}::real[] FROM {t} WHERE {c} IS NOT NULL ORDER BY random() LIMIT %s").format(
            c=sql.Identifier(column), t=ident
        ),
        (n,),
    ).fetchall()
    conn.close()
    if not rows:
        raise SystemExit(f"no vectors found in {table}.{column}")
    return np.asarray([r[0] for r in rows], dtype=np.float32)


def load(conn, base: np.ndarray, table: str, log=print) -> None:
    d = base.shape[1]
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(f"CREATE TABLE {table} (id int4 PRIMARY KEY, embedding vector({d}))")
    log(f"loading {len(base):,} vectors into {table}")
    with conn.cursor().copy(f"COPY {table} (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
        copy.set_types(["int4", "vector"])
        for i, row in enumerate(base):
            copy.write_row((i, row))
    conn.execute(f"VACUUM ANALYZE {table}")


def _session(conn) -> None:
    for stmt in (
        "SET maintenance_work_mem = '2GB'",
        "SET max_parallel_maintenance_workers = 4",
        "SET jit = off",
    ):
        # A managed server may refuse some of these; measure anyway with its defaults.
        with contextlib.suppress(Exception):
            conn.execute(stmt)


def _setting(option: Option, value: int) -> str:
    if option.extension == "vchord" and value == 0:
        return "flat scan (lists = [])"
    return f"{option.knob}={value}"


def _measure(conn, sql_text: str, option: Option, queries: np.ndarray, truth: np.ndarray, k: int, value: int):
    for stmt in option.settings(value):
        conn.execute(stmt)
    cur = conn.cursor()
    for q in queries[:WARMUP]:
        cur.execute(sql_text, option.params(q, k, value), prepare=True).fetchall()
    found = np.full((len(queries), k), -1, dtype=np.int64)
    times = np.empty(len(queries))
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        ids = [r[0] for r in cur.execute(sql_text, option.params(q, k, value), prepare=True).fetchall()]
        times[i] = time.perf_counter() - t0
        found[i, : len(ids)] = ids
    return recall_at_k(found, truth, k), times


def run_pg(
    dsn: str,
    base: np.ndarray,
    queries: np.ndarray,
    truth: np.ndarray,
    metric: Metric,
    k: int = 10,
    options: list[str] | None = None,
    keep: bool = False,
    log=print,
) -> tuple[list[PgPoint], dict]:
    conn = connect(dsn)
    exts = available(conn)
    if "vchord" in exts:
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
        except Exception as e:  # installed but not preloaded, or no permission
            log(f"  vchordrq             skipped: {str(e).splitlines()[0]}")
            log("                       (VectorChord needs shared_preload_libraries = 'vchord')")
            exts.pop("vchord")
    server = conn.execute("SHOW server_version").fetchone()[0]
    _session(conn)

    n, d = base.shape
    table = f"{SCHEMA}.items_{d}"
    load(conn, base, table, log)
    heap = conn.execute("SELECT pg_table_size(%s)", (table,)).fetchone()[0]

    points: list[PgPoint] = []
    for option in OPTIONS:
        if options and option.key not in options:
            continue
        if option.extension not in exts:
            log(f"  {option.key:<20} skipped: extension '{option.extension}' is not available on this server")
            continue
        idx = f"{SCHEMA}.idx_{option.key.replace('-', '_')}"
        conn.execute(f"DROP INDEX IF EXISTS {idx}")
        log(f"  {option.key:<20} building")
        t0 = time.perf_counter()
        conn.execute(option.create(table, d, metric, n).format(idx=idx.split(".")[1]))
        build = time.perf_counter() - t0
        size = conn.execute("SELECT pg_relation_size(%s)", (idx,)).fetchone()[0]
        sql_text = option.query(table, d, metric)
        for value in option.values(n):
            recall, times = _measure(conn, sql_text, option, queries, truth, k, value)
            ms = times * 1000
            point = PgPoint(
                option.key,
                option.label,
                option.knob,
                value,
                _setting(option, value),
                recall,
                float(np.percentile(ms, 50)),
                float(np.percentile(ms, 95)),
                float(1 / times.mean()),
                int(size),
                size / n,
                build,
            )
            points.append(point)
            log(
                f"  {option.key:<20} {point.setting:<22} recall@{k} {recall:.3f}  "
                f"p50 {point.p50_ms:6.2f} ms  index {size / 2**20:7.1f} MiB  built in {build:5.1f}s"
            )
        if not keep:
            conn.execute(f"DROP INDEX {idx}")
    if not keep:
        conn.execute(f"DROP TABLE {table}")
    conn.close()
    env = {"server_version": server, "extensions": exts, "rows": n, "dims": d, "table_bytes": int(heap)}
    return points, env


def as_dicts(points: list[PgPoint]) -> list[dict]:
    return [asdict(p) for p in points]
