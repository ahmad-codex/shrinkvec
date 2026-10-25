"""Pick the smallest option that meets a recall target, and say how to adopt it."""

from __future__ import annotations

from dataclasses import dataclass

from .lab import LabResult
from .pg import OPTIONS, PgPoint

# What each lab codec corresponds to in Postgres today.
WHERE = {
    "float32": "pgvector vector",
    "halfvec": "pgvector halfvec",
    "binary": "pgvector binary_quantize",
    "int8": "not in pgvector",
    "rabitq": "VectorChord vchordrq",
    "tq": "pg_turbovec",
}


def where(pipeline: str) -> str:
    head = pipeline.split(">")[0]
    return WHERE.get(head, WHERE.get(head.split("-")[0], ""))


@dataclass
class Recommendation:
    target: float
    choice: dict | None
    sql: str
    reason: str


def from_lab(results: list[LabResult], target: float) -> Recommendation:
    ok = [r for r in results if r.recall >= target]
    if not ok:
        best = max(results, key=lambda r: r.recall)
        return Recommendation(
            target,
            None,
            "",
            f"No option reached recall@10 {target:.2f}; the best was {best.pipeline} at {best.recall:.3f}. "
            "Keep float32, or rerank more candidates.",
        )
    pick = min(ok, key=lambda r: (r.search_bytes, r.total_bytes, -r.recall))
    stored = (
        f" The rerank reads more precise data for each candidate, so {pick.total_bytes:,.0f} bytes per"
        " vector stay stored; only the searched part has to be in memory."
        if pick.total_bytes > pick.search_bytes
        else ""
    )
    reason = (
        f"{pick.pipeline} keeps recall@10 at {pick.recall:.3f} with {pick.search_bytes:,.0f} bytes per vector "
        f"to search ({pick.compression:.1f}x smaller than float32).{stored} In Postgres: {where(pick.pipeline)}."
    )
    return Recommendation(target, {"pipeline": pick.pipeline, "recall": pick.recall}, "", reason)


def adoption_sql(point: PgPoint, table: str, dims: int, metric, rows: int, k: int = 10) -> str:
    """The same index, setting and query the benchmark measured, for your table."""
    option = next(o for o in OPTIONS if o.key == point.option)
    create = option.create(table, dims, metric, rows).replace("{idx} ", "")
    query = option.query(table, dims, metric)
    placeholders = ["$1"] * query.count("%s")
    params = list(option.params("$1", k, point.value))
    for i, value in enumerate(params):
        if value != "$1":
            placeholders[i] = str(value)
    lines = []
    if option.extension == "vchord":
        lines.append("CREATE EXTENSION IF NOT EXISTS vchord CASCADE;")
    lines += [
        create + ";",
        *[s + ";" for s in option.settings(point.value)],
        (query % tuple(placeholders)) + ";",
    ]
    return "\n".join(lines)


def from_pg(points: list[PgPoint], target: float, table: str, dims: int, metric, rows: int) -> Recommendation:
    ok = [p for p in points if p.recall >= target]
    if not ok:
        best = max(points, key=lambda p: p.recall)
        return Recommendation(
            target,
            None,
            "",
            f"No index reached recall@10 {target:.2f} in this sweep; the best was {best.option} "
            f"({best.setting}) at {best.recall:.3f}.",
        )
    pick = min(ok, key=lambda p: (p.index_bytes, p.p50_ms))
    reason = (
        f"{pick.label}, {pick.setting}: recall@10 {pick.recall:.3f}, "
        f"p50 {pick.p50_ms:.2f} ms, index {pick.index_bytes / 2**20:,.1f} MiB "
        f"({pick.bytes_per_vector:,.0f} bytes per vector). The smallest index that meets {target:.2f}."
    )
    return Recommendation(
        target,
        {"option": pick.option, "knob": pick.knob, "value": pick.value},
        adoption_sql(pick, table, dims, metric, rows),
        reason,
    )
