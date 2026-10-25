"""The codec lab: exact recall of every compression option on your vectors, no database needed.

Each option is a pipeline: search all codes with one codec, optionally rerank the best
``candidates`` with a second, more precise one (``rabitq-1>float32``). Every scan is exhaustive,
so the recall measures the codes themselves, not an index.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import asdict, dataclass

import numpy as np

from .codecs import Codec, make_codec
from .exact import Metric, exact_neighbors, merge_top_k, prepare, recall_at_k, top_k

DEFAULT_PIPELINES = [
    "float32",
    "halfvec",
    "int8",
    "tq-mse-4",
    "tq-prod-4",
    "tq-mse-3",
    "tq-mse-2",
    "rabitq-4",
    "rabitq-1",
    "binary",
    "tq-mse-3>tq-mse-8",
    "rabitq-1>float32",
    "binary>float32",
]


DETERMINISTIC = {"float32", "halfvec", "int8", "binary"}


@dataclass
class LabResult:
    pipeline: str
    label: str
    recall: float
    recall_std: float
    repeats: int
    candidate_recall: float | None
    search_bytes: float
    total_bytes: float
    compression: float
    encode_seconds: float


def parse_pipeline(spec: str, default_candidates: int) -> tuple[str, str | None, int]:
    """``search[>rerank[:candidates]]`` -> (search codec, rerank codec or None, candidates)."""
    if ">" not in spec:
        return spec, None, 0
    search, rerank = spec.split(">", 1)
    if ":" in rerank:
        rerank, n = rerank.split(":", 1)
        return search, rerank, int(n)
    return search, rerank, default_candidates


def chunk_rows(d: int) -> int:
    return max(2048, int(1.2e8 // (4 * d)))


def scan(codec: Codec, state, n_queries: int, k: int) -> np.ndarray:
    """Top-k ids per query over every code, best first."""
    best_ids = best_scores = None
    step = chunk_rows(codec.d)
    for start in range(0, codec.n, step):
        stop = min(start + step, codec.n)
        scores = codec.scores(state, start, stop)
        local = top_k(scores, k)
        best_ids, best_scores = merge_top_k(
            best_ids, best_scores, local + start, np.take_along_axis(scores, local, axis=1), k
        )
    assert best_ids is not None and len(best_ids) == n_queries
    return best_ids


def rerank(codec: Codec, state, candidates: np.ndarray, k: int) -> np.ndarray:
    out = np.empty((len(candidates), k), dtype=np.int64)
    for qi, ids in enumerate(candidates):
        scores = codec.scores_rows(state, qi, ids)
        out[qi] = ids[top_k(scores[None, :], k)[0]]
    return out


def codec_seed(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def run_lab(
    base: np.ndarray,
    queries: np.ndarray,
    metric: Metric = "cosine",
    pipelines: list[str] | None = None,
    k: int = 10,
    candidates: int = 100,
    seed: int = 42,
    truth: np.ndarray | None = None,
    repeats: int = 1,
    log=print,
) -> tuple[list[LabResult], np.ndarray]:
    """Recall of each pipeline. With ``repeats`` > 1, codecs that use a random rotation are rebuilt
    with seeds ``seed``, ``seed + 1``, ... and the recall is reported as mean and standard deviation."""
    pipelines = pipelines or DEFAULT_PIPELINES
    b, q = prepare(base, metric), prepare(queries, metric)
    if truth is None:
        log(f"exact top-{k} for {len(q):,} queries over {len(b):,} vectors (float64)")
        truth = exact_neighbors(base, queries, k, metric)

    fixed: dict[str, tuple[Codec, object, float]] = {}

    def randomized(spec: str) -> bool:
        search, rerank_name, _ = parse_pipeline(spec, candidates)
        return search not in DETERMINISTIC or (rerank_name is not None and rerank_name not in DETERMINISTIC)

    float32_bytes = 4 * b.shape[1]
    runs: dict[str, list[tuple[float, float | None]]] = {spec: [] for spec in pipelines}
    info: dict[str, tuple[str, float, float, float]] = {}
    seeded: dict[str, tuple[Codec, object, float]] = {}  # codecs with a random rotation, rebuilt per repeat

    def get(name: str, rep_seed: int) -> tuple[Codec, object, float]:
        cache = fixed if name in DETERMINISTIC else seeded
        if name not in cache:
            codec = make_codec(name)
            t0 = time.perf_counter()
            codec.build(b, metric, codec_seed(rep_seed, name))
            cache[name] = (codec, codec.prepare(q), time.perf_counter() - t0)
        return cache[name]

    for rep in range(repeats):
        seeded.clear()
        for spec in pipelines:
            if rep > 0 and not randomized(spec):
                continue
            search_name, rerank_name, n_cand = parse_pipeline(spec, candidates)
            search_codec, search_state, t_search = get(search_name, seed + rep)
            if rerank_name is None:
                found = scan(search_codec, search_state, len(q), k)
                cand_recall, total, t_total = None, search_codec.bytes_per_vector(), t_search
                label = search_codec.label
            else:
                rerank_codec, rerank_state, t_rerank = get(rerank_name, seed + rep)
                cands = scan(search_codec, search_state, len(q), n_cand)
                # How many of the true top-k made it into the candidates: the ceiling for the rerank.
                cand_recall = float(
                    np.mean([len(np.intersect1d(c, t[:k])) for c, t in zip(cands, truth, strict=True)]) / k
                )
                found = rerank(rerank_codec, rerank_state, cands, k)
                total = search_codec.bytes_per_vector() + rerank_codec.bytes_per_vector()
                t_total = t_search + t_rerank
                label = f"{search_codec.label}, rerank top {n_cand} with {rerank_codec.label}"
            recall = recall_at_k(found, truth, k)
            runs[spec].append((recall, cand_recall))
            info[spec] = (label, search_codec.bytes_per_vector(), total, t_total)
            tag = f" [seed {seed + rep}]" if repeats > 1 and randomized(spec) else ""
            log(
                f"  {spec:<22} recall@{k} {recall:.3f}   {info[spec][1]:>7.0f} B/vector "
                f"({float32_bytes / info[spec][1]:.1f}x){tag}"
            )

    results = []
    for spec in pipelines:
        recalls = np.array([r for r, _ in runs[spec]])
        cands = [c for _, c in runs[spec] if c is not None]
        label, search_bytes, total, seconds = info[spec]
        results.append(
            LabResult(
                pipeline=spec,
                label=label,
                recall=float(recalls.mean()),
                recall_std=float(recalls.std(ddof=1)) if len(recalls) > 1 else 0.0,
                repeats=len(recalls),
                candidate_recall=float(np.mean(cands)) if cands else None,
                search_bytes=search_bytes,
                total_bytes=total,
                compression=float32_bytes / search_bytes,
                encode_seconds=seconds,
            )
        )
    return results, truth


def as_dicts(results: list[LabResult]) -> list[dict]:
    return [asdict(r) for r in results]
