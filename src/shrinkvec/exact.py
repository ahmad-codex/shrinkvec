"""Exact nearest neighbours and recall: the ground truth every result is measured against."""

from __future__ import annotations

from typing import Literal

import numpy as np

Metric = Literal["cosine", "l2"]

CHUNK_ROWS = 65_536


def normalize(x: np.ndarray) -> np.ndarray:
    """Unit-length rows. Zero rows stay zero instead of becoming NaN."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def prepare(x: np.ndarray, metric: Metric) -> np.ndarray:
    """What every codec sees: float32 rows, normalised for cosine.

    Cosine search on unit vectors ranks exactly like inner product and like L2, which is what
    pgvector does with ``vector_cosine_ops``.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    return normalize(x) if metric == "cosine" else x


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first. Ties go to the lower index."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.lexsort((part, -np.take_along_axis(scores, part, axis=1)), axis=1)
    return np.take_along_axis(part, order, axis=1)


def merge_top_k(
    best_ids: np.ndarray | None, best_scores: np.ndarray | None, ids: np.ndarray, scores: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the k best of two candidate sets (per row)."""
    if best_ids is None or best_scores is None:
        all_ids, all_scores = ids, scores
    else:
        all_ids = np.concatenate([best_ids, ids], axis=1)
        all_scores = np.concatenate([best_scores, scores], axis=1)
    pick = top_k(all_scores, k)
    return np.take_along_axis(all_ids, pick, axis=1), np.take_along_axis(all_scores, pick, axis=1)


def exact_neighbors(base: np.ndarray, queries: np.ndarray, k: int, metric: Metric) -> np.ndarray:
    """True top-k ids per query, computed in float64 so float32 rounding can't reorder near ties."""
    b = prepare(base, metric).astype(np.float64)
    q = prepare(queries, metric).astype(np.float64)
    q_sq = (q * q).sum(axis=1, keepdims=True) if metric == "l2" else None
    best_ids = best_scores = None
    for start in range(0, len(b), CHUNK_ROWS):
        chunk = b[start : start + CHUNK_ROWS]
        scores = q @ chunk.T
        if metric == "l2":
            scores = -(q_sq - 2 * scores + (chunk * chunk).sum(axis=1)[None, :])
        local = top_k(scores, k)
        best_ids, best_scores = merge_top_k(
            best_ids, best_scores, local + start, np.take_along_axis(scores, local, axis=1), k
        )
    assert best_ids is not None
    return best_ids


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Mean fraction of the true top-k that appear in the found top-k."""
    hits = [len(np.intersect1d(f[:k], t[:k], assume_unique=True)) for f, t in zip(found, truth, strict=True)]
    return float(np.mean(hits)) / k
