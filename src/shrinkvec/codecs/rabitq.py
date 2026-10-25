"""RaBitQ (Gao and Long, SIGMOD 2024) and extended RaBitQ (Gao et al., SIGMOD 2025).

Each vector is centred on a centroid, normalised and rotated. Its code is the point of a scaled
grid with the smallest angle to it: one sign bit per dimension for the original RaBitQ, B bits per
dimension for the extended version. The inner product with a query is estimated as
<x̄, q'> / <x̄, o'>, which is unbiased, and the distance follows from two stored floats per row:
the distance to the centroid and <x̄, o'>.

This lab version scans every code with one centroid (the data mean) and a full-precision query.
VectorChord uses many centroids (IVF) and a 4-bit query; the paper reports both effects as small
compared with the code itself.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .base import Codec
from .rotation import random_orthogonal

EXACT_MAX_BITS = 5
BATCH_EVENTS = 1_000_000  # per thread; bounds memory at a few hundred MB in total
PAD = 64  # RaBitQ implementations pack codes in 64-dimension blocks, padding with zeros


def _pad(x: np.ndarray, width: int) -> np.ndarray:
    return x if x.shape[1] == width else np.hstack([x, np.zeros((len(x), width - x.shape[1]), x.dtype)])


def _best_magnitudes_exact(a: np.ndarray, levels: int) -> np.ndarray:
    """Grid magnitudes in {0.5, 1.5, ..., levels - 0.5} that maximise the cosine with ``a`` >= 0.

    For a scale t, coordinate i rounds to floor(t * a_i) + 0.5 (capped at the top level). The
    cosine only changes where t * a_i crosses an integer, so the optimum is at one of those
    critical points (extended RaBitQ, Algorithm 1). Every crossing raises the dot product by a_i
    and the squared norm by 2k, so all candidates come from two cumulative sums.
    """
    n, d = a.shape
    if levels == 1:
        return np.full_like(a, 0.5)
    a64 = a.astype(np.float64)
    k = np.arange(1, levels, dtype=np.float64)
    with np.errstate(divide="ignore"):
        events = (k[None, None, :] / a64[:, :, None]).reshape(n, -1)
    order = np.argsort(events, axis=1, kind="stable")
    t_sorted = np.take_along_axis(events, order, axis=1)
    # Event j belongs to coordinate j // (levels - 1) crossing level j % (levels - 1) + 1.
    d_dot = np.take_along_axis(a64, order // (levels - 1), axis=1)
    d_sq = 2.0 * (order % (levels - 1) + 1)

    dot0 = 0.5 * a.sum(axis=1, dtype=np.float64)
    sq0 = 0.25 * d
    dot = np.concatenate([dot0[:, None], dot0[:, None] + np.cumsum(d_dot, axis=1)], axis=1)
    sq = np.concatenate([np.full((n, 1), sq0), sq0 + np.cumsum(d_sq, axis=1)], axis=1)
    cosine = np.where(np.isfinite(np.concatenate([np.zeros((n, 1)), t_sorted], axis=1)), dot / np.sqrt(sq), -1)
    best = np.argmax(cosine, axis=1)

    # Pick a scale strictly inside the winning interval, so floor() can't land on the boundary.
    padded = np.concatenate([np.zeros((n, 1)), t_sorted, np.full((n, 1), np.inf)], axis=1)
    lo = np.take_along_axis(padded, best[:, None], axis=1)
    hi = np.take_along_axis(padded, best[:, None] + 1, axis=1)
    t = np.where(np.isfinite(hi), (lo + hi) / 2, lo * 2 + 1)
    return np.minimum(np.floor(t * a), levels - 1) + 0.5


def _best_magnitudes_grid(a: np.ndarray, levels: int, candidates: int = 256) -> np.ndarray:
    """Near-optimal magnitudes for high bit widths: try a geometric grid of scales, keep the best."""
    a64 = a.astype(np.float64)
    top = np.maximum(a64.max(axis=1, keepdims=True), 1e-12)
    scales = (levels / top) * np.geomspace(0.05, 1.5, candidates)[None, :]
    best_cos = np.full(len(a), -np.inf)
    best = np.full_like(a64, 0.5)
    for j in range(candidates):
        m = np.minimum(np.floor(scales[:, j : j + 1] * a64), levels - 1) + 0.5
        cos = (m * a64).sum(axis=1) / np.sqrt((m * m).sum(axis=1))
        better = cos > best_cos
        best_cos = np.where(better, cos, best_cos)
        best[better] = m[better]
    return best


class RaBitQ(Codec):
    family = "RaBitQ"

    def __init__(self, bits: int = 1) -> None:
        super().__init__()
        if not 1 <= bits <= 8:
            raise ValueError("RaBitQ bits must be between 1 and 8")
        self.bits = bits
        self.name = f"rabitq-{bits}"

    @property
    def label(self) -> str:
        return "RaBitQ 1-bit" if self.bits == 1 else f"RaBitQ {self.bits}-bit (extended)"

    def _build(self, base: np.ndarray, rng: np.random.Generator) -> None:
        self.padded = -(-self.d // PAD) * PAD
        self.centroid = base.mean(axis=0, dtype=np.float64).astype(np.float32)
        diff = base - self.centroid
        self.dist = np.linalg.norm(diff, axis=1).astype(np.float32)
        unit = np.divide(diff, self.dist[:, None], out=np.zeros_like(diff), where=self.dist[:, None] > 0)
        self.rot = random_orthogonal(self.padded, rng)
        rotated = _pad(unit, self.padded) @ self.rot  # P^T o for each row o

        levels = 2 ** (self.bits - 1)
        a = np.abs(rotated)
        magnitudes = np.empty_like(a)
        per_batch = max(1, BATCH_EVENTS // max(1, self.padded * (levels - 1)))
        search = _best_magnitudes_exact if self.bits <= EXACT_MAX_BITS else _best_magnitudes_grid

        def fill(s: int) -> None:
            magnitudes[s : s + per_batch] = search(a[s : s + per_batch], levels)

        # NumPy's sort releases the GIL, so batches run on every core; each batch writes its own rows.
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            list(pool.map(fill, range(0, self.n, per_batch)))
        # Stored as 2 * x̄: odd integers in [-(2^B - 1), 2^B - 1], i.e. B bits per dimension.
        self.code = (np.where(rotated >= 0, 1, -1) * (2 * magnitudes)).astype(np.int16)
        factor = (self.code * rotated).sum(axis=1, dtype=np.float64)  # <2x̄, o'>
        self.factor = np.where(factor > 0, factor, 1).astype(np.float32)

    def prepare(self, queries: np.ndarray) -> dict[str, np.ndarray]:
        diff = queries - self.centroid
        dist = np.linalg.norm(diff, axis=1).astype(np.float32)
        unit = np.divide(diff, dist[:, None], out=np.zeros_like(diff), where=dist[:, None] > 0)
        return {"rotated": _pad(unit, self.padded) @ self.rot, "dist": dist}

    def estimate_cosine(self, q: dict[str, np.ndarray], rows: slice | np.ndarray, qi: slice | int) -> np.ndarray:
        """Unbiased estimate of <o, q> between the normalised, centred row and query."""
        rotated = np.atleast_2d(q["rotated"][qi])
        return (rotated @ self.code[rows].T.astype(np.float32)) / self.factor[rows]

    def _to_score(self, est: np.ndarray, q: dict[str, np.ndarray], rows: slice | np.ndarray, qi) -> np.ndarray:
        r = self.dist[rows]
        qd = np.atleast_1d(q["dist"][qi])[:, None]
        return 2 * r * qd * est - r * r  # = |q - c|^2 - |o - q|^2, the query term is constant

    def scores(self, q: dict[str, np.ndarray], start: int, stop: int) -> np.ndarray:
        rows = slice(start, stop)
        return self._to_score(self.estimate_cosine(q, rows, slice(None)), q, rows, slice(None))

    def scores_rows(self, q: dict[str, np.ndarray], qi: int, ids: np.ndarray) -> np.ndarray:
        return self._to_score(self.estimate_cosine(q, ids, qi), q, ids, qi)[0]

    def bytes_per_vector(self) -> float:
        return float(self.bits * self.padded / 8 + 8)  # codes + two float32 factors

    @property
    def shared_bytes(self) -> int:
        return 4 * self.padded * self.padded + 4 * self.d
