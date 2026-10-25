"""The interface every codec implements, and the codecs that just store a smaller number type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..exact import Metric


class Codec(ABC):
    """Encodes base vectors once, then scores queries against the codes. Higher score = closer.

    ``build`` receives vectors already prepared for the metric (unit length for cosine).
    ``bytes_per_vector`` is what one row costs, including any per-row floats the codec keeps;
    shared state such as a rotation matrix is reported separately in ``shared_bytes``.
    """

    name: str
    family: str
    bits: float

    def __init__(self) -> None:
        self.metric: Metric = "cosine"
        self.d = 0
        self.n = 0

    def build(self, base: np.ndarray, metric: Metric, rng: np.random.Generator) -> None:
        self.metric, (self.n, self.d) = metric, base.shape
        self._build(base, rng)

    @abstractmethod
    def _build(self, base: np.ndarray, rng: np.random.Generator) -> None: ...

    def prepare(self, queries: np.ndarray) -> Any:
        """Per-query work done once (rotations, projections). Returned state goes to the scorers."""
        return queries

    @abstractmethod
    def scores(self, q: Any, start: int, stop: int) -> np.ndarray:
        """Scores of all queries against base rows [start, stop): shape (n_queries, stop - start)."""

    @abstractmethod
    def scores_rows(self, q: Any, qi: int, ids: np.ndarray) -> np.ndarray:
        """Scores of query ``qi`` against the base rows ``ids`` (used for reranking)."""

    @abstractmethod
    def bytes_per_vector(self) -> float: ...

    @property
    def shared_bytes(self) -> int:
        return 0

    @property
    def label(self) -> str:
        return self.name


class ReconstructionCodec(Codec):
    """Codecs that decode to approximate floats and score them like pgvector would."""

    @abstractmethod
    def decode(self, rows: slice | np.ndarray) -> np.ndarray: ...

    def _score(self, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        dots = q @ r.T
        if self.metric == "cosine":
            norms = np.linalg.norm(r, axis=1)
            return dots / np.where(norms > 0, norms, 1)
        return 2 * dots - (r * r).sum(axis=1)

    def scores(self, q: np.ndarray, start: int, stop: int) -> np.ndarray:
        return self._score(q, self.decode(slice(start, stop)))

    def scores_rows(self, q: np.ndarray, qi: int, ids: np.ndarray) -> np.ndarray:
        return self._score(q[qi : qi + 1], self.decode(ids))[0]


class Float32(ReconstructionCodec):
    """The uncompressed baseline: pgvector ``vector``."""

    name, family, bits = "float32", "float", 32
    label = "Full precision (float32)"

    def _build(self, base: np.ndarray, rng: np.random.Generator) -> None:
        self.x = base

    def decode(self, rows: slice | np.ndarray) -> np.ndarray:
        return self.x[rows]

    def bytes_per_vector(self) -> float:
        return 4 * self.d


class Float16(ReconstructionCodec):
    """Half precision: pgvector ``halfvec``."""

    name, family, bits = "halfvec", "float", 16
    label = "Half precision (float16)"

    def _build(self, base: np.ndarray, rng: np.random.Generator) -> None:
        self.x = base.astype(np.float16)

    def decode(self, rows: slice | np.ndarray) -> np.ndarray:
        return self.x[rows].astype(np.float32)

    def bytes_per_vector(self) -> float:
        return 2 * self.d


class Int8(ReconstructionCodec):
    """Scalar quantization to 8 bits per dimension, with a min and max learned per dimension."""

    name, family, bits = "int8", "scalar", 8
    label = "8-bit scalar, per-dimension range"

    def _build(self, base: np.ndarray, rng: np.random.Generator) -> None:
        self.lo = base.min(axis=0)
        span = base.max(axis=0) - self.lo
        self.step = np.where(span > 0, span / 255, 1).astype(np.float32)
        self.codes = np.clip(np.rint((base - self.lo) / self.step), 0, 255).astype(np.uint8)

    def decode(self, rows: slice | np.ndarray) -> np.ndarray:
        return self.lo + self.codes[rows].astype(np.float32) * self.step

    def bytes_per_vector(self) -> float:
        return float(self.d)

    @property
    def shared_bytes(self) -> int:
        return 8 * self.d


class Binary(Codec):
    """One sign bit per dimension compared by Hamming distance: pgvector ``binary_quantize``."""

    name, family, bits = "binary", "binary", 1
    label = "1 sign bit per dimension, Hamming distance"

    def _build(self, base: np.ndarray, rng: np.random.Generator) -> None:
        self.signs = np.where(base > 0, 1, -1).astype(np.int8)

    def prepare(self, queries: np.ndarray) -> np.ndarray:
        return np.where(queries > 0, 1.0, -1.0).astype(np.float32)

    def scores(self, q: np.ndarray, start: int, stop: int) -> np.ndarray:
        return q @ self.signs[start:stop].T.astype(np.float32)  # = d - 2 * hamming

    def scores_rows(self, q: np.ndarray, qi: int, ids: np.ndarray) -> np.ndarray:
        return self.signs[ids].astype(np.float32) @ q[qi]

    def bytes_per_vector(self) -> float:
        return np.ceil(self.d / 8)
