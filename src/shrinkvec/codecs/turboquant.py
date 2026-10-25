"""TurboQuant (Zandieh, Daliri, Hadian, Mirrokni, 2025; arXiv 2504.19874).

Rotate, then quantize every coordinate with a codebook that depends only on the dimension. Two
variants from the paper:

* ``mse`` spends all b bits on the scalar codes (TurboQuant_mse). Lowest reconstruction error, but
  inner products come out slightly biased. Scores use the decoded vector rescaled to unit length
  (one float per row, as turbovec does): the decoded length varies by about 1% between rows, and
  that noise costs up to 11 points of recall@10 at 2 bits if left in.
* ``prod`` spends b-1 bits on the codes and 1 bit on a QJL sign sketch of the residual
  (TurboQuant_prod). The inner-product estimate is unbiased. Same b bits per dimension in total,
  plus one float per row for the residual norm.

No training on the data: a vector can be encoded the moment it arrives.
"""

from __future__ import annotations

import numpy as np

from .base import Codec
from .rotation import random_orthogonal, sphere_codebook


class TurboQuant(Codec):
    family = "TurboQuant"

    def __init__(self, bits: int, variant: str = "mse") -> None:
        super().__init__()
        if variant not in ("mse", "prod"):
            raise ValueError("variant must be 'mse' or 'prod'")
        if bits < 1 or (variant == "prod" and bits < 2):
            raise ValueError("TurboQuant needs at least 1 bit (mse) or 2 bits (prod)")
        self.bits, self.variant = bits, variant
        self.code_bits = bits if variant == "mse" else bits - 1
        self.name = f"tq-{variant}-{bits}"

    @property
    def label(self) -> str:
        return f"TurboQuant {self.bits}-bit ({self.variant})"

    def _build(self, base: np.ndarray, rng: np.random.Generator) -> None:
        d = self.d
        self.rot = random_orthogonal(d, rng)
        norms = np.linalg.norm(base, axis=1).astype(np.float32)
        unit = np.divide(base, norms[:, None], out=np.zeros_like(base), where=norms[:, None] > 0)
        self.norms = norms if self.metric == "l2" else None

        self.centroids, bounds = sphere_codebook(d, self.code_bits)
        y = unit @ self.rot.T
        self.codes = np.searchsorted(bounds, y).astype(np.uint8)
        if self.variant == "mse":
            decoded_norm = np.linalg.norm(self.centroids[self.codes], axis=1)  # rotation keeps the length
            self.inv_norm = np.divide(1, decoded_norm, out=np.zeros_like(decoded_norm), where=decoded_norm > 0)

        if self.variant == "prod":
            self.sketch = rng.standard_normal((d, d)).astype(np.float32)
            residual = unit - self.centroids[self.codes] @ self.rot
            self.gamma = np.linalg.norm(residual, axis=1).astype(np.float32)
            self.signs = np.where(residual @ self.sketch.T >= 0, 1, -1).astype(np.int8)

    def prepare(self, queries: np.ndarray) -> dict[str, np.ndarray]:
        state = {"rotated": queries @ self.rot.T}
        if self.variant == "prod":
            state["sketched"] = queries @ self.sketch.T
        return state

    def _inner(self, q: dict[str, np.ndarray], rows: slice | np.ndarray, qi: slice | int) -> np.ndarray:
        """Estimated <query, unit row>."""
        rotated = np.atleast_2d(q["rotated"][qi])
        ip = rotated @ self.centroids[self.codes[rows]].T
        if self.variant == "mse":
            ip *= self.inv_norm[rows]
        else:
            sketched = np.atleast_2d(q["sketched"][qi])
            qjl = sketched @ self.signs[rows].T.astype(np.float32)
            ip += np.sqrt(np.pi / 2) / self.d * qjl * self.gamma[rows]
        return ip

    def _to_score(self, ip: np.ndarray, rows: slice | np.ndarray) -> np.ndarray:
        if self.norms is None:
            return ip
        n = self.norms[rows]
        return 2 * ip * n - n * n

    def scores(self, q: dict[str, np.ndarray], start: int, stop: int) -> np.ndarray:
        rows = slice(start, stop)
        return self._to_score(self._inner(q, rows, slice(None)), rows)

    def scores_rows(self, q: dict[str, np.ndarray], qi: int, ids: np.ndarray) -> np.ndarray:
        return self._to_score(self._inner(q, ids, qi), ids)[0]

    def reconstruct(self, rows: slice | np.ndarray) -> np.ndarray:
        """Decoded unit vectors (used by tests to check the paper's distortion figures)."""
        out = self.centroids[self.codes[rows]] @ self.rot
        if self.variant == "prod":
            out += np.sqrt(np.pi / 2) / self.d * self.gamma[rows, None] * (self.signs[rows] @ self.sketch)
        return out

    def bytes_per_vector(self) -> float:
        size = np.ceil(self.code_bits * self.d / 8) + 4  # codes + decoded norm (mse) or residual norm (prod)
        if self.variant == "prod":
            size += np.ceil(self.d / 8)  # QJL sign bits
        if self.metric == "l2":
            size += 4  # vector norm
        return float(size)

    @property
    def shared_bytes(self) -> int:
        return 4 * self.d * self.d * (2 if self.variant == "prod" else 1)
