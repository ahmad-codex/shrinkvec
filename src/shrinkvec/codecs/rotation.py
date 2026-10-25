"""Random rotations and the TurboQuant scalar codebook."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-distributed random orthogonal matrix.

    QR of a Gaussian matrix, with the signs of R's diagonal folded into Q. Without that
    correction the result is orthogonal but not uniformly distributed (Mezzadri, 2007).
    """
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


@lru_cache(maxsize=64)
def sphere_codebook(d: int, bits: int, grid: int = 400_001) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd-Max centroids and decision boundaries for one coordinate of a random unit vector.

    After a random rotation, each coordinate of a unit vector in d dimensions follows
    f(t) ∝ (1 - t²)^((d-3)/2) on [-1, 1]. TurboQuant quantizes every coordinate with the MSE-optimal
    scalar quantizer for that density, so the codebook depends only on d and the bit width, never on
    the data. Solved numerically on a fine grid that covers ±12 standard deviations.
    """
    if bits < 1:
        raise ValueError("bits must be at least 1")
    sigma = 1.0 / np.sqrt(d)
    half = min(1.0, 12 * sigma)
    t = np.linspace(-half, half, grid)
    log_f = ((d - 3) / 2) * np.log1p(-np.minimum(t * t, 1 - 1e-15)) if d > 3 else np.zeros_like(t)
    w = np.exp(log_f - log_f.max())
    w /= w.sum()

    # Start from the high-resolution optimum, where centroid density follows f^(1/3)
    # (Panter-Dite). Lloyd's iterations then only have to polish it.
    levels = 2**bits
    start = np.cumsum(np.cbrt(w))
    centroids = np.interp((np.arange(levels) + 0.5) / levels, start / start[-1], t)
    for _ in range(10_000):
        bounds = (centroids[1:] + centroids[:-1]) / 2
        cell = np.searchsorted(bounds, t)
        mass = np.bincount(cell, weights=w, minlength=levels)
        new = np.bincount(cell, weights=w * t, minlength=levels) / np.maximum(mass, 1e-300)
        if np.max(np.abs(new - centroids)) < 1e-13 * sigma:
            centroids = new
            break
        centroids = new
    centroids = (centroids - centroids[::-1]) / 2  # the density is symmetric; remove numerical drift
    return centroids.astype(np.float32), ((centroids[1:] + centroids[:-1]) / 2).astype(np.float32)
