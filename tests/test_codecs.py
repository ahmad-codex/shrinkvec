"""The codecs against the numbers their papers publish."""

import itertools

import numpy as np
import pytest

from shrinkvec.codecs import make_codec
from shrinkvec.codecs.rabitq import RaBitQ, _best_magnitudes_exact, _best_magnitudes_grid
from shrinkvec.codecs.rotation import random_orthogonal, sphere_codebook
from shrinkvec.codecs.turboquant import TurboQuant

D = 1536


def unit_rows(n: int, d: int, seed: int) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def pairs():
    return unit_rows(2000, D, 1), unit_rows(2000, D, 2)


def test_rotation_is_orthogonal():
    p = random_orthogonal(256, np.random.default_rng(0)).astype(np.float64)
    assert np.allclose(p @ p.T, np.eye(256), atol=1e-5)


@pytest.mark.parametrize(
    ("bits", "expected"),
    [(1, [0.7979]), (2, [0.4528, 1.510]), (3, [0.2451, 0.7560, 1.344, 2.152])],
)
def test_codebook_matches_gaussian_lloyd_max(bits, expected):
    # At high dimension the coordinate density is N(0, 1/d); these are the classic Lloyd-Max levels.
    centroids, bounds = sphere_codebook(D, bits)
    positive = centroids[len(centroids) // 2 :] * np.sqrt(D)
    assert np.allclose(positive, expected, rtol=2e-3)
    assert np.allclose(centroids, -centroids[::-1])
    assert np.all(np.diff(bounds) > 0)


@pytest.mark.parametrize(("bits", "paper"), [(1, 0.363), (2, 0.1175), (3, 0.0345), (4, 0.0095)])
def test_turboquant_mse_distortion_matches_paper(pairs, bits, paper):
    x, _ = pairs
    tq = TurboQuant(bits, "mse")
    tq.build(x, "cosine", np.random.default_rng(7))
    distortion = ((tq.reconstruct(slice(None)) - x) ** 2).sum(axis=1).mean()
    assert distortion == pytest.approx(paper, rel=0.03)


def test_turboquant_eight_bits_is_within_the_paper_bound(pairs):
    x, _ = pairs
    tq = TurboQuant(8, "mse")
    tq.build(x, "cosine", np.random.default_rng(7))
    distortion = ((tq.reconstruct(slice(None)) - x) ** 2).sum(axis=1).mean()
    assert distortion <= np.sqrt(3) * np.pi / 2 * 4.0**-8


def _estimates(codec, x, y):
    state = codec.prepare(y)
    return np.array([codec.scores_rows(state, i, np.array([i]))[0] for i in range(len(x))])


def test_turboquant_prod_is_unbiased_and_mse_shrinks(pairs):
    x, y = pairs
    true = (x * y).sum(axis=1)

    prod = TurboQuant(2, "prod")
    prod.build(x, "cosine", np.random.default_rng(7))
    est = _estimates(prod, x, y)
    assert np.polyfit(true, est, 1)[0] == pytest.approx(1.0, abs=0.03)
    assert abs((est - true).mean()) < 4 * (est - true).std() / np.sqrt(len(x))

    mse = TurboQuant(2, "mse")
    mse.build(x, "cosine", np.random.default_rng(7))
    raw = (mse.reconstruct(slice(None)) * y).sum(axis=1)  # the paper's estimate, before rescaling
    assert np.polyfit(true, raw, 1)[0] < 0.92  # biased towards zero, as the paper says


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_rabitq_estimate_is_unbiased_with_the_paper_variance(pairs, bits):
    x, y = pairs
    rq = RaBitQ(bits)
    rq.build(x, "cosine", np.random.default_rng(7))
    c = rq.centroid
    o = (x - c) / np.linalg.norm(x - c, axis=1, keepdims=True)
    q = (y - c) / np.linalg.norm(y - c, axis=1, keepdims=True)
    true = (o * q).sum(axis=1)
    state = rq.prepare(y)
    est = np.array([rq.estimate_cosine(state, np.array([i]), i)[0, 0] for i in range(len(x))])
    err = est - true

    assert abs(err.mean()) < 4 * err.std() / np.sqrt(len(err))
    # Error bound of RaBitQ (Theorem 3.2) with eps0 = 1.9. The error is close to Gaussian with
    # exactly that scale, so a 1.9 sigma bound is exceeded about 5.7% of the time.
    xbar = rq.code.astype(np.float64) / 2
    rotated = (o @ rq.rot).astype(np.float64)
    o_bar_o = (xbar * rotated).sum(axis=1) / np.linalg.norm(xbar, axis=1)
    bound = np.sqrt((1 - o_bar_o**2) / o_bar_o**2) * 1.9 / np.sqrt(D - 1)
    assert 0.03 < (np.abs(err) > bound).mean() < 0.09
    if bits == 1:
        assert o_bar_o.mean() == pytest.approx(np.sqrt(2 / np.pi), abs=2e-3)


@pytest.mark.parametrize("levels", [2, 4, 8])
def test_extended_rabitq_finds_the_best_code(levels):
    rng = np.random.default_rng(3)
    a = np.abs(rng.standard_normal((40, 4)))
    got = _best_magnitudes_exact(a, levels)
    grid = np.arange(levels) + 0.5
    for row, magnitudes in zip(a, got, strict=True):
        best = max(np.dot(m, row) / np.linalg.norm(m) for m in itertools.product(grid, repeat=4))
        assert np.dot(magnitudes, row) / np.linalg.norm(magnitudes) == pytest.approx(best, rel=1e-12)


def test_grid_search_is_close_to_the_exact_code():
    a = np.abs(np.random.default_rng(4).standard_normal((200, 64)))

    def cosine(m):
        return (m * a).sum(axis=1) / np.linalg.norm(m, axis=1) / np.linalg.norm(a, axis=1)

    assert np.all(cosine(_best_magnitudes_grid(a, 8)) >= cosine(_best_magnitudes_exact(a, 8)) - 1e-3)


@pytest.mark.parametrize(
    ("name", "bytes_1536"),
    [
        ("float32", 6144),
        ("halfvec", 3072),
        ("int8", 1536),
        ("binary", 192),
        ("tq-mse-4", 768 + 4),
        ("tq-prod-4", 576 + 192 + 4),
        ("rabitq-1", 192 + 8),
        ("rabitq-4", 768 + 8),
    ],
)
def test_bytes_per_vector(name, bytes_1536):
    codec = make_codec(name)
    codec.build(unit_rows(8, D, 0), "cosine", np.random.default_rng(0))
    assert codec.bytes_per_vector() == bytes_1536


def test_rabitq_pads_to_64_dimension_blocks_like_the_reference_library():
    codec = make_codec("rabitq-1")
    codec.build(unit_rows(8, 100, 0), "cosine", np.random.default_rng(0))
    assert codec.bytes_per_vector() == 128 / 8 + 8


def test_l2_adds_the_norm_for_turboquant():
    codec = make_codec("tq-mse-4")
    codec.build(unit_rows(8, D, 0) * 3, "l2", np.random.default_rng(0))
    assert codec.bytes_per_vector() == 768 + 4 + 4


def test_unknown_codec_names_are_rejected():
    with pytest.raises(ValueError, match="unknown codec"):
        make_codec("pq-8")
