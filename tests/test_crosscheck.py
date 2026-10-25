"""Recall against the reference implementations, on the same vectors.

Needs the reference libraries: ``uv run --group crosscheck pytest -m crosscheck``.
"""

import numpy as np
import pytest

from shrinkvec.exact import exact_neighbors, prepare, recall_at_k
from shrinkvec.lab import run_lab

pytestmark = pytest.mark.crosscheck


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    centers = rng.standard_normal((40, 256))
    base = (centers[rng.integers(0, 40, 30_000)] * 0.5 + rng.standard_normal((30_000, 256))).astype(np.float32)
    queries = (centers[rng.integers(0, 40, 300)] * 0.5 + rng.standard_normal((300, 256))).astype(np.float32)
    return base, queries, exact_neighbors(base, queries, 10, "cosine")


def ours(data, pipeline: str) -> float:
    base, queries, truth = data
    results, _ = run_lab(base, queries, "cosine", [pipeline], truth=truth, log=lambda *_: None)
    return results[0].recall


@pytest.mark.parametrize("bits", [1, 4])
def test_rabitq_matches_the_official_library(data, bits):
    rabitqlib = pytest.importorskip("rabitqlib")
    base, queries, truth = data
    b, q = prepare(base, "cosine"), prepare(queries, "cosine")
    index = rabitqlib.IvfIndex(b.shape[1], len(b), 1, bits, "l2")
    index.build(b, b.mean(axis=0, keepdims=True), np.zeros(len(b), dtype=np.uint32))
    ids, _ = index.search(q, 10, 1, high_accuracy=True)
    assert ours(data, f"rabitq-{bits}") == pytest.approx(recall_at_k(np.asarray(ids), truth, 10), abs=0.03)


@pytest.mark.parametrize("bits", [2, 4])
def test_turboquant_matches_turbovec(data, bits):
    turbovec = pytest.importorskip("turbovec")
    base, queries, truth = data
    index = turbovec.TurboQuantIndex(base.shape[1], bits)
    index.add(prepare(base, "cosine"))
    _, ids = index.search(prepare(queries, "cosine"), 10)
    assert ours(data, f"tq-mse-{bits}") == pytest.approx(recall_at_k(np.asarray(ids), truth, 10), abs=0.03)
