import numpy as np

from shrinkvec.exact import exact_neighbors, recall_at_k, top_k
from shrinkvec.lab import parse_pipeline, run_lab


def clustered(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((20, d))
    return (centers[rng.integers(0, 20, n)] * 0.6 + rng.standard_normal((n, d))).astype(np.float32)


def test_top_k_orders_by_score_then_index():
    scores = np.array([[0.1, 0.9, 0.5, 0.9]])
    assert top_k(scores, 3).tolist() == [[1, 3, 2]]


def test_exact_neighbors_match_brute_force():
    base, queries = clustered(3000, 32, 0), clustered(20, 32, 1)
    for metric in ("cosine", "l2"):
        got = exact_neighbors(base, queries, 10, metric)
        for q, ids in zip(queries, got, strict=True):
            if metric == "l2":
                expected = np.argsort(((base.astype(np.float64) - q) ** 2).sum(axis=1), kind="stable")[:10]
            else:
                b = base / np.linalg.norm(base, axis=1, keepdims=True)
                expected = np.argsort(-(b.astype(np.float64) @ (q / np.linalg.norm(q))), kind="stable")[:10]
            assert ids.tolist() == expected.tolist()


def test_recall_counts_overlap():
    assert recall_at_k(np.array([[1, 2, 3]]), np.array([[3, 4, 1]]), 3) == 2 / 3


def test_pipeline_syntax():
    assert parse_pipeline("rabitq-1", 100) == ("rabitq-1", None, 0)
    assert parse_pipeline("binary>float32", 100) == ("binary", "float32", 100)
    assert parse_pipeline("tq-mse-3>tq-mse-8:300", 100) == ("tq-mse-3", "tq-mse-8", 300)


def test_lab_is_exact_for_float32_and_deterministic():
    base, queries = clustered(4000, 64, 2), clustered(50, 64, 3)
    pipelines = ["float32", "tq-mse-3", "rabitq-1>float32"]
    first, _ = run_lab(base, queries, "cosine", pipelines, log=lambda *_: None)
    second, _ = run_lab(base, queries, "cosine", pipelines, log=lambda *_: None)
    assert first[0].recall == 1.0
    assert [r.recall for r in first] == [r.recall for r in second]
    reranked = first[2]
    assert reranked.recall <= reranked.candidate_recall
    assert reranked.total_bytes == 64 / 8 + 8 + 4 * 64


def test_repeats_report_spread_only_for_randomized_codecs():
    base, queries = clustered(3000, 64, 4), clustered(60, 64, 5)
    results, _ = run_lab(base, queries, "cosine", ["halfvec", "rabitq-1"], repeats=3, log=lambda *_: None)
    halfvec, rabitq = results
    assert (halfvec.repeats, halfvec.recall_std) == (1, 0.0)
    assert rabitq.repeats == 3 and rabitq.recall_std > 0
