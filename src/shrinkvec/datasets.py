"""Public datasets for reproducible results, and loaders for your own vectors."""

from __future__ import annotations

import importlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .exact import Metric


@dataclass(frozen=True)
class Dataset:
    name: str
    metric: Metric
    dims: int
    source: str
    license: str
    description: str


DATASETS = {
    "dbpedia-openai-1536": Dataset(
        "dbpedia-openai-1536",
        "cosine",
        1536,
        "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M",
        "MIT",
        "DBpedia entities embedded with OpenAI text-embedding-ada-002",
    ),
    "glove-100": Dataset(
        "glove-100",
        "cosine",
        100,
        "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "PDDL (GloVe)",
        "GloVe word vectors, 1.18M words (ann-benchmarks)",
    ),
    "sift-128": Dataset(
        "sift-128",
        "l2",
        128,
        "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "Public research dataset (INRIA TEXMEX)",
        "SIFT image descriptors, 1M vectors (ann-benchmarks)",
    ),
}

DBPEDIA_FILES = [
    "data/train-00000-of-00026-3c7b99d1c7eda36e.parquet",
    "data/train-00001-of-00026-2b24035a6390fdcb.parquet",
    "data/train-00002-of-00026-b05ce48965853dad.parquet",
    "data/train-00003-of-00026-d116c3c239aa7895.parquet",
]


def cache_dir() -> Path:
    path = Path(os.environ.get("SHRINKVEC_DATA", Path.home() / ".cache" / "shrinkvec"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, dest: Path, log=print) -> Path:
    if dest.exists():
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    log(f"downloading {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "shrinkvec (+https://github.com/ahmad-codex/shrinkvec)"}
    )
    with urllib.request.urlopen(request) as response, open(part, "wb") as out:
        total = int(response.headers.get("content-length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total and sys.stderr.isatty():
                print(f"\r  {done / total:6.1%} of {total / 1e6:,.0f} MB", end="", file=sys.stderr)
    if total and sys.stderr.isatty():
        print(file=sys.stderr)
    part.replace(dest)
    return dest


def _require(module: str, extra: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise SystemExit(f"this needs {module}: pip install 'shrinkvec[{extra}]'") from e


def load_public(name: str, n: int | None, n_queries: int, seed: int, log=print) -> tuple[np.ndarray, np.ndarray]:
    """Base vectors and held-out queries for a public dataset.

    ann-benchmarks files ship their own query set, so the first ``n_queries`` of it are used. For
    DBpedia the queries are rows held out of the base, chosen at random with ``seed``.
    """
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset '{name}'. Available: {', '.join(DATASETS)}")
    info = DATASETS[name]
    if info.source.endswith(".hdf5"):
        h5py = _require("h5py", "datasets")
        path = _download(info.source, cache_dir() / Path(info.source).name, log)
        with h5py.File(path, "r") as f:
            base = np.asarray(f["train"][: n or None], dtype=np.float32)
            queries = np.asarray(f["test"][:n_queries], dtype=np.float32)
        return base, queries

    rows_needed = (n or 1_000_000) + n_queries
    vectors = _load_dbpedia(rows_needed, log)
    order = np.random.default_rng(seed).permutation(len(vectors))
    queries = vectors[order[:n_queries]]
    base = vectors[np.sort(order[n_queries : n_queries + (n or len(vectors))])]
    return base, queries


def _load_dbpedia(rows: int, log=print) -> np.ndarray:
    pq = _require("pyarrow.parquet", "datasets")
    cached = cache_dir() / "dbpedia-openai-1536.npy"
    if cached.exists():
        vectors = np.load(cached, mmap_mode="r")
        if len(vectors) >= rows:
            return np.asarray(vectors[:rows], dtype=np.float32)
    parts, total = [], 0
    base_url = "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main/"
    for file in DBPEDIA_FILES:
        if total >= rows:
            break
        path = _download(base_url + file, cache_dir() / Path(file).name, log)
        column = pq.read_table(path, columns=["openai"]).column("openai").combine_chunks()
        part = column.values.to_numpy(zero_copy_only=False).reshape(len(column), -1).astype(np.float32)
        parts.append(part)
        total += len(part)
    if total < rows:
        raise SystemExit(f"dbpedia-openai-1536 supports up to {total:,} rows in shrinkvec")
    vectors = np.concatenate(parts)
    np.save(cached, vectors)
    return vectors[:rows]


def read_fvecs(path: Path) -> np.ndarray:
    """The .fvecs/.bvecs/.ivecs format used by SIFT and GIST: each row is an int32 length, then values."""
    raw = np.fromfile(path, dtype=np.uint8)
    dim = int(raw[:4].view(np.int32)[0])
    if path.suffix == ".bvecs":
        return raw.reshape(-1, 4 + dim)[:, 4:].astype(np.float32)
    words = raw.view(np.int32).reshape(-1, 1 + dim)[:, 1:]
    return words.view(np.float32).copy() if path.suffix == ".fvecs" else words.astype(np.float32)


def load_file(path: str, column: str | None = None) -> np.ndarray:
    """Vectors from .npy, .fvecs/.bvecs, or a parquet list column."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p).astype(np.float32)
    if p.suffix in (".fvecs", ".bvecs", ".ivecs"):
        return read_fvecs(p)
    if p.suffix == ".parquet":
        pq = _require("pyarrow.parquet", "datasets")
        table = pq.read_table(p, columns=[column] if column else None)
        name = column or table.column_names[0]
        col = table.column(name).combine_chunks()
        return col.values.to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)
    raise SystemExit(f"unsupported file type '{p.suffix}': use .npy, .fvecs, .bvecs or .parquet")


def split_queries(vectors: np.ndarray, n: int | None, n_queries: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out random rows as queries, so no query is also in the base."""
    if n_queries >= len(vectors):
        raise SystemExit(f"only {len(vectors):,} vectors: ask for fewer than that many queries")
    order = np.random.default_rng(seed).permutation(len(vectors))
    queries = vectors[order[:n_queries]]
    rest = np.sort(order[n_queries:])
    return vectors[rest[:n] if n else rest], queries
