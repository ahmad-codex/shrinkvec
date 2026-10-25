"""Where the vectors come from: a public dataset, a file, or a column of your own table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import DATASETS, load_file, load_public, split_queries
from .exact import Metric


@dataclass
class Source:
    name: str
    base: np.ndarray
    queries: np.ndarray
    metric: Metric
    note: str


def from_args(args, log=print) -> Source:
    if args.dataset:
        info = DATASETS.get(args.dataset)
        if info is None:
            raise SystemExit(f"unknown dataset '{args.dataset}'. Available: {', '.join(DATASETS)}")
        base, queries = load_public(args.dataset, args.n, args.queries, args.seed, log)
        return Source(info.name, base, queries, args.metric or info.metric, f"{info.description}. {info.license}.")
    if args.file:
        vectors = load_file(args.file, args.column)
        base, queries = split_queries(vectors, args.n, args.queries, args.seed)
        return Source(args.file, base, queries, args.metric or "cosine", "Vectors from a file; queries held out.")
    if args.table:
        from .pg import sample_table

        vectors = sample_table(args.dsn, args.table, args.column, (args.n or 100_000) + args.queries, args.seed)
        base, queries = split_queries(vectors, args.n, args.queries, args.seed)
        return Source(
            f"{args.table}.{args.column}",
            base,
            queries,
            args.metric or "cosine",
            "Random sample of your table; queries held out.",
        )
    raise SystemExit("choose the vectors: --dataset NAME, --file PATH, or --dsn ... --table T --column C")
