"""Command line: ``shrinkvec lab``, ``shrinkvec pg``, ``shrinkvec run``, ``shrinkvec datasets``."""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .datasets import DATASETS
from .exact import exact_neighbors
from .lab import DEFAULT_PIPELINES, as_dicts, run_lab
from .recommend import from_lab, from_pg
from .sources import from_args


def _log(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


def _source_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("vectors (choose one)")
    g.add_argument("--dataset", choices=sorted(DATASETS), help="a public dataset, downloaded and cached")
    g.add_argument("--file", help=".npy, .fvecs, .bvecs or .parquet")
    g.add_argument("--table", help="sample a table (with --dsn), e.g. public.documents")
    g.add_argument("--column", help="vector column of --table, or list column of a .parquet file")
    p.add_argument(
        "--dsn", default=os.environ.get("SHRINKVEC_DSN"), help="Postgres connection string (or SHRINKVEC_DSN)"
    )
    p.add_argument("--metric", choices=["cosine", "l2"], help="default: the dataset's metric, else cosine")
    p.add_argument("-n", "--n", type=int, help="number of base vectors (default: all, 100,000 for a table)")
    p.add_argument("--queries", type=int, default=1000, help="held-out queries (default 1000)")
    p.add_argument("-k", type=int, default=10, help="neighbours to find (default 10)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target", type=float, default=0.95, help="recall@k the recommendation must keep (default 0.95)")
    p.add_argument("--min-recall", type=float, help="exit 1 if any measured option is below this (for CI)")
    p.add_argument("--out", default="shrinkvec-report", help="report directory (default ./shrinkvec-report)")


def _command_line() -> str:
    """The command as typed, with the connection string replaced so no password lands in a report."""
    out, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            out.append('"$SHRINKVEC_DSN"')
            skip = False
        elif a == "--dsn":
            out.append(a)
            skip = True
        elif a.startswith("--dsn="):
            out.append('--dsn="$SHRINKVEC_DSN"')
        else:
            out.append(shlex.quote(a))
    return "shrinkvec " + " ".join(out)


def _meta(args, source, created: str) -> dict:
    return {
        "source": source.name,
        "note": source.note,
        "rows": int(len(source.base)),
        "dims": int(source.base.shape[1]),
        "metric": source.metric,
        "queries": int(len(source.queries)),
        "k": args.k,
        "seed": args.seed,
        "created": created,
        "command": _command_line(),
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} threads, "
        f"Python {platform.python_version()}",
        "version": __version__,
    }


def _check_floor(values: list[tuple[str, float]], floor: float | None) -> int:
    if floor is None:
        return 0
    below = [(name, r) for name, r in values if r < floor]
    for name, r in below:
        _log(f"FAIL {name}: recall {r:.3f} is below --min-recall {floor:.3f}")
    return 1 if below else 0


def cmd_datasets(args) -> int:
    for d in DATASETS.values():
        print(f"{d.name:<22} {d.dims:>5}d  {d.metric:<6}  {d.description} [{d.license}]")
    return 0


def _run(args, do_lab: bool, do_pg: bool) -> int:
    from .report import write

    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    t0 = time.time()
    source = from_args(args, _log)
    _log(
        f"{source.name}: {len(source.base):,} vectors x {source.base.shape[1]} dims, "
        f"{len(source.queries):,} queries, {source.metric}"
    )
    data: dict = {"meta": _meta(args, source, created), "target": args.target}
    floors: list[tuple[str, float]] = []

    pipelines = (
        [p.strip() for p in args.pipelines.split(",")] if getattr(args, "pipelines", None) else DEFAULT_PIPELINES
    )
    _log(f"exact top-{args.k} for {len(source.queries):,} queries (float64)")
    truth = exact_neighbors(source.base, source.queries, args.k, source.metric)
    if do_lab:
        lab_results, _ = run_lab(
            source.base,
            source.queries,
            source.metric,
            pipelines,
            k=args.k,
            candidates=args.candidates,
            repeats=args.repeats,
            seed=args.seed,
            truth=truth,
            log=_log,
        )
        data["lab"] = as_dicts(lab_results)
        data["lab_recommendation"] = vars(from_lab(lab_results, args.target))
        floors += [(r.pipeline, r.recall) for r in lab_results]

    if do_pg:
        if not args.dsn:
            raise SystemExit("Postgres needs --dsn (or SHRINKVEC_DSN)")
        from .pg import as_dicts as pg_dicts
        from .pg import run_pg

        options = [o.strip() for o in args.indexes.split(",")] if args.indexes else None
        points, env = run_pg(
            args.dsn, source.base, source.queries, truth, source.metric, args.k, options, args.keep, _log
        )
        data["pg"], data["pg_env"] = pg_dicts(points), env
        if points:
            table = args.table or "items"
            data["pg_recommendation"] = vars(
                from_pg(points, args.target, table, env["dims"], source.metric, env["rows"])
            )
        floors += [(f"{p.option} {p.setting}", p.recall) for p in points]

    path = write(Path(args.out), data)
    rec = data.get("pg_recommendation") or data.get("lab_recommendation")
    if rec:
        _log(f"\n{rec['reason']}")
        if rec.get("sql"):
            _log(rec["sql"])
    _log(f"\nreport: {path.resolve()}  ({time.time() - t0:.0f}s)")
    return _check_floor(floors, args.min_recall)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        prog="shrinkvec",
        description="How much can your embeddings shrink before search gets worse? Measured on your own data.",
    )
    parser.add_argument("--version", action="version", version=f"shrinkvec {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    lab = sub.add_parser("lab", help="recall of every compression option, no database needed")
    _source_args(lab)
    lab.add_argument("--pipelines", help=f"comma-separated (default: {','.join(DEFAULT_PIPELINES)})")
    lab.add_argument("--candidates", type=int, default=100, help="candidates kept for a rerank (default 100)")
    lab.add_argument("--repeats", type=int, default=1, help="rebuild randomized codecs with N seeds (mean, std)")

    pg = sub.add_parser("pg", help="build real indexes in Postgres and measure them")
    _source_args(pg)
    pg.add_argument("--indexes", help="comma-separated: hnsw-vector,hnsw-halfvec,hnsw-binary-rerank,vchordrq")
    pg.add_argument("--keep", action="store_true", help="keep the scratch table and indexes")
    pg.set_defaults(candidates=100, pipelines=None, repeats=1)

    run = sub.add_parser("run", help="lab + Postgres (when --dsn is set) + report")
    _source_args(run)
    run.add_argument("--pipelines", help="lab pipelines, comma-separated")
    run.add_argument("--candidates", type=int, default=100)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--indexes", help="Postgres indexes, comma-separated")
    run.add_argument("--keep", action="store_true")

    sub.add_parser("datasets", help="list the public datasets")

    args = parser.parse_args(argv)
    if args.command == "datasets":
        return cmd_datasets(args)
    if args.command == "lab":
        return _run(args, do_lab=True, do_pg=False)
    if args.command == "pg":
        return _run(args, do_lab=False, do_pg=True)
    return _run(args, do_lab=True, do_pg=bool(args.dsn))


if __name__ == "__main__":
    raise SystemExit(main())
