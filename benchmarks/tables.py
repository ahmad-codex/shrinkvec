"""Print the README's result tables from docs/results/*/results.json, so no number is copied by hand.

uv run python benchmarks/tables.py
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "docs" / "results"


def recall(r: dict) -> str:
    spread = f" ± {r['recall_std']:.3f}" if r.get("repeats", 1) > 1 else ""
    return f"{r['recall']:.3f}{spread}"


def lab_table(names: list[str]) -> str:
    runs = {n: json.loads((RESULTS / n / "results.json").read_text())["lab"] for n in names}
    pipelines = [r["pipeline"] for r in runs[names[0]]]
    head = "| Option | " + " | ".join(f"{n}<br>recall@10" for n in names) + " | Bytes/vector searched (1536 d) |"
    lines = [head, "|---|" + "---:|" * len(names) + "---:|"]
    first = {r["pipeline"]: r for r in runs[names[0]]}
    for p in pipelines:
        cells = []
        for n in names:
            match = next((r for r in runs[n] if r["pipeline"] == p), None)
            cells.append(recall(match) if match else "")
        lines.append(f"| `{p}` | " + " | ".join(cells) + f" | {first[p]['search_bytes']:,.0f} |")
    return "\n".join(lines)


def pg_table(name: str) -> str:
    data = json.loads((RESULTS / name / "results.json").read_text())
    lines = [
        "| Index | Setting | recall@10 | p50 ms | p95 ms | Index size | Build |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for p in data["pg"]:
        lines.append(
            f"| `{p['option']}` | {p['setting']} | {p['recall']:.3f} | {p['p50_ms']:.2f} | {p['p95_ms']:.2f} | "
            f"{p['index_bytes'] / 2**20:,.0f} MiB | {p['build_seconds']:.0f} s |"
        )
    env = data["pg_env"]
    lines.append(
        f"\nTable (the vectors themselves): {env['table_bytes'] / 2**20:,.0f} MiB. "
        f"PostgreSQL {env['server_version'].split()[0]}, "
        + ", ".join(f"{k} {v}" for k, v in env["extensions"].items())
        + f". {data['meta']['machine']}."
    )
    return "\n".join(lines)


def pg_summary(name: str, target: float = 0.95) -> str:
    """One row per index: the first setting that reaches the target, and the best recall it reached."""
    data = json.loads((RESULTS / name / "results.json").read_text())
    lines = [
        f"| Index | Size on disk | First setting with recall@10 ≥ {target} | p50 there | Best recall@10 |",
        "|---|---:|---|---:|---:|",
    ]
    for option in dict.fromkeys(p["option"] for p in data["pg"]):
        points = [p for p in data["pg"] if p["option"] == option]
        ok = next((p for p in points if p["recall"] >= target), None)
        best = max(points, key=lambda p: p["recall"])
        setting = f"{ok['setting']} ({ok['recall']:.3f})" if ok else "not reached"
        p50 = f"{ok['p50_ms']:.2f} ms" if ok else ""
        lines.append(
            f"| `{option}` | {points[0]['index_bytes'] / 2**20:,.0f} MiB | {setting} | {p50} | {best['recall']:.3f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(lab_table(["dbpedia-openai-1536", "glove-100", "sift-128"]))
    print()
    print(pg_summary("dbpedia-openai-1536"))
    print()
    print(pg_table("dbpedia-openai-1536"))
