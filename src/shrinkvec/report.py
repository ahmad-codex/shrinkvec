"""A single self-contained HTML report (plus Markdown and JSON) with inline SVG charts."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

from . import __version__
from .recommend import where

# Validated categorical slots (light / dark); a scatter uses at most three plus gray.
SERIES = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"), ("#eda100", "#c98500")]
FAMILY = {"float": 0, "binary": 0, "RaBitQ": 1, "TurboQuant": 2}
FAMILY_NAME = ["pgvector types", "RaBitQ", "TurboQuant", "Other"]

W, H = 720, 380
PAD_L, PAD_R, PAD_T, PAD_B = 64, 150, 24, 52


def esc(x) -> str:
    return html.escape(str(x))


def family_of(pipeline: str) -> int:
    head = pipeline.split(">")[0]
    if head in ("float32", "halfvec", "binary"):
        return 0
    if head.startswith("rabitq"):
        return 1
    if head.startswith("tq"):
        return 2
    return 3


def _scale(lo: float, hi: float, a: float, b: float, log: bool):
    if log:
        lo, hi = math.log10(lo), math.log10(hi)

    def f(v: float) -> float:
        v = math.log10(v) if log else v
        return a + (v - lo) / (hi - lo or 1) * (b - a)

    return f


def _log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10**e
            if lo <= v <= hi:
                ticks.append(v)
    return ticks


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if v >= 10 else f"{v:g}"


def _axes(x, y, xticks, yticks, xlabel: str, ylabel: str) -> list[str]:
    out = []
    for t in yticks:
        out.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        out.append(f'<text class="tick" x="{PAD_L - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.1f}</text>')
    for t in xticks:
        out.append(f'<text class="tick" x="{x(t):.1f}" y="{H - PAD_B + 18}" text-anchor="middle">{_fmt(t)}</text>')
    out.append(f'<line class="axis" x1="{PAD_L}" x2="{W - PAD_R}" y1="{H - PAD_B}" y2="{H - PAD_B}"/>')
    out.append(
        f'<text class="label" x="{(PAD_L + W - PAD_R) / 2}" y="{H - 12}" text-anchor="middle">{esc(xlabel)}</text>'
    )
    out.append(
        f'<text class="label" transform="translate(16 {(PAD_T + H - PAD_B) / 2}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>'
    )
    return out


def _place_label(text: str, cx: float, cy: float, obstacles: list[tuple[float, float, float, float]]) -> str:
    """Put a label next to its point where it overlaps nothing: right, left, above, below, then further out."""
    w, h = 6.6 * len(text) + 4, 13
    candidates = [(9, -h / 2, "start"), (-9, -h / 2, "end"), (-w / 2, -h - 8, "middle"), (-w / 2, 8, "middle")]
    candidates += [
        (9, -h / 2 - 14, "start"),
        (9, -h / 2 + 14, "start"),
        (-9, -h / 2 - 14, "end"),
        (-9, -h / 2 + 14, "end"),
    ]
    for dx, dy, anchor in candidates:
        x0 = cx + dx if anchor == "start" else cx + dx - w if anchor == "end" else cx + dx
        box = (x0, cy + dy, x0 + w, cy + dy + h)
        inside = box[0] >= PAD_L and box[2] <= W - PAD_R + 4 and box[1] >= PAD_T - 6 and box[3] <= H - PAD_B
        if inside and not any(box[0] < o[2] and o[0] < box[2] and box[1] < o[3] and o[1] < box[3] for o in obstacles):
            obstacles.append(box)
            tx = box[0] if anchor != "end" else box[2]
            ta = "start" if anchor != "end" else "end"
            return f'<text class="pl" x="{tx:.1f}" y="{box[3] - 3:.1f}" text-anchor="{ta}">{esc(text)}</text>'
    return ""  # no free spot: the table and the tooltip still name it


def lab_chart(results: list[dict], target: float) -> str:
    xs = [r["search_bytes"] for r in results]
    lo, hi = min(xs) / 1.4, max(xs) * 1.4
    ymin = max(0.0, math.floor(min(r["recall"] for r in results) * 10) / 10 - 0.05)
    x = _scale(lo, hi, PAD_L, W - PAD_R, log=True)
    y = _scale(ymin, 1.0, H - PAD_B, PAD_T, log=False)
    parts = _axes(
        x,
        y,
        _log_ticks(lo, hi),
        [v / 10 for v in range(math.ceil(ymin * 10), 11)],
        "bytes per vector to search (log scale)",
        "recall@10",
    )
    parts.append(f'<line class="target" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(target):.1f}" y2="{y(target):.1f}"/>')
    parts.append(
        f'<text class="tick" x="{W - PAD_R - 4}" y="{y(target) - 6:.1f}" text-anchor="end">target {target:.2f}</text>'
    )

    # Pareto frontier: the best recall available at or below each size.
    frontier, best = [], -1.0
    for r in sorted(results, key=lambda r: r["search_bytes"]):
        if r["recall"] > best:
            frontier.append(r)
            best = r["recall"]
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x(r['search_bytes']):.1f},{y(r['recall']):.1f}" for i, r in enumerate(frontier)
    )
    parts.append(f'<path class="frontier" d="{path}"/>')

    centers = [(x(r["search_bytes"]), y(r["recall"])) for r in results]
    obstacles = [(cx - 7, cy - 7, cx + 7, cy + 7) for cx, cy in centers]
    obstacles.append((W - PAD_R - 80, y(target) - 18, W - PAD_R, y(target) - 2))  # target label
    labels = []
    for r, (cx, cy) in sorted(zip(results, centers, strict=True), key=lambda rc: -rc[0]["recall"]):
        fam = family_of(r["pipeline"])
        cls = f"s{fam}" + (" hollow" if ">" in r["pipeline"] else "")
        tip = f"{r['label']}\nrecall@10 {r['recall']:.3f}\n{r['search_bytes']:,.0f} B/vector ({r['compression']:.1f}x)"
        parts.append(
            f'<g class="pt"><title>{esc(tip)}</title><circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="10"/>'
            f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="5"/></g>'
        )
        labels.append(_place_label(r["pipeline"], cx, cy, obstacles))
    parts += labels

    legend = []
    used = sorted({family_of(r["pipeline"]) for r in results})
    for i, fam in enumerate(used):
        ly = PAD_T + 14 + i * 20
        legend.append(
            f'<circle class="s{fam}" cx="{W - PAD_R + 22}" cy="{ly - 4}" r="5"/>'
            f'<text class="tick" x="{W - PAD_R + 32}" y="{ly}">{FAMILY_NAME[fam]}</text>'
        )
    ly = PAD_T + 14 + len(used) * 20
    legend.append(
        f'<circle class="s0 hollow" cx="{W - PAD_R + 22}" cy="{ly - 4}" r="5"/>'
        f'<text class="tick" x="{W - PAD_R + 32}" y="{ly}">+ rerank</text>'
    )
    legend.append(
        f'<path class="frontier" d="M{W - PAD_R + 14},{ly + 16} h16"/>'
        f'<text class="tick" x="{W - PAD_R + 32}" y="{ly + 20}">best per size</text>'
    )
    return _svg(parts + legend, "Recall@10 against bytes per vector for every compression option")


def pg_latency_chart(points: list[dict], target: float) -> str:
    options = list(dict.fromkeys(p["option"] for p in points))
    lat = [p["p50_ms"] for p in points]
    lo, hi = min(lat) / 1.3, max(lat) * 1.3
    ymin = max(0.0, math.floor(min(p["recall"] for p in points) * 10) / 10 - 0.05)
    x = _scale(lo, hi, PAD_L, W - PAD_R, log=True)
    y = _scale(ymin, 1.0, H - PAD_B, PAD_T, log=False)
    parts = _axes(
        x,
        y,
        _log_ticks(lo, hi),
        [v / 10 for v in range(math.ceil(ymin * 10), 11)],
        "p50 latency per query, ms (log scale)",
        "recall@10",
    )
    parts.append(f'<line class="target" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(target):.1f}" y2="{y(target):.1f}"/>')
    for n, opt in enumerate(options):
        i = [0, 1, 2, 4, 5][n % 5]
        pts = sorted((p for p in points if p["option"] == opt), key=lambda p: p["p50_ms"])
        d = " ".join(f"{'M' if j == 0 else 'L'}{x(p['p50_ms']):.1f},{y(p['recall']):.1f}" for j, p in enumerate(pts))
        parts.append(f'<path class="line s{i}" d="{d}"/>')
        for p in pts:
            tip = f"{p['label']}\n{p['setting']}\nrecall@10 {p['recall']:.3f}\np50 {p['p50_ms']:.2f} ms, p95 {p['p95_ms']:.2f} ms"
            parts.append(
                f'<g class="pt"><title>{esc(tip)}</title><circle class="hit" cx="{x(p["p50_ms"]):.1f}" '
                f'cy="{y(p["recall"]):.1f}" r="10"/><circle class="s{i}" cx="{x(p["p50_ms"]):.1f}" '
                f'cy="{y(p["recall"]):.1f}" r="4.5"/></g>'
            )
        ly = PAD_T + 14 + n * 20
        parts.append(
            f'<path class="line s{i}" d="M{W - PAD_R + 12},{ly - 4} h18"/>'
            f'<text class="tick" x="{W - PAD_R + 36}" y="{ly}">{esc(opt)}</text>'
        )
    return _svg(parts, "Recall@10 against p50 latency for each Postgres index, over its search knob")


def pg_size_chart(points: list[dict], table_bytes: int) -> str:
    sizes = {}
    for p in points:
        sizes[p["option"]] = p["index_bytes"]
    rows = [("table (the vectors themselves)", table_bytes, True)] + [(k, v, False) for k, v in sizes.items()]
    bar_h, gap = 22, 10
    height = PAD_T + len(rows) * (bar_h + gap) + 30
    left, right = 230, W - 90
    top = max(v for _, v, _ in rows)
    parts = []
    for i, (name, v, ref) in enumerate(rows):
        yy = PAD_T + i * (bar_h + gap)
        w = max(2.0, (right - left) * v / top)
        parts.append(f'<text class="tick" x="{left - 10}" y="{yy + 15}" text-anchor="end">{esc(name)}</text>')
        parts.append(
            f'<g class="pt"><title>{esc(name)}: {v / 2**20:,.1f} MiB</title>'
            f'<rect class="{"ref" if ref else "bar"}" x="{left}" y="{yy}" width="{w:.1f}" height="{bar_h}" rx="4"/></g>'
        )
        parts.append(f'<text class="tick" x="{left + w + 8:.1f}" y="{yy + 15}">{v / 2**20:,.1f} MiB</text>')
    return _svg(parts, "Size on disk of each index and of the table", height=height)


def _svg(parts: list[str], title: str, height: int = H) -> str:
    return (
        f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="{esc(title)}">'
        f"<title>{esc(title)}</title>{''.join(parts)}</svg>"
    )


CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e4e3df;
--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--s3:#a3a29c;--s4:#eda100;--s5:#e87ba4;--accent:#2a78d6;--code:#f4f3f0}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){color-scheme:dark;--bg:#1a1a19;--card:#222220;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;--line:#34332f;--s0:#3987e5;--s1:#d95926;--s2:#199e70;--s3:#77766f;--s4:#c98500;--s5:#d55181;--accent:#3987e5;--code:#2b2b28}}
:root[data-theme=dark]{color-scheme:dark;--bg:#1a1a19;--card:#222220;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;--line:#34332f;
--s0:#3987e5;--s1:#d95926;--s2:#199e70;--s3:#77766f;--s4:#c98500;--s5:#d55181;--accent:#3987e5;--code:#2b2b28}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1060px;margin:0 auto;padding:32px 16px 64px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 10px}
.sub{color:var(--ink2);margin:0 0 20px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0}
.rec{border-left:4px solid var(--accent)}.rec b{font-size:17px}.facts{display:flex;flex-wrap:wrap;gap:8px 28px;color:var(--ink2);font-size:14px}
.facts span b{color:var(--ink);font-weight:600}pre{background:var(--code);border-radius:8px;padding:12px 14px;overflow:auto;font-size:13px}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--ink2);font-weight:600}
td:first-child,th:first-child,td.l,th.l{text-align:left}td.l{white-space:normal;min-width:180px}
th{white-space:normal;vertical-align:bottom}.wrap{overflow-x:auto}.ok{font-weight:600}.pm{color:var(--muted);font-weight:400}
svg{width:100%;height:auto;display:block}svg .grid{stroke:var(--line)}svg .axis{stroke:var(--muted)}
svg .tick,svg .pl{fill:var(--ink2);font-size:11.5px}svg .pl{paint-order:stroke;stroke:var(--card);stroke-width:4px;stroke-linejoin:round}svg .label{fill:var(--ink2);font-size:12.5px}
svg .target{stroke:var(--muted);stroke-dasharray:4 4}svg .frontier{fill:none;stroke:var(--muted);stroke-width:1.5}
svg .s0{fill:var(--s0)}svg .s1{fill:var(--s1)}svg .s2{fill:var(--s2)}svg .s3{fill:var(--s3)}svg .s4{fill:var(--s4)}svg .s5{fill:var(--s5)}
svg circle{stroke:var(--card);stroke-width:2}svg circle.hollow{fill:var(--card);stroke-width:2.5}
svg circle.s0.hollow{stroke:var(--s0)}svg circle.s1.hollow{stroke:var(--s1)}svg circle.s2.hollow{stroke:var(--s2)}svg circle.s3.hollow{stroke:var(--s3)}
svg .line{fill:none;stroke-width:2}svg .line.s0{stroke:var(--s0)}svg .line.s1{stroke:var(--s1)}svg .line.s2{stroke:var(--s2)}svg .line.s4{stroke:var(--s4)}svg .line.s5{stroke:var(--s5)}
svg .hit{fill:transparent;stroke:none}svg .pt:hover circle:not(.hit){r:7}svg .bar{fill:var(--s0)}svg .ref{fill:var(--s3)}
.note{color:var(--ink2);font-size:13.5px}footer{margin-top:40px;color:var(--muted);font-size:13px}
"""


def _lab_table(results: list[dict], target: float) -> str:
    rows = []
    for r in results:
        ok = "ok" if r["recall"] >= target else ""
        cand = f"{r['candidate_recall']:.3f}" if r["candidate_recall"] is not None else ""
        spread = f" <span class='pm'>&plusmn; {r['recall_std']:.3f}</span>" if r.get("repeats", 1) > 1 else ""
        rows.append(
            f"<tr><td><code>{esc(r['pipeline'])}</code></td><td class='l'>{esc(r['label'])}</td>"
            f"<td class='{ok}'>{r['recall']:.3f}{spread}</td><td>{cand}</td><td>{r['search_bytes']:,.0f}</td>"
            f"<td>{r['total_bytes']:,.0f}</td><td>{r['compression']:.1f}x</td><td class='l'>{esc(where(r['pipeline']))}</td></tr>"
        )
    return (
        "<div class='wrap'><table><tr><th>pipeline</th><th class='l'>what it is</th><th>recall@10</th>"
        "<th>true top-10 in candidates</th><th>bytes/vector searched</th><th>bytes/vector stored</th><th>vs float32</th>"
        f"<th class='l'>in Postgres</th></tr>{''.join(rows)}</table></div>"
    )


def _pg_table(points: list[dict], target: float) -> str:
    rows = []
    for p in points:
        ok = "ok" if p["recall"] >= target else ""
        rows.append(
            f"<tr><td><code>{esc(p['option'])}</code></td><td>{esc(p['setting'])}</td>"
            f"<td class='{ok}'>{p['recall']:.3f}</td><td>{p['p50_ms']:.2f}</td><td>{p['p95_ms']:.2f}</td>"
            f"<td>{p['qps']:,.0f}</td><td>{p['index_bytes'] / 2**20:,.1f}</td><td>{p['bytes_per_vector']:,.0f}</td>"
            f"<td>{p['build_seconds']:,.1f}</td></tr>"
        )
    return (
        "<div class='wrap'><table><tr><th>index</th><th>setting</th><th>recall@10</th><th>p50 ms</th><th>p95 ms</th>"
        "<th>QPS (1 client)</th><th>index MiB</th><th>index B/vector</th><th>build s</th></tr>"
        f"{''.join(rows)}</table></div>"
    )


def write(out: Path, data: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (out / "results.md").write_text(markdown(data), encoding="utf-8")
    path = out / "report.html"
    path.write_text(render(data), encoding="utf-8")
    return path


def render(data: dict) -> str:
    meta, target = data["meta"], data["target"]
    rec = data.get("pg_recommendation") or data.get("lab_recommendation") or {}
    facts = "".join(
        f"<span>{esc(k)} <b>{esc(v)}</b></span>"
        for k, v in [
            ("vectors", f"{meta['rows']:,}"),
            ("dimensions", meta["dims"]),
            ("metric", meta["metric"]),
            ("queries", f"{meta['queries']:,}"),
            ("k", meta["k"]),
            ("seed", meta["seed"]),
        ]
    )
    body = [
        f"<h1>shrinkvec report</h1><p class='sub'>{esc(meta['source'])}. {esc(meta.get('note', ''))}</p>",
        f"<div class='facts'>{facts}</div>",
    ]
    if rec:
        sql = f"<pre><code>{esc(rec['sql'])}</code></pre>" if rec.get("sql") else ""
        body.append(
            f"<div class='card rec'><b>Recommendation for recall@10 &ge; {target:.2f}</b><p>{esc(rec['reason'])}</p>{sql}</div>"
        )
    if data.get("lab"):
        body += [
            "<h2>Codec lab: what the codes can do</h2>",
            "<p class='note'>Every option scans all codes exactly, so recall reflects the compression itself, not an index. "
            "Ground truth is exact search in float64. Hollow markers rerank the best candidates with a more precise codec.</p>",
            f"<div class='card'>{lab_chart(data['lab'], target)}</div>",
            _lab_table(data["lab"], target),
        ]
    if data.get("pg"):
        env = data["pg_env"]
        exts = ", ".join(f"{k} {v}" for k, v in env["extensions"].items())
        body += [
            "<h2>Postgres: real indexes on this server</h2>",
            f"<p class='note'>PostgreSQL {esc(env['server_version'])} with {esc(exts)}. Each index is built on a copy of the "
            "vectors, then every query runs once to warm up and once timed, from one client. Latency includes the round trip.</p>",
            f"<div class='card'>{pg_latency_chart(data['pg'], target)}</div>",
            f"<div class='card'>{pg_size_chart(data['pg'], env['table_bytes'])}</div>",
            _pg_table(data["pg"], target),
        ]
    body.append(
        "<h2>Reproduce</h2>"
        f"<pre><code>pip install 'shrinkvec[pg,datasets]'\n{esc(meta['command'])}</code></pre>"
        f"<footer>shrinkvec {__version__} &middot; {esc(meta['created'])} &middot; {esc(meta.get('machine', ''))}</footer>"
    )
    return (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>shrinkvec report</title><style>{CSS}</style></head><body><main>{''.join(body)}</main></body></html>"
    )


def markdown(data: dict) -> str:
    meta, target = data["meta"], data["target"]
    lines = [
        f"# shrinkvec: {meta['source']}",
        "",
        f"{meta['rows']:,} vectors, {meta['dims']} dimensions, {meta['metric']}, {meta['queries']:,} queries, recall@{meta['k']}.",
        "",
    ]
    rec = data.get("pg_recommendation") or data.get("lab_recommendation")
    if rec:
        lines += [f"**Recommendation (recall@10 >= {target:.2f}):** {rec['reason']}", ""]
        if rec.get("sql"):
            lines += ["```sql", rec["sql"], "```", ""]
    if data.get("lab"):
        lines += [
            "## Codec lab",
            "",
            "| pipeline | recall@10 | B/vector searched | B/vector stored | vs float32 |",
            "|---|---:|---:|---:|---:|",
        ]
        lines += [
            f"| `{r['pipeline']}` | {r['recall']:.3f} | {r['search_bytes']:,.0f} | {r['total_bytes']:,.0f} | {r['compression']:.1f}x |"
            for r in data["lab"]
        ]
        lines.append("")
    if data.get("pg"):
        lines += [
            "## Postgres",
            "",
            "| index | setting | recall@10 | p50 ms | p95 ms | index MiB | build s |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        lines += [
            f"| `{p['option']}` | {p['setting']} | {p['recall']:.3f} | {p['p50_ms']:.2f} | {p['p95_ms']:.2f} | "
            f"{p['index_bytes'] / 2**20:,.1f} | {p['build_seconds']:,.1f} |"
            for p in data["pg"]
        ]
        lines.append("")
    lines.append(f"Reproduce: `{meta['command']}`")
    return "\n".join(lines) + "\n"
