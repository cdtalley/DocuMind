#!/usr/bin/env python3
"""
Systematic retrieval-strategy comparison against a live DocuMind API.

Runs the same benchmark cases under baseline, flare, hyde, and multi_query (configurable),
with retrieve_only=true by default so latency reflects retrieval + strategy LLM helpers, not
full answer synthesis.

Outputs:
  - Per (case × strategy) CSV rows
  - Summary JSON
  - Executive Markdown report (paste into slides / Notion)

Examples:

  python scripts/run_retrieval_ablation.py --base-url http://127.0.0.1:8001
  python scripts/run_retrieval_ablation.py --base-url http://127.0.0.1:8001 \\
      --bench evaluation/wiki_public_bench.json --max-cases 12 --report-md reports/ablation.md
  python scripts/run_retrieval_ablation.py --strategies baseline,flare --retrieve-only false
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH = ROOT / "evaluation" / "retrieval_ablation.json"
ALL_STRATEGIES = ("baseline", "flare", "hyde", "multi_query")


def _percentile_ms(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    w = k - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _chunk_keys(sources: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for s in sources:
        doc = str(s.get("doc_id") or "")
        ci = s.get("chunk_index")
        if doc and ci is not None:
            keys.add(f"{doc}:{ci}")
        else:
            prev = (s.get("content_preview") or "")[:80]
            if prev:
                keys.add(f"preview:{hash(prev)}")
    return keys


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _unique_doc_ids(sources: list[dict[str, Any]]) -> int:
    return len({str(s.get("doc_id") or "") for s in sources if s.get("doc_id")})


def _load_bench(path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    library = str(data.get("library", "public"))
    cases = data["cases"]
    return library, cases


def _build_markdown_report(
    *,
    base_url: str,
    bench_path: Path,
    strategies: list[str],
    retrieve_only: bool,
    corpus: dict[str, Any],
    by_strategy: dict[str, dict[str, Any]],
    pairwise_jaccard_vs_baseline: dict[str, float],
    rows: list[dict[str, Any]],
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# DocuMind retrieval strategy ablation",
        "",
        f"- **When:** {ts}",
        f"- **API:** `{base_url}`",
        f"- **Benchmark:** `{bench_path.name}` ({len({r['case_id'] for r in rows})} cases × {len(strategies)} strategies)",
        f"- **Mode:** `retrieve_only={'true' if retrieve_only else 'false'}` (skips final answer LLM when true)",
        "",
        "## Corpus snapshot",
        "",
        f"| Library | Documents | Chunks |",
        f"|---------|-----------|--------|",
        f"| public | {corpus.get('public_docs', '—')} | {corpus.get('public_chunks', '—')} |",
        f"| papers | {corpus.get('papers_docs', '—')} | {corpus.get('papers_chunks', '—')} |",
        "",
        "## Strategy summary (headline metrics)",
        "",
        "| Strategy | Grounded rate | Avg sources | Avg unique docs | Avg chunks searched | p50 latency (ms) | p95 latency (ms) | 2nd-pass rate |",
        "|----------|---------------|-------------|-----------------|---------------------|------------------|--------------------|---------------|",
    ]
    for strat in strategies:
        s = by_strategy[strat]
        lines.append(
            f"| **{strat}** | {s['grounded_rate_pct']:.0f}% | {s['avg_sources']:.2f} | "
            f"{s['avg_unique_docs']:.2f} | {s['avg_chunks_searched']:.1f} | "
            f"{s['latency_ms_p50']:.0f} | {s['latency_ms_p95']:.0f} | {s['followup_rate_pct']:.0f}% |"
        )
    lines.extend(
        [
            "",
            "_Grounded rate = share of runs with `has_answer` and at least one source. "
            "2nd-pass rate = `flare_followup_retrieval` (meaningful for **flare** only)._",
            "",
            "## Overlap vs baseline (mean Jaccard on chunk keys)",
            "",
            "| Strategy | Mean Jaccard vs baseline | Interpretation |",
            "|----------|--------------------------|----------------|",
        ]
    )
    for strat in strategies:
        if strat == "baseline":
            continue
        j = pairwise_jaccard_vs_baseline.get(strat, 0.0)
        note = "high overlap — similar recall" if j >= 0.7 else "moderate divergence" if j >= 0.4 else "low overlap — different chunk sets"
        lines.append(f"| **{strat}** | {j:.3f} | {note} |")
    lines.extend(
        [
            "",
            "## How to talk about this (elevator)",
            "",
            "- **baseline** — single dense-vector pass + keyword rerank + diversity (production default).",
            "- **flare** — FLARE-*inspired*: forward-looking draft; second search only when draft shows `???` or hedges (not token logprobs).",
            "- **hyde** — HyDE: embed a hypothetical passage instead of the raw question (Gao et al.).",
            "- **multi_query** — RAG-Fusion style: LLM sub-queries + reciprocal rank fusion (RRF).",
            "",
            "Pick the strategy that maximizes **grounded rate** and **unique docs** at acceptable **p95 latency** for your corpus slice.",
            "",
            "## References",
            "",
            "- Jiang et al., FLARE — [arXiv:2305.06983](https://arxiv.org/abs/2305.06983)",
            "- Gao et al., HyDE — [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)",
            "- RAG-Fusion / multi-query fusion — standard RRF over ranked lists",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare retrieval strategies on a live DocuMind API")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    ap.add_argument(
        "--strategies",
        default=",".join(ALL_STRATEGIES),
        help=f"Comma-separated subset of: {','.join(ALL_STRATEGIES)}",
    )
    ap.add_argument("--max-cases", type=int, default=0, help="Run first N cases (0=all)")
    ap.add_argument(
        "--retrieve-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip final answer synthesis (default: true, faster ablation)",
    )
    ap.add_argument("--csv", type=Path, help="Write per-run CSV")
    ap.add_argument("--json-out", type=Path, help="Write summary JSON")
    ap.add_argument("--report-md", type=Path, help="Write executive Markdown report")
    ns = ap.parse_args()

    strategies = [s.strip() for s in ns.strategies.split(",") if s.strip()]
    bad = [s for s in strategies if s not in ALL_STRATEGIES]
    if bad:
        print(f"Unknown strategies: {bad}. Allowed: {ALL_STRATEGIES}", file=sys.stderr)
        return 1
    if not ns.bench.is_file():
        print(f"Missing benchmark: {ns.bench}", file=sys.stderr)
        return 1

    library, cases = _load_bench(ns.bench)
    n_cases = len(cases) if not ns.max_cases else min(ns.max_cases, len(cases))
    cases = cases[:n_cases]
    base = ns.base_url.rstrip("/")

    rows: list[dict[str, Any]] = []
    keys_by_case_strategy: dict[tuple[str, str], set[str]] = {}

    try:
        with httpx.Client(timeout=300.0) as client:
            client.get(f"{base}/health/live").raise_for_status()
            libs = client.get(f"{base}/api/v1/libraries").json()
            pub = libs.get("public") or {}
            pap = libs.get("papers") or {}
            corpus = {
                "public_docs": pub.get("paper_count"),
                "public_chunks": pub.get("total_chunks"),
                "papers_docs": pap.get("paper_count"),
                "papers_chunks": pap.get("total_chunks"),
            }

            for case in cases:
                case_id = case["id"]
                for strat in strategies:
                    payload = {
                        "query": case["query"],
                        "library": library,
                        "top_k": int(case.get("top_k", 10)),
                        "query_mode": case.get("query_mode", "general"),
                        "section_filter": case.get("section_filter"),
                        "retrieval_strategy": strat,
                        "retrieve_only": ns.retrieve_only,
                        "use_flare": False,
                    }
                    t0 = time.perf_counter()
                    r = client.post(f"{base}/api/v1/query", json=payload)
                    elapsed = (time.perf_counter() - t0) * 1000
                    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                    if not isinstance(body, dict):
                        body = {}
                    sources = body.get("sources") or []
                    keys = _chunk_keys(sources)
                    grounded = bool(body.get("has_answer")) and len(sources) > 0
                    row = {
                        "case_id": case_id,
                        "strategy": strat,
                        "http_status": r.status_code,
                        "elapsed_ms": round(elapsed, 2),
                        "grounded": grounded,
                        "n_sources": len(sources),
                        "unique_docs": _unique_doc_ids(sources),
                        "confidence": body.get("confidence"),
                        "chunks_searched": body.get("chunks_searched"),
                        "retrieval_passes": body.get("retrieval_passes"),
                        "retrieval_strategy_echo": body.get("retrieval_strategy"),
                        "flare_followup": body.get("flare_followup_retrieval"),
                        "query_mode": payload["query_mode"],
                        "top_k": payload["top_k"],
                    }
                    rows.append(row)
                    if r.status_code == 200:
                        keys_by_case_strategy[(case_id, strat)] = keys
                    print(
                        f"{case_id} {strat:12} status={r.status_code} grounded={grounded} "
                        f"src={row['n_sources']} docs={row['unique_docs']} "
                        f"chunks={row['chunks_searched']} passes={row['retrieval_passes']} "
                        f"flare2={row['flare_followup']} {elapsed:.0f}ms"
                    )
    except Exception as exc:
        print(f"Ablation failed: {exc}", file=sys.stderr)
        return 1

    by_strategy: dict[str, dict[str, Any]] = {}

    for strat in strategies:
        strat_rows = [r for r in rows if r["strategy"] == strat and r["http_status"] == 200]
        latencies = [float(r["elapsed_ms"]) for r in strat_rows]
        grounded_n = sum(1 for r in strat_rows if r["grounded"])
        n = len(strat_rows) or 1
        followups = sum(1 for r in strat_rows if r.get("flare_followup"))
        by_strategy[strat] = {
            "runs": len(strat_rows),
            "grounded_rate_pct": 100.0 * grounded_n / n,
            "avg_sources": sum(r["n_sources"] for r in strat_rows) / n,
            "avg_unique_docs": sum(r["unique_docs"] for r in strat_rows) / n,
            "avg_chunks_searched": sum(float(r.get("chunks_searched") or 0) for r in strat_rows) / n,
            "latency_ms_p50": round(_percentile_ms(latencies, 50.0), 2),
            "latency_ms_p95": round(_percentile_ms(latencies, 95.0), 2) if latencies else 0.0,
            "followup_rate_pct": 100.0 * followups / n,
        }

    pairwise_jaccard_vs_baseline: dict[str, float] = {}
    if "baseline" in strategies:
        for strat in strategies:
            if strat == "baseline":
                continue
            scores: list[float] = []
            for case in cases:
                cid = case["id"]
                b = keys_by_case_strategy.get((cid, "baseline"))
                o = keys_by_case_strategy.get((cid, strat))
                if b is not None and o is not None:
                    scores.append(_jaccard(b, o))
            pairwise_jaccard_vs_baseline[strat] = sum(scores) / len(scores) if scores else 0.0

    summary = {
        "base_url": base,
        "bench": str(ns.bench),
        "library": library,
        "cases": n_cases,
        "strategies": strategies,
        "retrieve_only": ns.retrieve_only,
        "corpus": corpus,
        "by_strategy": by_strategy,
        "pairwise_jaccard_vs_baseline": pairwise_jaccard_vs_baseline,
    }

    print("\n=== Summary ===")
    for strat in strategies:
        s = by_strategy[strat]
        print(
            f"{strat:12} grounded={s['grounded_rate_pct']:.0f}% "
            f"avg_src={s['avg_sources']:.2f} avg_docs={s['avg_unique_docs']:.2f} "
            f"p50={s['latency_ms_p50']:.0f}ms p95={s['latency_ms_p95']:.0f}ms "
            f"flare_2nd={s['followup_rate_pct']:.0f}%"
        )
    if pairwise_jaccard_vs_baseline:
        print("\nJaccard vs baseline:", ", ".join(f"{k}={v:.3f}" for k, v in pairwise_jaccard_vs_baseline.items()))

    if ns.json_out:
        ns.json_out.parent.mkdir(parents=True, exist_ok=True)
        ns.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {ns.json_out}")

    if ns.csv and rows:
        ns.csv.parent.mkdir(parents=True, exist_ok=True)
        with ns.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {ns.csv}")

    if ns.report_md:
        md = _build_markdown_report(
            base_url=base,
            bench_path=ns.bench,
            strategies=strategies,
            retrieve_only=ns.retrieve_only,
            corpus=corpus,
            by_strategy=by_strategy,
            pairwise_jaccard_vs_baseline=pairwise_jaccard_vs_baseline,
            rows=rows,
        )
        ns.report_md.parent.mkdir(parents=True, exist_ok=True)
        ns.report_md.write_text(md, encoding="utf-8")
        print(f"Wrote {ns.report_md}")

    failed = [r for r in rows if r["http_status"] != 200]
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())