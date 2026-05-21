#!/usr/bin/env python3
"""
Live HTTP RAG contract eval against a running DocuMind API (real Chroma + Ollama).

Uses the same case table as CI (tests/query_eval_cases.py) via eval_case_violations so
pytest and this script cannot drift on what “pass” means.

Tiered validation (production pattern: separate corpus-agnostic checks from golden QA):

  --tier structural  Default. AnswerResponse shape, query_mode echo, confidence/chunks
                     invariants. Safe on arbitrary libraries.

  --tier full        Identical rules to tests/test_rag_query_suite.py (has_answer, source
                     counts, answer_substrings). Requires an index aligned with the eval
                     fixtures (e.g. seed_eval_corpus parity). Pair with
                     --skip-empty-corpus-cases for mixed corpora.

  --tier http        Status code only (legacy smoke).

Outputs human-readable rows plus p50/p95 latency; optional --csv and --json-out for artifacts.
503 from upstream gets bounded retries (transient dependency failures).

Case list covers: compare-mode context, section filters, FLARE second pass, injection-shaped
queries, multi-doc diversity, long-query caps.

  python scripts/run_query_eval.py --base-url http://127.0.0.1:8001 --tier structural
  python scripts/run_query_eval.py --base-url http://127.0.0.1:8001 --tier full --skip-empty-corpus-cases
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.query_eval_cases import (  # noqa: E402
    QUERY_EVAL_CASES,
    EvalTier,
    eval_case_violations,
    metrics_from_response,
)


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


def _post_with_retries(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    max_retries: int,
) -> httpx.Response:
    last: httpx.Response | None = None
    for attempt in range(max_retries + 1):
        r = client.post(url, json=payload)
        last = r
        if r.status_code != 503 or attempt >= max_retries:
            return r
        time.sleep(0.35 * (attempt + 1))
    assert last is not None
    return last


def main() -> int:
    ap = argparse.ArgumentParser(description="Live HTTP eval for query_eval_cases")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--csv", type=Path, help="Optional path to write per-case metrics CSV")
    ap.add_argument("--json-out", type=Path, help="Optional path to write summary JSON (use - for stdout)")
    ap.add_argument("--skip-empty-corpus-cases", action="store_true", help="Skip cases meant for empty index")
    ap.add_argument(
        "--tier",
        choices=("http", "structural", "full"),
        default="structural",
        help="Contract depth: structural (default), full (=pytest), or http only",
    )
    ap.add_argument("--retries", type=int, default=2, help="Extra POST attempts on HTTP 503")
    ns = ap.parse_args()
    base = ns.base_url.rstrip("/")
    tier = cast(EvalTier, ns.tier)

    rows: list[dict[str, Any]] = []
    try:
        with httpx.Client(timeout=300.0) as client:
            client.get(f"{base}/health/live").raise_for_status()
            for case in QUERY_EVAL_CASES:
                if ns.skip_empty_corpus_cases and case.skip_for_empty_corpus:
                    continue
                payload = {
                    "query": case.query,
                    "library": "papers",
                    "top_k": case.top_k,
                    "query_mode": case.query_mode,
                    "section_filter": case.section_filter,
                    "use_flare": case.use_flare,
                }
                t0 = time.perf_counter()
                r = _post_with_retries(client, f"{base}/api/v1/query", payload, max_retries=ns.retries)
                elapsed = (time.perf_counter() - t0) * 1000
                ct = r.headers.get("content-type", "")
                if ct.startswith("application/json"):
                    body: Any = r.json()
                else:
                    body = {}
                if not isinstance(body, dict):
                    body = {}
                m = metrics_from_response(r.status_code, body, elapsed)
                viol = eval_case_violations(case, r.status_code, body, tier=tier)
                m["case_id"] = case.id
                m["expect_status"] = case.expect_status
                m["tier"] = tier
                m["violations"] = "; ".join(viol)
                m["pass"] = len(viol) == 0
                rows.append(m)
    except Exception as exc:
        print(f"Eval failed: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("No cases executed (all skipped?).", file=sys.stderr)
        return 1

    hdr = "case_id status ms pass viol has_ans conf chunks srcs ans_chars flare fu"
    print(hdr)
    print("-" * len(hdr))
    latencies = [float(m["elapsed_ms"]) for m in rows]
    for m in rows:
        vshort = (m.get("violations") or "")[:44] + ("…" if len(m.get("violations") or "") > 44 else "")
        print(
            f"{m['case_id'][:28]:28} {int(m['http_status']):3} {m['elapsed_ms']:7.0f} "
            f"{str(m['pass']):5} {vshort:44} {str(m.get('has_answer')):5} {m.get('confidence')!s:4} "
            f"{str(m.get('chunks_searched')):5} {m['n_sources']:3} {m['answer_chars']:5} "
            f"{m.get('flare_enabled')} {m.get('flare_followup')}"
        )

    n = len(latencies)
    summary = {
        "base_url": base,
        "tier": tier,
        "cases_run": n,
        "pass_count": sum(1 for m in rows if m["pass"]),
        "fail_count": sum(1 for m in rows if not m["pass"]),
        "latency_ms_p50": round(_percentile_ms(latencies, 50.0), 2),
        "latency_ms_p95": round(_percentile_ms(latencies, 95.0), 2) if n else 0.0,
        "failures": [
            {"case_id": m["case_id"], "violations": m.get("violations")}
            for m in rows
            if not m["pass"]
        ],
    }
    print(
        f"\nSummary: tier={tier} pass={summary['pass_count']}/{n} "
        f"p50={summary['latency_ms_p50']:.0f}ms p95={summary['latency_ms_p95']:.0f}ms"
    )

    if ns.json_out is not None:
        text = json.dumps(summary, indent=2)
        if str(ns.json_out) == "-":
            print(text)
        else:
            ns.json_out.parent.mkdir(parents=True, exist_ok=True)
            ns.json_out.write_text(text, encoding="utf-8")
            print(f"Wrote {ns.json_out}")

    if ns.csv:
        ns.csv.parent.mkdir(parents=True, exist_ok=True)
        with ns.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {ns.csv}")

    if summary["fail_count"]:
        print("\nViolations detail:", file=sys.stderr)
        for m in rows:
            if not m["pass"]:
                print(f"  {m['case_id']}: {m['violations']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
