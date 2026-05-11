#!/usr/bin/env python3
"""
Run the 20-case query suite against a live DocuMind API (real Chroma + Ollama).

Does not seed the eval corpus — results reflect your actual library. Use for integration
demos and latency tables; use pytest tests/test_rag_query_suite.py for deterministic regression.

  python scripts/run_query_eval.py --base-url http://127.0.0.1:8001
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.query_eval_cases import QUERY_EVAL_CASES, metrics_from_response  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Live HTTP eval for query_eval_cases")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--csv", type=Path, help="Optional path to write metrics CSV")
    ap.add_argument("--skip-empty-corpus-cases", action="store_true", help="Skip cases meant for empty index")
    ns = ap.parse_args()
    base = ns.base_url.rstrip("/")

    rows: list[dict] = []
    try:
        with httpx.Client(timeout=300.0) as client:
            client.get(f"{base}/health/live").raise_for_status()
            for case in QUERY_EVAL_CASES:
                if ns.skip_empty_corpus_cases and case.skip_for_empty_corpus:
                    continue
                payload = {
                    "query": case.query,
                    "top_k": case.top_k,
                    "query_mode": case.query_mode,
                    "section_filter": case.section_filter,
                    "use_flare": case.use_flare,
                }
                t0 = time.perf_counter()
                r = client.post(f"{base}/api/v1/query", json=payload)
                elapsed = (time.perf_counter() - t0) * 1000
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                m = metrics_from_response(r.status_code, body if isinstance(body, dict) else {}, elapsed)
                m["case_id"] = case.id
                m["expect_status"] = case.expect_status
                rows.append(m)
    except Exception as exc:
        print(f"Eval failed: {exc}", file=sys.stderr)
        return 1

    hdr = "case_id status ms has_ans conf chunks srcs ans_chars flare fu"
    print(hdr)
    print("-" * len(hdr))
    for m in rows:
        print(
            f"{m['case_id'][:28]:28} {int(m['http_status']):3} {m['elapsed_ms']:7.0f} "
            f"{str(m.get('has_answer')):5} {m.get('confidence')!s:4} {str(m.get('chunks_searched')):5} "
            f"{m['n_sources']:3} {m['answer_chars']:5} {m.get('flare_enabled')} {m.get('flare_followup')}"
        )

    fails = [m for m in rows if m["http_status"] != m["expect_status"]]
    if fails:
        print("\nStatus mismatches:", [f["case_id"] for f in fails], file=sys.stderr)
        return 2

    if ns.csv:
        ns.csv.parent.mkdir(parents=True, exist_ok=True)
        with ns.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {ns.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
