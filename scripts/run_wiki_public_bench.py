#!/usr/bin/env python3
"""
Run evaluation/wiki_public_bench.json against a live DocuMind API (public library).

  python scripts/run_wiki_public_bench.py --base-url http://127.0.0.1:8001
  python scripts/run_wiki_public_bench.py --base-url http://127.0.0.1:8001 --snapshot-only

Prints corpus snapshot from /api/v1/libraries, then per-case: status, has_answer, source count, latency.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "evaluation" / "wiki_public_bench.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="HTTP bench for wiki_public_bench.json (library=public).")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--bench", type=Path, default=BENCH)
    ap.add_argument("--snapshot-only", action="store_true", help="Only GET /api/v1/libraries and exit")
    ap.add_argument("--max-cases", type=int, default=0, help="Run first N cases (0=all)")
    ns = ap.parse_args()

    base = ns.base_url.rstrip("/")
    if not ns.bench.is_file():
        print(f"Missing {ns.bench}", file=sys.stderr)
        return 1

    data = json.loads(ns.bench.read_text(encoding="utf-8"))
    cases = data["cases"]

    with httpx.Client(timeout=180.0) as client:
        try:
            libs = client.get(f"{base}/api/v1/libraries").json()
        except httpx.HTTPError as exc:
            print(f"Libraries GET failed: {exc}", file=sys.stderr)
            return 1

        pub = libs.get("public") or {}
        print("=== Corpus snapshot (GET /api/v1/libraries) ===")
        print(
            json.dumps(
                {
                    "public_collection": pub.get("collection_name"),
                    "public_documents": pub.get("paper_count"),
                    "public_chunks": pub.get("total_chunks"),
                    "papers_collection": (libs.get("papers") or {}).get("collection_name"),
                    "papers_documents": (libs.get("papers") or {}).get("paper_count"),
                    "papers_chunks": (libs.get("papers") or {}).get("total_chunks"),
                    "default_library": libs.get("default_library"),
                },
                indent=2,
            )
        )
        print()

        if ns.snapshot_only:
            return 0

        n = len(cases) if not ns.max_cases else min(ns.max_cases, len(cases))
        latencies: list[float] = []
        hits = 0
        for i, row in enumerate(cases[:n]):
            payload = {
                "query": row["query"],
                "library": data.get("library", "public"),
                "top_k": int(row.get("top_k", 10)),
                "query_mode": row.get("query_mode", "general"),
                "section_filter": row.get("section_filter"),
                "use_flare": bool(row.get("use_flare", False)),
            }
            t0 = time.perf_counter()
            r = client.post(f"{base}/api/v1/query", json=payload)
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)
            ok = r.status_code == 200
            body = r.json() if ok else {}
            has_ans = bool(body.get("has_answer")) if ok else False
            nsrc = len(body.get("sources") or []) if ok else 0
            if has_ans and nsrc:
                hits += 1
            flare = body.get("flare_followup_retrieval") if ok else None
            print(
                f"{row['id']:4} status={r.status_code} has_answer={has_ans} sources={nsrc} "
                f"flare_2nd={flare} {ms:.0f}ms  mode={payload['query_mode']} top_k={payload['top_k']}"
            )

        if latencies:
            xs = sorted(latencies)
            p50 = xs[len(xs) // 2]
            p95 = xs[int(0.95 * (len(xs) - 1))]
            print()
            print(f"Cases run: {n}  grounded_hits (has_answer & sources>0): {hits}")
            print(f"Latency ms: p50={p50:.0f} p95={p95:.0f} max={max(latencies):.0f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
