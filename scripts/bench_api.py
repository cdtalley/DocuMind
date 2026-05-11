#!/usr/bin/env python3
"""
Measure latency against a running DocuMind API (real Chroma + Ollama).

Usage (API up on 8001):
  python scripts/bench_api.py --base-url http://127.0.0.1:8001

Does not start servers; fails fast if unreachable.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Callable

import httpx


def _ms(t: float) -> float:
    return t * 1000.0


def _percentile(sorted_samples: list[float], p: float) -> float:
    if not sorted_samples:
        return 0.0
    k = (len(sorted_samples) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_samples) - 1)
    if f == c:
        return sorted_samples[f]
    return sorted_samples[f] + (sorted_samples[c] - sorted_samples[f]) * (k - f)


def _bench(name: str, fn: Callable[[], None], iterations: int) -> dict[str, float]:
    fn()  # warmup
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return {
        "name": name,
        "n": float(iterations),
        "min_ms": _ms(samples[0]),
        "p50_ms": _ms(statistics.median(samples)),
        "p95_ms": _ms(_percentile(samples, 0.95)),
        "max_ms": _ms(samples[-1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="HTTP latency benchmark for DocuMind API")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001", help="API root (no trailing slash)")
    ap.add_argument("-n", "--iterations", type=int, default=30, help="Requests per endpoint")
    ap.add_argument(
        "--query",
        default=(
            "DEMO — Cross-paper benchmark audit. Retrieval: GLUE ImageNet Cora C4 transformers tabular graph. "
            "Compare papers in context with a markdown table: method, exact title, datasets named, claim, limitation."
        ),
        help="Body for POST /api/v1/query",
    )
    ns = ap.parse_args()
    base = ns.base_url.rstrip("/")

    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.get(f"{base}/health/live")
            r.raise_for_status()
    except Exception as exc:
        print(f"Cannot reach API at {base}: {exc}", file=sys.stderr)
        return 1

    results: list[dict[str, float]] = []

    with httpx.Client(timeout=120.0) as client:

        def live() -> None:
            client.get(f"{base}/health/live").raise_for_status()

        def ready() -> None:
            client.get(f"{base}/health/ready").raise_for_status()

        def stats() -> None:
            client.get(f"{base}/api/v1/collection/stats").raise_for_status()

        def query_post() -> None:
            client.post(
                f"{base}/api/v1/query",
                json={"query": ns.query, "top_k": 8, "query_mode": "general"},
            ).raise_for_status()

        results.append(_bench("GET /health/live", live, ns.iterations))
        results.append(_bench("GET /health/ready", ready, ns.iterations))
        results.append(_bench("GET /api/v1/collection/stats", stats, ns.iterations))
        results.append(_bench("POST /api/v1/query", query_post, max(5, ns.iterations // 3)))

    print(f"Base URL: {base}\n")
    hdr = f"{'endpoint':<32} {'n':>4} {'min':>8} {'p50':>8} {'p95':>8} {'max':>8}"
    print(hdr)
    print("-" * len(hdr))
    for row in results:
        print(
            f"{row['name']:<32} {int(row['n']):>4} "
            f"{row['min_ms']:>7.1f} {row['p50_ms']:>7.1f} {row['p95_ms']:>7.1f} {row['max_ms']:>7.1f}"
        )
    print("\nTimes in ms. Query latency includes Ollama generation — dominate variable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
