"""
Synthetic API latency regression tests (TestClient + fakes — no real Ollama/Chroma I/O).

Budgets are loose enough for laptops and CI; tighten locally if you profile optimized builds.
"""
from __future__ import annotations

import statistics
import time


def _median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000


def test_health_live_median_under_budget(client) -> None:
    client.get("/health/live")  # warmup
    times: list[float] = []
    for _ in range(40):
        t0 = time.perf_counter()
        r = client.get("/health/live")
        times.append(time.perf_counter() - t0)
        assert r.status_code == 200
    med = _median_ms(times)
    assert med < 400.0, f"/health/live median {med:.1f}ms (budget 400ms)"


def test_health_ready_median_under_budget(client) -> None:
    client.get("/health/ready")  # warmup
    times: list[float] = []
    for _ in range(40):
        t0 = time.perf_counter()
        r = client.get("/health/ready")
        times.append(time.perf_counter() - t0)
        assert r.status_code == 200
    med = _median_ms(times)
    assert med < 500.0, f"/health/ready median {med:.1f}ms (budget 500ms)"


def test_query_empty_collection_median_under_budget(client) -> None:
    payload = {"query": "noop performance probe", "top_k": 6, "query_mode": "general"}
    client.post("/api/v1/query", json=payload)  # warmup
    times: list[float] = []
    for _ in range(25):
        t0 = time.perf_counter()
        r = client.post("/api/v1/query", json=payload)
        times.append(time.perf_counter() - t0)
        assert r.status_code == 200
    med = _median_ms(times)
    assert med < 600.0, f"POST /api/v1/query (empty index) median {med:.1f}ms (budget 600ms)"


def test_collection_stats_median_under_budget(client) -> None:
    client.get("/api/v1/collection/stats")  # warmup
    times: list[float] = []
    for _ in range(40):
        t0 = time.perf_counter()
        r = client.get("/api/v1/collection/stats")
        times.append(time.perf_counter() - t0)
        assert r.status_code == 200
    med = _median_ms(times)
    assert med < 500.0, f"GET /api/v1/collection/stats median {med:.1f}ms (budget 500ms)"
