def test_health_live_returns_200(client) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_health_ready_ok_when_ollama_up(client) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["ollama_available"] is True
    assert body["chroma_reachable"] is True


def test_diagnostics_returns_snapshot(client) -> None:
    response = client.get("/api/v1/diagnostics")
    assert response.status_code == 200
    body = response.json()
    assert body["api_version"]
    assert "uptime_seconds" in body
    assert body["uptime_seconds"] >= 0
    assert body["default_library"] in ("public", "papers")
    assert "public_chunks" in body and "papers_chunks" in body
    assert body["chunk_size"] > 0
    assert "process_started_at_utc" in body
