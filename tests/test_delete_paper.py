def test_delete_unknown_paper_returns_404(client) -> None:
    response = client.delete("/api/v1/papers/does-not-exist-uuid")
    assert response.status_code == 404
    assert "No indexed document" in response.json()["detail"]


def test_delete_unknown_ingest_returns_404(client) -> None:
    response = client.delete("/api/v1/ingest/does-not-exist-uuid")
    assert response.status_code == 404
