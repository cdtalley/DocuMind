def test_health_returns_200(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_ingest_txt_file(client) -> None:
    payload = b"This paper introduces XGBoost for fraud detection on the IEEE-CIS dataset."
    response = client.post("/api/v1/ingest", files={"file": ("sample.txt", payload, "text/plain")})
    assert response.status_code == 200
    body = response.json()
    assert body["chunks_created"] >= 1


def test_ingest_invalid_extension(client) -> None:
    response = client.post("/api/v1/ingest", files={"file": ("bad.csv", b"col1,col2", "text/csv")})
    assert response.status_code == 400


def test_ingest_file_too_large(client) -> None:
    huge = b"a" * (60 * 1024 * 1024)
    response = client.post("/api/v1/ingest", files={"file": ("big.txt", huge, "text/plain")})
    assert response.status_code == 413


def test_delete_after_ingest(client) -> None:
    payload = b"A paper about transformers and attention mechanisms."
    ingest = client.post("/api/v1/ingest", files={"file": ("delete_me.txt", payload, "text/plain")})
    assert ingest.status_code == 200
    doc_id = ingest.json()["doc_id"]
    deleted = client.delete(f"/api/v1/ingest/{doc_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
