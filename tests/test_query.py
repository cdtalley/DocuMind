def test_query_empty_collection(client) -> None:
    response = client.post(
        "/api/v1/query",
        json={"query": "What optimizer was used?", "top_k": 6, "query_mode": "general"},
    )
    assert response.status_code == 200
    assert response.json()["has_answer"] is False


def test_query_general_mode(client) -> None:
    ingest = client.post(
        "/api/v1/ingest",
        files={
            "file": (
                "query_test.txt",
                b"This paper uses Adam optimizer and evaluates on IEEE-CIS fraud detection dataset.",
                "text/plain",
            )
        },
    )
    assert ingest.status_code == 200

    response = client.post(
        "/api/v1/query",
        json={"query": "Which optimizer was used?", "top_k": 6, "query_mode": "general"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["has_answer"] is True


def test_arxiv_invalid_id(client) -> None:
    response = client.post("/api/v1/fetch-arxiv", json={"arxiv_id": "not-valid"})
    assert response.status_code == 400


def test_collection_stats(client) -> None:
    response = client.get("/api/v1/collection/stats")
    assert response.status_code == 200
    assert response.json()["total_chunks"] >= 0
