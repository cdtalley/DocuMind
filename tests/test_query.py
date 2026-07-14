def test_query_empty_collection(client) -> None:
    response = client.post(
        "/api/v1/query",
        json={"query": "What optimizer was used?", "top_k": 6, "query_mode": "general"},
    )
    assert response.status_code == 200
    assert response.json()["has_answer"] is False


def test_query_invalid_section_returns_422(client) -> None:
    response = client.post(
        "/api/v1/query",
        json={
            "query": "test",
            "top_k": 4,
            "query_mode": "general",
            "section_filter": "not-a-real-section",
        },
    )
    assert response.status_code == 422


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


def test_query_use_flare_reflects_in_response(client) -> None:
    ingest = client.post(
        "/api/v1/ingest",
        files={
            "file": (
                "flare_query.txt",
                b"FLARE test document about retrieval and follow-up questions.",
                "text/plain",
            )
        },
    )
    assert ingest.status_code == 200

    response = client.post(
        "/api/v1/query",
        json={
            "query": "What is this about?",
            "top_k": 4,
            "query_mode": "general",
            "use_flare": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["flare_enabled"] is True
    assert "flare_followup_retrieval" in body
    assert body["retrieval_strategy"] == "flare"


def test_query_retrieval_strategy_hyde_echo(client) -> None:
    ingest = client.post(
        "/api/v1/ingest",
        files={
            "file": (
                "hyde_test.txt",
                b"Retrieval strategy test document about vector search and embeddings.",
                "text/plain",
            )
        },
    )
    assert ingest.status_code == 200

    response = client.post(
        "/api/v1/query",
        json={
            "query": "What is this document about?",
            "top_k": 4,
            "query_mode": "general",
            "retrieval_strategy": "hyde",
            "retrieve_only": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_strategy"] in ("hyde", "baseline")
    assert body["answer"] == ""


def test_query_stream_returns_sse_events(client) -> None:
    ingest = client.post(
        "/api/v1/ingest",
        files={
            "file": (
                "stream_test.txt",
                b"Streaming test document about transformers and retrieval augmented generation.",
                "text/plain",
            )
        },
    )
    assert ingest.status_code == 200

    with client.stream(
        "POST",
        "/api/v1/query/stream",
        json={"query": "What is this about?", "top_k": 4, "query_mode": "general"},
    ) as response:
        assert response.status_code == 200
        body = response.read().decode()
        assert "event: retrieval" in body
        assert "event: done" in body


def test_arxiv_invalid_id(client) -> None:
    response = client.post("/api/v1/fetch-arxiv", json={"arxiv_id": "not-valid"})
    assert response.status_code == 400


def test_libraries_snapshot(client) -> None:
    r = client.get("/api/v1/libraries")
    assert r.status_code == 200
    body = r.json()
    assert "public" in body and "papers" in body
    assert body["public"]["collection_name"]
    assert body["papers"]["collection_name"]
    assert body["default_library"] in ("public", "papers")


def test_collection_stats(client) -> None:
    response = client.get("/api/v1/collection/stats")
    assert response.status_code == 200
    assert response.json()["total_chunks"] >= 0
