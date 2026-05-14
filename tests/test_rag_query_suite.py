"""
Twenty-case RAG regression suite: real RAGService + ranking fake Chroma + deterministic Ollama.

Metrics asserted: HTTP status, has_answer, source counts, answer substrings, chunks_searched bounds.
"""
from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_document_service, get_embedding_registry, get_ollama_client
from app.services.document_service import DocumentService
from app.services.embedding_registry import EmbeddingRegistry
from app.services.rag_service import RAGService
from app.utils.chunker import DocumentChunker

from tests.query_eval_cases import QUERY_EVAL_CASES, QueryEvalCase, eval_case_violations
from tests.ranking_fake_embedding import RankingFakeEmbeddingService, seed_eval_corpus


class DeterministicOllama:
    """Avoids network; FLARE draft returns ??? to exercise second retrieval when enabled."""

    def health_check(self) -> dict[str, Any]:
        return {"available": True, "models": ["llama3"]}

    def embed(self, text: str) -> list[float]:
        return [0.01] * 8

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.1) -> str:
        system = messages[0].get("content", "") if messages else ""
        user = messages[-1].get("content", "") if messages else ""
        if "forward-looking preview" in user or "Write the forward-looking preview" in user:
            return "The excerpts mention calibration but ??? details on holdout splits are missing."
        needles = ("GLUE", "ImageNet", "CIFAR", "Cora", "PubMed", "IEEE-CIS", "ECE", "SuperGLUE")
        found = [n for n in needles if n in user]
        lead = "Structured answer."
        if found:
            lead += " Terms: " + ", ".join(found) + "."
        return lead


def _make_client(embedding: RankingFakeEmbeddingService) -> TestClient:
    document = DocumentService(chunker=DocumentChunker(chunk_size=800, chunk_overlap=100))
    ollama = DeterministicOllama()
    from app.config import get_settings

    settings = get_settings()
    rag_papers = RAGService(embedding, ollama, settings, content_library="papers")
    rag_public = RAGService(embedding, ollama, settings, content_library="public")
    reg = EmbeddingRegistry(
        papers=embedding,  # type: ignore[arg-type]
        public=embedding,  # type: ignore[arg-type]
        rag_papers=rag_papers,
        rag_public=rag_public,
    )
    app.dependency_overrides[get_embedding_registry] = lambda: reg
    app.dependency_overrides[get_document_service] = lambda: document
    app.dependency_overrides[get_ollama_client] = lambda: ollama
    return TestClient(app)


@pytest.mark.parametrize("case", QUERY_EVAL_CASES, ids=lambda c: c.id)
def test_query_eval_case(case: QueryEvalCase) -> None:
    emb = RankingFakeEmbeddingService()
    if not case.skip_for_empty_corpus:
        seed_eval_corpus(emb)
    client = _make_client(emb)
    payload = {
        "query": case.query,
        "library": "papers",
        "top_k": case.top_k,
        "query_mode": case.query_mode,
        "section_filter": case.section_filter,
        "use_flare": case.use_flare,
    }
    try:
        t0 = time.perf_counter()
        response = client.post("/api/v1/query", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    finally:
        app.dependency_overrides.clear()

    body = response.json()
    viol = eval_case_violations(case, response.status_code, body, tier="full")
    assert not viol, f"{case.id}: " + "; ".join(viol) + f" | body={response.text[:500]}"

    # Soft perf guard: synthetic stack should stay fast (adjust if CI machines regress)
    assert elapsed_ms < 8000.0, f"{case.id} slow: {elapsed_ms:.0f}ms"


def test_query_whitespace_only_rejected() -> None:
    emb = RankingFakeEmbeddingService()
    seed_eval_corpus(emb)
    client = _make_client(emb)
    try:
        r = client.post("/api/v1/query", json={"query": "   \n\t  ", "top_k": 6})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 422
