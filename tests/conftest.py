from __future__ import annotations

from collections import defaultdict

import pytest
from fastapi.testclient import TestClient
from langchain.schema import Document

from app.main import app, get_document_service, get_embedding_service, get_ollama_client, get_rag_service
from app.models.response_models import AnswerResponse
from app.services.document_service import DocumentService
from app.utils.chunker import DocumentChunker


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.store: dict[str, list[dict]] = defaultdict(list)

    def add_documents(self, documents: list[Document], doc_id: str) -> int:
        for doc in documents:
            self.store[doc_id].append({"content": doc.page_content, "metadata": doc.metadata, "distance": 0.1})
        return len(documents)

    def search(self, query: str, top_k: int, section_filter: str | None = None) -> list[dict]:
        rows: list[dict] = []
        for items in self.store.values():
            for item in items:
                if section_filter and item["metadata"].get("section") != section_filter:
                    continue
                rows.append(item)
        return rows[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.store:
            return False
        del self.store[doc_id]
        return True

    def list_papers(self) -> list[dict]:
        payload = []
        for doc_id, chunks in self.store.items():
            if not chunks:
                continue
            md = chunks[0]["metadata"]
            payload.append(
                {
                    "doc_id": doc_id,
                    "filename": md.get("filename", ""),
                    "title": md.get("title", ""),
                    "authors": md.get("authors", ""),
                    "year": md.get("year", ""),
                    "arxiv_id": md.get("arxiv_id", ""),
                    "chunk_count": len(chunks),
                }
            )
        return payload

    def collection_stats(self) -> dict:
        return {
            "total_chunks": sum(len(chunks) for chunks in self.store.values()),
            "collection_name": "test_collection",
            "paper_count": len([k for k, v in self.store.items() if v]),
        }


class FakeRagService:
    def __init__(self, embedding_service: FakeEmbeddingService) -> None:
        self.embedding_service = embedding_service

    def answer(self, query: str, top_k: int, query_mode: str = "general", section_filter: str | None = None):
        results = self.embedding_service.search(query, top_k, section_filter)
        if not results:
            return AnswerResponse(
                answer="No relevant papers found.",
                sources=[],
                confidence=0.0,
                has_answer=False,
                query=query,
                query_mode=query_mode,
                model_used="llama3",
                chunks_searched=0,
            )
        first = results[0]
        return AnswerResponse(
            answer=f"Matched content: {first['content'][:120]}",
            sources=[],
            confidence=0.9,
            has_answer=True,
            query=query,
            query_mode=query_mode,
            model_used="llama3",
            chunks_searched=len(results),
        )


@pytest.fixture
def client() -> TestClient:
    embedding = FakeEmbeddingService()
    document = DocumentService(chunker=DocumentChunker(chunk_size=800, chunk_overlap=100))
    rag = FakeRagService(embedding)
    ollama = type("FakeOllama", (), {"health_check": lambda self: {"available": True, "models": ["llama3"]}})()

    app.dependency_overrides[get_embedding_service] = lambda: embedding
    app.dependency_overrides[get_document_service] = lambda: document
    app.dependency_overrides[get_rag_service] = lambda: rag
    app.dependency_overrides[get_ollama_client] = lambda: ollama
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()
