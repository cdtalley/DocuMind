from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app, get_document_service, get_embedding_registry, get_ollama_client
from app.models.library import LibraryId
from app.models.response_models import AnswerResponse
from app.services.document_service import DocumentService
from app.services.rag_service import RAGService
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

    def answer(
        self,
        query: str,
        top_k: int,
        query_mode: str = "general",
        section_filter: str | None = None,
        use_flare: bool = False,
        retrieval_strategy: str = "baseline",
        retrieve_only: bool = False,
    ):
        strategy = RAGService._effective_retrieval_strategy(
            retrieval_strategy,
            use_flare=use_flare,
            flare_active_default=False,
            query_mode=query_mode,
        )
        flare_on = strategy == "flare"
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
                flare_enabled=flare_on,
                flare_followup_retrieval=False,
                retrieval_strategy=strategy,
                retrieval_passes=1,
                library="public",
            )
        first = results[0]
        answer = "" if retrieve_only else f"Matched content: {first['content'][:120]}"
        return AnswerResponse(
            answer=answer,
            sources=[],
            confidence=0.9,
            has_answer=True,
            query=query,
            query_mode=query_mode,
            model_used="llama3",
            chunks_searched=len(results),
            flare_enabled=flare_on,
            flare_followup_retrieval=False,
            retrieval_strategy=strategy,
            retrieval_passes=1 if strategy != "multi_query" else 3,
            library="public",
        )

    def answer_stream(
        self,
        query: str,
        top_k: int,
        query_mode: str = "general",
        section_filter: str | None = None,
        use_flare: bool = False,
        retrieval_strategy: str = "baseline",
        retrieve_only: bool = False,
    ):
        result = self.answer(
            query,
            top_k,
            query_mode=query_mode,
            section_filter=section_filter,
            use_flare=use_flare,
            retrieval_strategy=retrieval_strategy,
            retrieve_only=retrieve_only,
        )
        payload = result.model_dump(mode="json")
        if result.has_answer or result.sources:
            yield {
                "event": "retrieval",
                "data": {
                    **payload,
                    "sources": payload.get("sources") or [],
                },
            }
        if result.answer:
            yield {"event": "token", "data": {"text": result.answer}}
        yield {"event": "done", "data": payload}


@dataclass
class FakeEmbeddingRegistry:
    """Single fake store behind both library keys (matches local dual-collection wiring)."""

    emb: FakeEmbeddingService
    rag_svc: FakeRagService

    @property
    def papers(self) -> FakeEmbeddingService:
        return self.emb

    @property
    def public(self) -> FakeEmbeddingService:
        return self.emb

    def embedding(self, library: LibraryId) -> FakeEmbeddingService:
        return self.emb

    def rag(self, library: LibraryId) -> FakeRagService:
        return self.rag_svc


@pytest.fixture
def client() -> TestClient:
    embedding = FakeEmbeddingService()
    document = DocumentService(chunker=DocumentChunker(chunk_size=800, chunk_overlap=100))
    rag = FakeRagService(embedding)
    ollama = type("FakeOllama", (), {"health_check": lambda self: {"available": True, "models": ["llama3"]}})()

    reg = FakeEmbeddingRegistry(emb=embedding, rag_svc=rag)
    app.dependency_overrides[get_embedding_registry] = lambda: reg
    app.dependency_overrides[get_document_service] = lambda: document
    app.dependency_overrides[get_ollama_client] = lambda: ollama
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()
