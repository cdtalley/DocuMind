from __future__ import annotations

import os
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

# Default Chroma anonymized telemetry off unless explicitly enabled in the environment.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.api.client import Client as ChromaClient
from langchain_core.documents import Document

from app.utils.ollama_client import OllamaClient


class ChromaEmbeddingService:
    def __init__(
        self,
        collection_name: str,
        ollama_client: OllamaClient,
        *,
        persist_dir: str | None = None,
        chroma_client: ChromaClient | None = None,
    ) -> None:
        if (persist_dir is not None) == (chroma_client is not None):
            raise ValueError("Exactly one of persist_dir or chroma_client is required")
        self.client = chroma_client if chroma_client is not None else chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.ollama_client = ollama_client
        self._collection_name = collection_name

    def _stamp_metadata(self, md: dict[str, Any]) -> dict[str, Any]:
        """Provenance for re-embedding jobs, drift analysis, and multi-collection ops."""
        out = dict(md)
        out["embedding_model"] = self.ollama_client.embedding_model
        out["chroma_collection"] = self._collection_name
        out["indexed_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
        return out

    def add_documents(self, documents: list[Document], doc_id: str) -> int:
        if not documents:
            return 0
        ids = [f"{doc_id}_{i}" for i in range(len(documents))]
        embeddings = [self.ollama_client.embed(doc.page_content) for doc in documents]
        documents_texts = [doc.page_content for doc in documents]
        metadatas = [self._stamp_metadata(dict(doc.metadata or {})) for doc in documents]
        self.collection.add(ids=ids, embeddings=embeddings, documents=documents_texts, metadatas=metadatas)
        return len(documents)

    def add_indexed_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """Upsert many precomputed vectors in one Chroma call (bulk indexer path)."""
        if not ids:
            return 0
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("ids, embeddings, documents, metadatas must have equal length")
        stamped = [self._stamp_metadata(dict(m)) for m in metadatas]
        self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=stamped)
        return len(ids)

    def search(self, query: str, top_k: int, section_filter: str | None = None) -> list[dict]:
        query_embedding = self.ollama_client.embed(query)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if section_filter:
            kwargs["where"] = {"section": section_filter}

        raw = self.collection.query(**kwargs)
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        rows = [{"content": d, "metadata": m or {}, "distance": dist} for d, m, dist in zip(docs, metas, distances)]
        rows.sort(key=lambda x: x["distance"])
        return rows

    def document_has_chunks(self, doc_id: str) -> bool:
        res = self.collection.get(where={"doc_id": doc_id}, limit=1)
        ids = res.get("ids") or []
        return len(ids) > 0

    def delete_document(self, doc_id: str) -> bool:
        if not self.document_has_chunks(doc_id):
            return False
        self.collection.delete(where={"doc_id": doc_id})
        return True

    def list_papers(self) -> list[dict]:
        data = self.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        grouped: dict[str, dict] = {}
        counts = defaultdict(int)

        for md in metadatas:
            if not md:
                continue
            current_doc_id = md.get("doc_id", "")
            if not current_doc_id:
                continue
            counts[current_doc_id] += 1
            if current_doc_id not in grouped:
                grouped[current_doc_id] = {
                    "doc_id": current_doc_id,
                    "filename": md.get("filename", ""),
                    "title": md.get("title", ""),
                    "authors": md.get("authors", ""),
                    "year": md.get("year", ""),
                    "arxiv_id": md.get("arxiv_id", ""),
                    "chunk_count": 0,
                }

        for current_doc_id, count in counts.items():
            grouped[current_doc_id]["chunk_count"] = count

        papers = list(grouped.values())

        def year_key(item: dict) -> int:
            try:
                return int(item.get("year") or 0)
            except ValueError:
                return 0

        papers.sort(key=year_key, reverse=True)
        return papers

    def collection_stats(self) -> dict:
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection.name,
            "paper_count": len(self.list_papers()),
        }
