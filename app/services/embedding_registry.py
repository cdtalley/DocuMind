from __future__ import annotations

from dataclasses import dataclass

from app.models.library import LibraryId
from app.services.embedding_service import ChromaEmbeddingService
from app.services.rag_service import RAGService


@dataclass(frozen=True, slots=True)
class EmbeddingRegistry:
    """Routes retrieval + RAG to the correct Chroma collection per library."""

    papers: ChromaEmbeddingService
    public: ChromaEmbeddingService
    rag_papers: RAGService
    rag_public: RAGService

    def embedding(self, library: LibraryId) -> ChromaEmbeddingService:
        return self.public if library == "public" else self.papers

    def rag(self, library: LibraryId) -> RAGService:
        return self.rag_public if library == "public" else self.rag_papers
