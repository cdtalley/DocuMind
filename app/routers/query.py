from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import get_settings
from app.main import get_embedding_registry, get_ollama_client
from app.models.library import LibraryId
from app.models.request_models import QueryRequest
from app.models.response_models import AnswerResponse, CollectionStats, LibrariesResponse
from app.services.embedding_registry import EmbeddingRegistry
from app.services.rag_service import RAGService
from app.utils.ollama_client import OllamaClient
from app.utils.ollama_client import OllamaConnectionError

router = APIRouter()
logger = logging.getLogger("documind.query")


def get_rag_for_query(
    request: QueryRequest,
    registry: EmbeddingRegistry = Depends(get_embedding_registry),
) -> RAGService:
    return registry.rag(request.library)


@router.post("/query", response_model=AnswerResponse)
async def query_papers(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_for_query),
    ollama_client: OllamaClient = Depends(get_ollama_client),
) -> AnswerResponse:
    if not ollama_client.health_check().get("available", False):
        raise HTTPException(status_code=503, detail="Ollama is unavailable. Start Ollama first (`ollama serve`).")
    try:
        return rag_service.answer(
            query=request.query,
            top_k=request.top_k,
            query_mode=request.query_mode,
            section_filter=request.section_filter,
            use_flare=request.use_flare,
            retrieval_strategy=request.retrieval_strategy,
            retrieve_only=request.retrieve_only,
        )
    except OllamaConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is unavailable. Start Ollama first (`ollama serve`). Details: {exc}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("RAG query failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Answer generation failed. Check API logs for details.",
        ) from exc


@router.get("/libraries", response_model=LibrariesResponse)
async def list_libraries(
    registry: EmbeddingRegistry = Depends(get_embedding_registry),
) -> LibrariesResponse:
    """Both Chroma collections in one call — use for capacity planning and split-brain checks."""
    settings = get_settings()
    return LibrariesResponse(
        public=CollectionStats(**registry.public.collection_stats()),
        papers=CollectionStats(**registry.papers.collection_stats()),
        default_library=settings.DEFAULT_LIBRARY,
    )


@router.get("/collection/stats", response_model=CollectionStats)
async def collection_stats(
    library: LibraryId = Query("public", description="Target index: public | papers"),
    registry: EmbeddingRegistry = Depends(get_embedding_registry),
) -> CollectionStats:
    return CollectionStats(**registry.embedding(library).collection_stats())
