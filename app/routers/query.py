from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.main import get_embedding_service, get_ollama_client, get_rag_service
from app.models.request_models import QueryRequest
from app.models.response_models import AnswerResponse, CollectionStats
from app.services.embedding_service import ChromaEmbeddingService
from app.services.rag_service import RAGService
from app.utils.ollama_client import OllamaClient
from app.utils.ollama_client import OllamaConnectionError

router = APIRouter()
logger = logging.getLogger("documind.query")


@router.post("/query", response_model=AnswerResponse)
async def query_papers(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
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


@router.get("/collection/stats", response_model=CollectionStats)
async def collection_stats(
    embedding_service: ChromaEmbeddingService = Depends(get_embedding_service),
) -> CollectionStats:
    return CollectionStats(**embedding_service.collection_stats())
