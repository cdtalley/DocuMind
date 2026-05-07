from __future__ import annotations

from fastapi import APIRouter, Depends

from app.main import get_embedding_service, get_rag_service
from app.models.request_models import QueryRequest
from app.models.response_models import AnswerResponse, CollectionStats
from app.services.embedding_service import ChromaEmbeddingService
from app.services.rag_service import RAGService

router = APIRouter()


@router.post("/query", response_model=AnswerResponse)
async def query_papers(request: QueryRequest, rag_service: RAGService = Depends(get_rag_service)) -> AnswerResponse:
    return rag_service.answer(
        query=request.query,
        top_k=request.top_k,
        query_mode=request.query_mode,
        section_filter=request.section_filter,
    )


@router.get("/collection/stats", response_model=CollectionStats)
async def collection_stats(
    embedding_service: ChromaEmbeddingService = Depends(get_embedding_service),
) -> CollectionStats:
    return CollectionStats(**embedding_service.collection_stats())
