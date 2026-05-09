from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.main import get_embedding_service
from app.models.response_models import PaperCard
from app.services.embedding_service import ChromaEmbeddingService

router = APIRouter()


@router.get("/papers", response_model=list[PaperCard])
async def list_papers(
    embedding_service: ChromaEmbeddingService = Depends(get_embedding_service),
) -> list[PaperCard]:
    papers = embedding_service.list_papers()
    return [PaperCard(**paper) for paper in papers]


@router.get("/papers/{doc_id}", response_model=PaperCard)
async def get_paper(
    doc_id: str, embedding_service: ChromaEmbeddingService = Depends(get_embedding_service)
) -> PaperCard:
    papers = embedding_service.list_papers()
    match = next((paper for paper in papers if paper["doc_id"] == doc_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Paper not found: {doc_id}")
    return PaperCard(**match)


@router.delete("/papers/{doc_id}")
async def delete_paper(
    doc_id: str, embedding_service: ChromaEmbeddingService = Depends(get_embedding_service)
) -> dict:
    if not embedding_service.delete_document(doc_id):
        raise HTTPException(status_code=404, detail=f"No indexed document: {doc_id}")
    return {"deleted": True, "doc_id": doc_id}
