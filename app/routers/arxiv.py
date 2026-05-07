from __future__ import annotations

import re
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.main import get_document_service, get_embedding_service, get_ollama_client
from app.models.request_models import ArxivFetchRequest, IngestResponse
from app.services.document_service import DocumentService
from app.services.embedding_service import ChromaEmbeddingService
from app.utils.ollama_client import OllamaClient
from app.utils.ollama_client import OllamaConnectionError

router = APIRouter()


@router.post("/fetch-arxiv", response_model=IngestResponse)
async def fetch_arxiv(
    request: ArxivFetchRequest,
    document_service: DocumentService = Depends(get_document_service),
    embedding_service: ChromaEmbeddingService = Depends(get_embedding_service),
    ollama_client: OllamaClient = Depends(get_ollama_client),
) -> IngestResponse:
    from app.main import settings

    arxiv_id = request.arxiv_id.strip().replace("arXiv:", "").strip()
    if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id):
        raise HTTPException(status_code=400, detail=f"Invalid arXiv ID format: {request.arxiv_id}")
    if not ollama_client.health_check().get("available", False):
        raise HTTPException(status_code=503, detail="Ollama is unavailable. Start Ollama first (`ollama serve`).")

    url = f"{settings.ARXIV_BASE_URL}/{arxiv_id}"
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        response = await client.get(url)

    if response.status_code != 200:
        raise HTTPException(status_code=404, detail=f"Paper not found on arXiv: {arxiv_id}")

    doc_id = f"arxiv_{arxiv_id.replace('.', '_')}"
    filename = f"arxiv_{arxiv_id}.pdf"
    start = time.perf_counter()
    try:
        documents, paper_metadata = document_service.process(response.content, filename, doc_id)
        chunks_created = embedding_service.add_documents(documents, doc_id)
    except OllamaConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is unavailable. Start Ollama first (`ollama serve`). Details: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000

    return IngestResponse(
        doc_id=doc_id,
        filename=filename,
        title=paper_metadata["title"],
        authors=paper_metadata["authors"],
        year=paper_metadata["year"],
        chunks_created=chunks_created,
        processing_time_ms=elapsed_ms,
    )
