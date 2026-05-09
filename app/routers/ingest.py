from __future__ import annotations

import time
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.main import get_document_service, get_embedding_service, get_ollama_client
from app.models.request_models import IngestResponse
from app.services.document_service import DocumentService
from app.services.embedding_service import ChromaEmbeddingService
from app.utils.ollama_client import OllamaClient
from app.utils.ollama_client import OllamaConnectionError

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
    embedding_service: ChromaEmbeddingService = Depends(get_embedding_service),
    ollama_client: OllamaClient = Depends(get_ollama_client),
) -> IngestResponse:
    filename = file.filename or "uploaded_file"
    extension = f".{filename.split('.')[-1].lower()}" if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file extension. Use .pdf, .docx, or .txt")

    from app.main import settings

    file_bytes = await file.read()
    if len(file_bytes) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File too large. Max allowed: {settings.MAX_FILE_SIZE_MB}MB",
        )
    if not ollama_client.health_check().get("available", False):
        raise HTTPException(status_code=503, detail="Ollama is unavailable. Start Ollama first (`ollama serve`).")

    doc_id = str(uuid4())
    start = time.perf_counter()
    try:
        documents, paper_metadata = document_service.process(file_bytes, filename, doc_id)
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


@router.delete("/ingest/{doc_id}")
async def delete_ingested_document(
    doc_id: str, embedding_service: ChromaEmbeddingService = Depends(get_embedding_service)
) -> dict:
    if not embedding_service.delete_document(doc_id):
        raise HTTPException(status_code=404, detail=f"No indexed document: {doc_id}")
    return {"deleted": True, "doc_id": doc_id}
