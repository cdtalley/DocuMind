from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import get_settings
from app.models.response_models import CollectionStats, HealthResponse, LivenessResponse, ReadinessResponse
from app.services.document_service import DocumentService
from app.services.embedding_service import ChromaEmbeddingService
from app.services.rag_service import RAGService
from app.utils.chunker import DocumentChunker
from app.utils.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger("documind")

settings = get_settings()
ollama_client: OllamaClient | None = None
embedding_service: ChromaEmbeddingService | None = None
document_service: DocumentService | None = None
rag_service: RAGService | None = None


def seed_sample_docs() -> None:
    global document_service, embedding_service
    assert document_service is not None and embedding_service is not None
    if ollama_client is None or not ollama_client.health_check().get("available", False):
        logger.info("Skipping sample doc indexing because Ollama is unavailable.")
        return
    project_root = Path(__file__).resolve().parent.parent
    sample_dir = project_root / "data" / "sample_docs"
    if not sample_dir.exists():
        return

    persist = Path(settings.CHROMA_PERSIST_DIR)
    persist.mkdir(parents=True, exist_ok=True)
    marker = persist / ".sample_corpus_version"
    target_version = settings.SAMPLE_CORPUS_VERSION
    current_version = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""

    if current_version != target_version:
        logger.info(
            "Sample corpus version %s -> %s: refreshing bundled `sample_*` papers.",
            current_version or "(none)",
            target_version,
        )
        for paper in list(embedding_service.list_papers()):
            if str(paper.get("doc_id", "")).startswith("sample_"):
                embedding_service.delete_document(paper["doc_id"])
        marker.write_text(target_version, encoding="utf-8")

    for sample_file in sorted(sample_dir.glob("*.txt")):
        if sample_file.name.startswith("."):
            continue
        doc_id = f"sample_{sample_file.stem}"
        existing = [paper for paper in embedding_service.list_papers() if paper["doc_id"] == doc_id]
        if existing:
            continue
        file_bytes = sample_file.read_bytes()
        docs, _ = document_service.process(file_bytes, sample_file.name, doc_id)
        embedding_service.add_documents(docs, doc_id)
        logger.info("Seeded sample paper: %s", sample_file.name)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ollama_client, embedding_service, document_service, rag_service
    logging.getLogger().setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    chunker = DocumentChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    ollama_client = OllamaClient(
        base_url=settings.OLLAMA_BASE_URL,
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
    )
    embedding_service = ChromaEmbeddingService(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        ollama_client=ollama_client,
    )
    document_service = DocumentService(chunker=chunker)
    rag_service = RAGService(embedding_service=embedding_service, ollama_client=ollama_client, settings=settings)
    logger.info("Ollama health: %s", ollama_client.health_check())
    try:
        seed_sample_docs()
    except Exception as exc:
        logger.warning("Failed to seed sample docs: %s", exc)
    yield
    logger.info("DocuMind shutting down")


_openapi_url = None if settings.DISABLE_OPENAPI else "/openapi.json"
_docs_url = None if settings.DISABLE_OPENAPI else "/docs"
_redoc_url = None if settings.DISABLE_OPENAPI else "/redoc"

app = FastAPI(
    title="DocuMind",
    description="Data Science Research Paper Intelligence — powered by local Ollama. Zero API costs.",
    version="1.0.0",
    lifespan=lifespan,
    openapi_url=_openapi_url,
    docs_url=_docs_url,
    redoc_url=_redoc_url,
)

_cors_origins = settings.cors_origin_list()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)

_th = settings.trusted_host_list()
if _th:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_th)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if not isinstance(detail, str | list | dict):
            detail = str(detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})
    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
    rid = getattr(request.state, "request_id", None) or "unknown"
    logger.error("request_id=%s unhandled error", rid, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "request_id": rid},
    )


@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


def get_ollama_client() -> OllamaClient:
    assert ollama_client is not None
    return ollama_client


def get_embedding_service() -> ChromaEmbeddingService:
    assert embedding_service is not None
    return embedding_service


def get_document_service() -> DocumentService:
    assert document_service is not None
    return document_service


def get_rag_service() -> RAGService:
    assert rag_service is not None
    return rag_service


from app.routers import arxiv, ingest, papers, query  # noqa: E402

app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(arxiv.router, prefix="/api/v1", tags=["ArXiv"])
app.include_router(papers.router, prefix="/api/v1", tags=["Papers"])


@app.get("/health", response_model=HealthResponse)
async def health(
    client: OllamaClient = Depends(get_ollama_client),
    embedding: ChromaEmbeddingService = Depends(get_embedding_service),
) -> HealthResponse:
    status_info = client.health_check()
    stats = CollectionStats(**embedding.collection_stats())
    return HealthResponse(
        status="ok" if status_info["available"] else "degraded",
        ollama_available=status_info["available"],
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        collection_stats=stats,
    )


@app.get("/health/live", response_model=LivenessResponse)
async def health_live() -> LivenessResponse:
    return LivenessResponse()


@app.get("/health/ready", response_model=None)
async def health_ready(
    client: OllamaClient = Depends(get_ollama_client),
    embedding: ChromaEmbeddingService = Depends(get_embedding_service),
) -> JSONResponse:
    ollama_ok = bool(client.health_check().get("available", False))
    chroma_ok = True
    stats_raw: dict = {}
    try:
        stats_raw = embedding.collection_stats()
    except Exception as exc:
        chroma_ok = False
        logger.error("readiness chroma check failed: %s", exc)
    stats = CollectionStats(**stats_raw) if chroma_ok else None
    ready = ollama_ok and chroma_ok
    body = ReadinessResponse(
        ready=ready,
        ollama_available=ollama_ok,
        chroma_reachable=chroma_ok,
        total_chunks=stats.total_chunks if stats else 0,
        paper_count=stats.paper_count if stats else 0,
        detail="" if ready else "Ollama or vector store not ready for inference.",
    )
    code = 200 if ready else 503
    return JSONResponse(status_code=code, content=body.model_dump())


@app.get("/")
async def root() -> dict:
    return {
        "message": "DocuMind API",
        "docs": None if settings.DISABLE_OPENAPI else "/docs",
        "health": "/health",
        "health_live": "/health/live",
        "health_ready": "/health/ready",
    }
