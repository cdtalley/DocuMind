from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import chromadb
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import get_settings
from app.logging_config import configure_logging
from app.runtime_info import mark_process_started
from app.models.response_models import CollectionStats, HealthResponse, LivenessResponse, ReadinessResponse
from app.services.document_service import DocumentService
from app.services.embedding_registry import EmbeddingRegistry
from app.services.embedding_service import ChromaEmbeddingService
from app.services.rag_service import RAGService
from app.utils.chunker import DocumentChunker
from app.utils.ollama_client import OllamaClient

settings = get_settings()
configure_logging(use_json=settings.LOG_JSON, level_name=settings.LOG_LEVEL)
logger = logging.getLogger("documind")
ollama_client: OllamaClient | None = None
embedding_registry: EmbeddingRegistry | None = None
document_service: DocumentService | None = None


def _chroma_load_is_recoverable(exc: BaseException) -> bool:
    """Chroma 1.x can raise PyO3 PanicException when the SQLite store is corrupt or from an incompatible version."""
    if isinstance(exc, (SystemExit, KeyboardInterrupt)):
        return False
    if exc.__class__.__name__ == "PanicException":
        return True
    msg = str(exc).lower()
    if "out of range for slice" in msg:
        return True
    return False


def _quarantine_chroma_persist_dir(persist: Path) -> None:
    """Rename persist directory aside so a fresh empty store can be created (development recovery only)."""
    if not persist.exists():
        persist.mkdir(parents=True, exist_ok=True)
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    broken = persist.parent / f"{persist.name}.broken.{stamp}"
    persist.rename(broken)
    logger.warning(
        "Renamed unreadable Chroma persist directory to %s — re-index papers/public corpora as needed.",
        broken,
    )
    persist.mkdir(parents=True, exist_ok=True)


def seed_sample_docs(registry: EmbeddingRegistry) -> None:
    """Optional legacy bundle: synthetic DS briefs into the *papers* collection only.

    Wikipedia-first deployments should keep ``SEED_SAMPLE_DOCS=false`` (default) and use
    ``scripts/bulk_index_public.py`` / ``build_public_corpus.py`` for the public index.
    """
    global document_service
    assert document_service is not None
    if not settings.SEED_SAMPLE_DOCS:
        return
    emb = registry.papers
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
            "Sample corpus version %s -> %s: refreshing bundled `sample_*` papers in papers collection.",
            current_version or "(none)",
            target_version,
        )
        for paper in list(emb.list_papers()):
            if str(paper.get("doc_id", "")).startswith("sample_"):
                emb.delete_document(paper["doc_id"])
        marker.write_text(target_version, encoding="utf-8")

    for sample_file in sorted(sample_dir.glob("*.txt")):
        if sample_file.name.startswith("."):
            continue
        doc_id = f"sample_{sample_file.stem}"
        existing = [paper for paper in emb.list_papers() if paper["doc_id"] == doc_id]
        if existing:
            continue
        file_bytes = sample_file.read_bytes()
        docs, _ = document_service.process(file_bytes, sample_file.name, doc_id)
        emb.add_documents(docs, doc_id)
        logger.info("Seeded sample paper: %s", sample_file.name)


def _curated_public_demo_files(sample_dir: Path) -> list[Path]:
    """Hand-authored briefs only — skip the 400-file synthetic sample_corpus_p7_* bundle."""
    return sorted(
        p
        for p in sample_dir.glob("*.txt")
        if not p.name.startswith(".") and not p.name.startswith("sample_corpus_p7_")
    )


def seed_public_if_empty(registry: EmbeddingRegistry) -> None:
    """Index curated demo articles into the public library when Chroma has zero vectors."""
    global document_service
    assert document_service is not None
    if not settings.SEED_PUBLIC_IF_EMPTY:
        return
    public = registry.public
    if int(public.collection_stats().get("total_chunks", 0)) > 0:
        return
    if ollama_client is None or not ollama_client.health_check().get("available", False):
        logger.info("Skipping public demo seed because Ollama is unavailable.")
        return
    project_root = Path(__file__).resolve().parent.parent
    sample_dir = project_root / "data" / "sample_docs"
    if not sample_dir.exists():
        return
    demo_files = _curated_public_demo_files(sample_dir)
    if not demo_files:
        return
    logger.info(
        "Public index empty — seeding %s curated demo article(s) into %s.",
        len(demo_files),
        settings.CHROMA_COLLECTION_PUBLIC,
    )
    seeded = 0
    for sample_file in demo_files:
        doc_id = f"demo_{sample_file.stem}"
        if any(paper["doc_id"] == doc_id for paper in public.list_papers()):
            continue
        file_bytes = sample_file.read_bytes()
        docs, _ = document_service.process(file_bytes, sample_file.name, doc_id)
        public.add_documents(docs, doc_id)
        seeded += 1
        logger.info("Seeded public demo article: %s", sample_file.name)
    logger.info("Public demo seed complete (%s new document(s)).", seeded)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ollama_client, embedding_registry, document_service
    chunker = DocumentChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    ollama_client = OllamaClient(
        base_url=settings.OLLAMA_BASE_URL,
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        request_timeout_sec=float(settings.OLLAMA_REQUEST_TIMEOUT_SEC),
    )

    def _init_chroma_pair() -> tuple[ChromaEmbeddingService, ChromaEmbeddingService]:
        persist_path = Path(settings.CHROMA_PERSIST_DIR)
        persist_path.mkdir(parents=True, exist_ok=True)
        shared = chromadb.PersistentClient(path=str(persist_path.resolve()))
        papers = ChromaEmbeddingService(
            settings.CHROMA_COLLECTION_NAME,
            ollama_client,
            chroma_client=shared,
        )
        public = ChromaEmbeddingService(
            settings.CHROMA_COLLECTION_PUBLIC,
            ollama_client,
            chroma_client=shared,
        )
        return papers, public

    try:
        papers_svc, public_svc = _init_chroma_pair()
    except BaseException as exc:
        if isinstance(exc, (SystemExit, KeyboardInterrupt)):
            raise
        if settings.APP_ENV != "development" or not _chroma_load_is_recoverable(exc):
            raise
        logger.error(
            "Chroma could not open %s (%s). APP_ENV=development: quarantining persist dir.",
            settings.CHROMA_PERSIST_DIR,
            exc,
        )
        _quarantine_chroma_persist_dir(Path(settings.CHROMA_PERSIST_DIR))
        # Chroma's Rust bindings can be left in a bad state after a PyO3 panic; do not open a new
        # PersistentClient in this process. Operator must restart once so a fresh interpreter attaches
        # to the new empty directory (see chroma-core/chroma#5909 and similar).
        raise RuntimeError(
            "Chroma on-disk store was unreadable (version skew, partial write, or corruption). "
            f"It was renamed beside `{Path(settings.CHROMA_PERSIST_DIR).name}` as `*.broken.*`. "
            "Restart the DocuMind process to initialize a fresh store, then re-index corpora."
        ) from exc
    rag_papers = RAGService(
        embedding_service=papers_svc,
        ollama_client=ollama_client,
        settings=settings,
        content_library="papers",
    )
    rag_public = RAGService(
        embedding_service=public_svc,
        ollama_client=ollama_client,
        settings=settings,
        content_library="public",
    )
    embedding_registry = EmbeddingRegistry(
        papers=papers_svc,
        public=public_svc,
        rag_papers=rag_papers,
        rag_public=rag_public,
    )
    document_service = DocumentService(chunker=chunker)
    logger.info("Ollama health: %s", ollama_client.health_check())
    if settings.SEED_SAMPLE_DOCS:
        logger.warning(
            "SEED_SAMPLE_DOCS=true: indexing bundled sample_docs into papers only. "
            "For Wikipedia-primary deployments keep SEED_SAMPLE_DOCS=false and grow CHROMA_COLLECTION_PUBLIC via bulk jobs."
        )
        try:
            seed_sample_docs(embedding_registry)
        except Exception as exc:
            logger.warning("Failed to seed sample docs: %s", exc)
    try:
        seed_public_if_empty(embedding_registry)
    except Exception as exc:
        logger.warning("Failed to seed public demo articles: %s", exc)
    mark_process_started()
    yield
    logger.info("DocuMind shutting down")


_openapi_url = None if settings.DISABLE_OPENAPI else "/openapi.json"
_docs_url = None if settings.DISABLE_OPENAPI else "/docs"
_redoc_url = None if settings.DISABLE_OPENAPI else "/redoc"

app = FastAPI(
    title="DocuMind",
    description="Dual-index RAG API (public + papers). Ollama for embeddings and chat by default.",
    version="1.1.0",
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

if settings.ENABLE_RESPONSE_GZIP:
    app.add_middleware(GZipMiddleware, minimum_size=500)


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

    if (
        settings.API_KEY
        and request.method != "OPTIONS"
        and request.url.path.startswith("/api/v1")
    ):
        supplied = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
        if supplied != settings.API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key header."},
                headers={"X-Request-ID": request_id},
            )

    start = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    if settings.APP_ENV == "production":
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
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


def get_embedding_registry() -> EmbeddingRegistry:
    assert embedding_registry is not None
    return embedding_registry


def get_embedding_service(
    registry: EmbeddingRegistry = Depends(get_embedding_registry),
) -> ChromaEmbeddingService:
    return registry.embedding(settings.DEFAULT_LIBRARY)


def get_document_service() -> DocumentService:
    assert document_service is not None
    return document_service


def get_rag_service(
    registry: EmbeddingRegistry = Depends(get_embedding_registry),
) -> RAGService:
    return registry.rag(settings.DEFAULT_LIBRARY)


def get_papers_embedding_service(
    registry: EmbeddingRegistry = Depends(get_embedding_registry),
) -> ChromaEmbeddingService:
    return registry.papers


from app.routers import arxiv, diagnostics, ingest, papers, query  # noqa: E402

app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(diagnostics.router, prefix="/api/v1", tags=["Diagnostics"])
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
        "message": "DocuMind API — dual-library RAG (public + papers)",
        "docs": None if settings.DISABLE_OPENAPI else "/docs",
        "health": "/health",
        "health_live": "/health/live",
        "health_ready": "/health/ready",
        "libraries": {"public": settings.CHROMA_COLLECTION_PUBLIC, "papers": settings.CHROMA_COLLECTION_NAME},
        "default_library": settings.DEFAULT_LIBRARY,
        "diagnostics": "/api/v1/diagnostics",
    }
