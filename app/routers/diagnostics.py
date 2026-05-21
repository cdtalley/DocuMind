"""Operator diagnostics: version, uptime, and effective retrieval settings (no secrets)."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from app.config import get_settings
from app.main import get_embedding_registry
from app.models.response_models import DiagnosticsResponse
from app.runtime_info import process_started_at_utc
from app.services.embedding_registry import EmbeddingRegistry

router = APIRouter()


@router.get("/diagnostics", response_model=DiagnosticsResponse)
async def get_diagnostics(
    request: Request,
    registry: EmbeddingRegistry = Depends(get_embedding_registry),
) -> DiagnosticsResponse:
    settings = get_settings()
    started = process_started_at_utc()
    now = datetime.now(UTC)
    uptime = float((now - started).total_seconds()) if started else 0.0
    started_iso = started.isoformat().replace("+00:00", "Z") if started else ""

    pub = registry.public.collection_stats()
    pap = registry.papers.collection_stats()

    return DiagnosticsResponse(
        api_version=str(getattr(request.app, "version", "") or "0.0.0"),
        openapi_disabled=bool(settings.DISABLE_OPENAPI),
        git_sha=(os.environ.get("DOCMIND_GIT_SHA") or "").strip(),
        app_env=settings.APP_ENV,
        process_started_at_utc=started_iso,
        uptime_seconds=round(uptime, 3),
        python_version=sys.version.split()[0],
        seed_sample_docs=settings.SEED_SAMPLE_DOCS,
        sample_corpus_version=settings.SAMPLE_CORPUS_VERSION,
        default_library=settings.DEFAULT_LIBRARY,
        chroma_persist_basename=Path(settings.CHROMA_PERSIST_DIR).name,
        chroma_collection_public=settings.CHROMA_COLLECTION_PUBLIC,
        chroma_collection_papers=settings.CHROMA_COLLECTION_NAME,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        top_k_default=settings.TOP_K_RESULTS,
        relevance_threshold_papers=settings.RELEVANCE_THRESHOLD,
        public_relevance_threshold=settings.PUBLIC_RELEVANCE_THRESHOLD,
        keyword_rerank_weight_papers=settings.KEYWORD_RERANK_WEIGHT,
        public_keyword_rerank_weight=settings.PUBLIC_KEYWORD_RERANK_WEIGHT,
        enable_fallback_retrieval=settings.ENABLE_FALLBACK_RETRIEVAL,
        flare_active_retrieval_default=settings.FLARE_ACTIVE_RETRIEVAL,
        public_chunks=int(pub.get("total_chunks", 0)),
        public_docs=int(pub.get("paper_count", 0)),
        papers_chunks=int(pap.get("total_chunks", 0)),
        papers_docs=int(pap.get("paper_count", 0)),
    )
