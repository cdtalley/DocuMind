from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    # HTTP timeouts for Ollama (embed can spike on long chunks during bulk jobs).
    OLLAMA_REQUEST_TIMEOUT_SEC: int = 120
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    # Legacy env name: Chroma collection for PDFs / arXiv / optional sample_docs seed (papers library).
    CHROMA_COLLECTION_NAME: str = "documind_papers"
    # Primary portfolio corpus: Wikipedia or other public text bulk-indexed here.
    CHROMA_COLLECTION_PUBLIC: str = "documind_wikipedia"
    # Default API library: public (empty until bulk_index_public / ingest); papers for DS-only demos.
    DEFAULT_LIBRARY: Literal["public", "papers"] = "public"
    # When true, startup indexes data/sample_docs/* into the papers collection (synthetic DS briefs).
    # Deprecated for Wikipedia-first / production-style deployments: leave false (default) and bulk-index public text instead.
    SEED_SAMPLE_DOCS: bool = False
    # When true and the public index is empty, ingest curated hand-authored files from data/sample_docs/
    # (excludes sample_corpus_p7_* synthetics) so the dashboard is usable on first boot.
    SEED_PUBLIC_IF_EMPTY: bool = True
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    TOP_K_RESULTS: int = 6
    RELEVANCE_THRESHOLD: float = 0.45
    # Public (encyclopedia) index: looser cosine gate + stronger lexical rerank for same embed model.
    PUBLIC_RELEVANCE_THRESHOLD: float = 0.52
    PUBLIC_KEYWORD_RERANK_WEIGHT: float = 0.22
    MAX_FILE_SIZE_MB: int = 50
    ARXIV_BASE_URL: str = "https://export.arxiv.org/pdf"
    ENABLE_FALLBACK_RETRIEVAL: bool = True
    FALLBACK_TOP_N: int = 3
    KEYWORD_RERANK_WEIGHT: float = 0.15
    # Bump when `data/sample_docs/` changes; triggers purge + re-index of `sample_*` docs on startup.
    SAMPLE_CORPUS_VERSION: str = "7"
    # Comma-separated origins. When CORS_ALLOW_ALL is true, any origin is accepted (local demos only).
    CORS_ORIGINS: str = (
        "http://127.0.0.1:3002,http://localhost:3002,"
        "http://127.0.0.1:3000,http://localhost:3000"
    )
    CORS_ALLOW_ALL: bool = False
    # development | staging | production — affects docs visibility and logging expectations
    APP_ENV: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: str = "INFO"
    # Comma-separated Host headers (e.g. api.example.com,localhost). Empty disables TrustedHostMiddleware.
    TRUSTED_HOSTS: str = ""
    # When True, OpenAPI /docs and /redoc are disabled (recommended behind ingress in production).
    DISABLE_OPENAPI: bool = False
    # When set, all /api/v1/* routes require header X-API-Key matching this value (except OPTIONS for CORS).
    API_KEY: str = ""
    # One line per log entry as JSON (easier for log platforms). When False, use human-readable format.
    LOG_JSON: bool = False
    # Send gzip-compressed responses when client accepts encoding (reduces bandwidth for large JSON).
    ENABLE_RESPONSE_GZIP: bool = True
    # FLARE-inspired active retrieval (Jiang et al., EMNLP 2023 / arXiv:2305.06983): optional second vector search
    # driven by a short forward-looking draft. Ollama does not expose per-token logprobs here; we trigger follow-up
    # retrieval on ??? markers and explicit hedges in the draft (see rag_service).
    FLARE_ACTIVE_RETRIEVAL: bool = False
    # Cap total characters from first-pass chunks fed into the draft prompt (keeps latency predictable).
    FLARE_DRAFT_MAX_CONTEXT_CHARS: int = 3200

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def trusted_host_list(self) -> list[str] | None:
        raw = self.TRUSTED_HOSTS.strip()
        if not raw:
            return None
        return [h.strip() for h in raw.split(",") if h.strip()]

    def cors_origin_list(self) -> list[str]:
        if self.CORS_ALLOW_ALL:
            return ["*"]
        parts = [p.strip() for p in self.CORS_ORIGINS.split(",") if p.strip()]
        return parts if parts else ["http://127.0.0.1:3002", "http://localhost:3002"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
