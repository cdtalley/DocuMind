from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    # Legacy / research PDFs + optional bundled sample_docs (when SEED_SAMPLE_DOCS=true).
    CHROMA_COLLECTION_NAME: str = "documind_papers"
    # Primary portfolio corpus: Wikipedia or other public text bulk-indexed here.
    CHROMA_COLLECTION_PUBLIC: str = "documind_wikipedia"
    # Default API library: public (empty until bulk_index_public / ingest); papers for DS-only demos.
    DEFAULT_LIBRARY: Literal["public", "papers"] = "public"
    # When true, startup indexes data/sample_docs/* into the papers collection (synthetic DS briefs).
    SEED_SAMPLE_DOCS: bool = False
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    TOP_K_RESULTS: int = 6
    RELEVANCE_THRESHOLD: float = 0.45
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
