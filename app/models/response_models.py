from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    doc_id: str
    paper_title: str
    authors: str
    year: str
    section: str
    page_number: int
    chunk_index: int
    content_preview: str
    distance: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    confidence: float
    has_answer: bool
    query: str
    query_mode: str
    model_used: str
    chunks_searched: int
    library: str = Field(
        default="public",
        description="Indexed corpus: public (encyclopedia-scale) or papers (research PDFs).",
    )
    flare_enabled: bool = False
    flare_followup_retrieval: bool = Field(
        default=False,
        description="True when a second retrieval pass ran (draft indicated missing or uncertain evidence).",
    )
    retrieval_strategy: str = Field(
        default="baseline",
        description="Strategy used for this request (baseline | flare | hyde | multi_query).",
    )
    retrieval_passes: int = Field(
        default=1,
        description="Number of vector search passes executed (e.g. 2 when FLARE follow-up ran).",
    )


class PaperCard(BaseModel):
    doc_id: str
    filename: str
    title: str
    authors: str
    year: str
    arxiv_id: str
    chunk_count: int


class CollectionStats(BaseModel):
    total_chunks: int
    paper_count: int
    collection_name: str


class LibrariesResponse(BaseModel):
    """Snapshot of both vector collections for ops dashboards and capacity planning."""

    public: CollectionStats
    papers: CollectionStats
    default_library: str = Field(description="API default when library is omitted on query/ingest.")


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    llm_model: str
    embedding_model: str
    collection_stats: CollectionStats


class LivenessResponse(BaseModel):
    """Process is running (Kubernetes / load balancer liveness)."""

    status: str = "alive"


class DiagnosticsResponse(BaseModel):
    """Operator / dashboard snapshot: build identity, runtime, and active retrieval configuration."""

    api_version: str = Field(description="FastAPI app version string.")
    openapi_disabled: bool = Field(description="True when /docs is turned off (production-style).")
    git_sha: str = Field(default="", description="Set DOCMIND_GIT_SHA in deploy env for traceability.")
    app_env: str
    process_started_at_utc: str = Field(description="ISO-8601 UTC when the API process finished startup.")
    uptime_seconds: float = Field(description="Wall seconds since process_started_at_utc.")
    python_version: str = Field(description="Short interpreter label (e.g. 3.11.9).")
    seed_sample_docs: bool
    sample_corpus_version: str
    default_library: str
    chroma_persist_basename: str = Field(description="Last segment of CHROMA_PERSIST_DIR (no full host paths).")
    chroma_collection_public: str
    chroma_collection_papers: str
    chunk_size: int
    chunk_overlap: int
    top_k_default: int = Field(description="Settings TOP_K_RESULTS (UI top_k overrides per request).")
    relevance_threshold_papers: float
    public_relevance_threshold: float
    keyword_rerank_weight_papers: float
    public_keyword_rerank_weight: float
    enable_fallback_retrieval: bool
    flare_active_retrieval_default: bool
    public_chunks: int
    public_docs: int
    papers_chunks: int
    papers_docs: int


class ReadinessResponse(BaseModel):
    """Dependency checks for traffic (embed + LLM available)."""

    ready: bool
    ollama_available: bool
    chroma_reachable: bool
    total_chunks: int
    paper_count: int
    detail: str = ""
