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


class ReadinessResponse(BaseModel):
    """Dependency checks for traffic (embed + LLM available)."""

    ready: bool
    ollama_available: bool
    chroma_reachable: bool
    total_chunks: int
    paper_count: int
    detail: str = ""
