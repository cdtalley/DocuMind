from pydantic import BaseModel


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
