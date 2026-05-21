from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.models.library import LibraryId

RetrievalStrategy = Literal["baseline", "flare", "hyde", "multi_query"]

SectionFilter = Literal[
    "abstract",
    "introduction",
    "methodology",
    "experiments",
    "results",
    "conclusion",
]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    library: LibraryId = Field(
        default="public",
        description="Target index: public (Wikipedia-scale) or papers (PDFs / arXiv / legacy bundle).",
    )
    top_k: int = Field(default=6, ge=1, le=24)
    query_mode: Literal["general", "compare", "methodology", "datasets", "reproduce"] = "general"
    section_filter: Optional[SectionFilter] = Field(
        default=None,
        description=(
            "Restrict retrieval to chunks whose `section` metadata matches. "
            "For **library=papers** (PDFs / arXiv), sections come from heading heuristics (abstract, methodology, …). "
            "For **library=public** (Wikipedia-style text), most chunks are labeled `body`; non-matching filters often "
            "return no hits — omit this field for public queries unless you know sections were set."
        ),
    )
    use_flare: bool = Field(
        default=False,
        description=(
            "Legacy toggle: when true and retrieval_strategy is baseline, selects flare. "
            "Prefer retrieval_strategy=flare for new clients."
        ),
    )
    retrieval_strategy: RetrievalStrategy = Field(
        default="baseline",
        description=(
            "Retrieval path before answer synthesis: baseline (single dense pass), flare (draft + optional 2nd pass), "
            "hyde (hypothetical passage embedding), multi_query (LLM sub-queries + RRF fusion). "
            "Ignored for datasets mode (always baseline extraction)."
        ),
    )
    retrieve_only: bool = Field(
        default=False,
        description="If true, return retrieved sources without final answer LLM synthesis (retrieval ablation).",
    )

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v


class ArxivFetchRequest(BaseModel):
    arxiv_id: str = Field(description="ArXiv paper ID e.g. 2401.12345 or 1706.03762")


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    title: str
    authors: str
    year: str
    chunks_created: int
    processing_time_ms: float
    status: str = "success"
