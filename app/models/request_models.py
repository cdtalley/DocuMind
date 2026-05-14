from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.models.library import LibraryId

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
        description="Restrict retrieval to chunks whose detected section matches (see chunker metadata).",
    )
    use_flare: bool = Field(
        default=False,
        description=(
            "If true (or FLARE_ACTIVE_RETRIEVAL in settings), run FLARE-inspired active retrieval: "
            "a short forward-looking draft may trigger a second embedding search, then merge chunks before generation. "
            "Ignored for datasets mode."
        ),
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
