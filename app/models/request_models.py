from typing import Literal, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)
    query_mode: Literal["general", "compare", "methodology", "datasets", "reproduce"] = "general"
    section_filter: Optional[str] = None


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
