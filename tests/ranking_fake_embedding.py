"""
In-memory vector fake: ranks chunks by keyword overlap with the query (same signal as RAG rerank helper).

Distances are synthetic (lower = more relevant). Tuned so strong matches sit below Settings.RELEVANCE_THRESHOLD.
"""
from __future__ import annotations

from collections import defaultdict

from app.services.rag_service import RAGService


class RankingFakeEmbeddingService:
    def __init__(self) -> None:
        self.store: dict[str, list[dict]] = defaultdict(list)

    def add_chunk(
        self,
        doc_id: str,
        content: str,
        *,
        title: str,
        section: str = "body",
        authors: str = "Test Author",
        year: str = "2024",
        filename: str = "test.txt",
        arxiv_id: str = "",
        chunk_index: int = 0,
        page_number: int = 1,
    ) -> None:
        md = {
            "doc_id": doc_id,
            "title": title,
            "section": section,
            "authors": authors,
            "year": year,
            "filename": filename,
            "arxiv_id": arxiv_id,
            "chunk_index": chunk_index,
            "page_number": page_number,
        }
        self.store[doc_id].append({"content": content, "metadata": md})

    def search(self, query: str, top_k: int, section_filter: str | None = None) -> list[dict]:
        overlap_q = query
        scored: list[tuple[float, dict]] = []
        for items in self.store.values():
            for item in items:
                md = item["metadata"]
                if section_filter and md.get("section") != section_filter:
                    continue
                ov = RAGService._keyword_overlap_score(overlap_q, item["content"])
                # Synthetic cosine-like distance: strong overlap falls below RELEVANCE_THRESHOLD (0.45).
                distance = 0.52 - 0.48 * min(1.0, ov * 1.05)
                distance = max(0.02, min(0.92, distance))
                scored.append(
                    (
                        distance,
                        {
                            "content": item["content"],
                            "metadata": dict(md),
                            "distance": distance,
                        },
                    )
                )
        scored.sort(key=lambda x: x[0])
        return [row for _, row in scored[:top_k]]

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.store:
            return False
        del self.store[doc_id]
        return True

    def list_papers(self) -> list[dict]:
        payload: list[dict] = []
        for doc_id, chunks in self.store.items():
            if not chunks:
                continue
            md = chunks[0]["metadata"]
            payload.append(
                {
                    "doc_id": doc_id,
                    "filename": md.get("filename", ""),
                    "title": md.get("title", ""),
                    "authors": md.get("authors", ""),
                    "year": md.get("year", ""),
                    "arxiv_id": md.get("arxiv_id", ""),
                    "chunk_count": len(chunks),
                }
            )
        return payload

    def collection_stats(self) -> dict:
        return {
            "total_chunks": sum(len(chunks) for chunks in self.store.values()),
            "collection_name": "ranking_fake",
            "paper_count": len([k for k, v in self.store.items() if v]),
        }


def seed_eval_corpus(emb: RankingFakeEmbeddingService) -> None:
    """Fixed library for query-suite regression tests."""
    emb.add_chunk(
        "doc_nlp",
        "We evaluate on GLUE and SuperGLUE benchmarks using a Transformer encoder. "
        "Training uses AdamW with cosine learning rate schedule.",
        title="Transformer Baselines for GLUE",
        section="experiments",
        chunk_index=0,
    )
    emb.add_chunk(
        "doc_vision",
        "Our model reaches strong accuracy on ImageNet classification and CIFAR-100. "
        "We compare against residual CNN baselines.",
        title="Vision Scaling on ImageNet",
        section="methodology",
        chunk_index=0,
    )
    emb.add_chunk(
        "doc_graph",
        "Node classification uses Cora and PubMed citation graphs with graph convolution layers.",
        title="Semi-supervised Learning on Cora",
        section="experiments",
        chunk_index=0,
    )
    emb.add_chunk(
        "doc_tabular",
        "Gradient boosting on IEEE-CIS fraud features with calibration metrics and ECE reporting.",
        title="Tabular Risk Scoring with Boosting",
        section="results",
        chunk_index=0,
    )
    emb.add_chunk(
        "doc_ts",
        "Streaming PCA detects drift in high-dimensional sensors; evaluation uses temporal splits.",
        title="Drift Detection for Sensor Streams",
        section="methodology",
        chunk_index=0,
    )
    emb.add_chunk(
        "doc_sparse",
        "Abstract only paper mention without dataset names in body text for fallback tests.",
        title="Obscure Methods Note",
        section="abstract",
        chunk_index=0,
    )
