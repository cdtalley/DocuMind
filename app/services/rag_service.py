from __future__ import annotations

import re

from app.models.response_models import AnswerResponse, SourceCitation
from app.services.embedding_service import ChromaEmbeddingService
from app.utils.ollama_client import OllamaClient

SYSTEM_PROMPTS = {
    "general": "You are DocuMind, an expert data science research assistant. Answer questions based ONLY on the provided research paper context. Cite sources by paper title and section. If the answer is not in the context, say so clearly.",
    "compare": "You are DocuMind, a data science research analyst. Compare how the provided papers approach the topic. Structure your answer as: 1) Overview of approaches, 2) A comparison table (Method | Paper | Key Difference), 3) Summary of trade-offs. Use ONLY information from the context.",
    "methodology": "You are DocuMind. The user wants to understand implementation details. Focus ONLY on methodology, architecture, and implementation sections in the context. List: algorithms used, model architecture, training procedure, key hyperparameters mentioned.",
    "datasets": "You are DocuMind. Extract and list all datasets mentioned in the context. Format as a bullet list: Dataset Name — Paper Title — How it was used. If no datasets are mentioned, say so.",
    "reproduce": "You are DocuMind. The user wants to reproduce results from these papers. List exactly: 1) Required datasets, 2) Model architecture details, 3) Key hyperparameters, 4) Evaluation metrics used, 5) Any missing details that would block reproduction. Use ONLY information from the context.",
}


class RAGService:
    def __init__(
        self, embedding_service: ChromaEmbeddingService, ollama_client: OllamaClient, settings
    ) -> None:
        self.embedding_service = embedding_service
        self.ollama_client = ollama_client
        self.settings = settings

    @staticmethod
    def _keyword_overlap_score(query: str, content: str) -> float:
        query_terms = {term for term in re.findall(r"\w+", query.lower()) if len(term) >= 4}
        if not query_terms:
            return 0.0
        content_terms = set(re.findall(r"\w+", content.lower()))
        overlap = len(query_terms.intersection(content_terms))
        return overlap / max(len(query_terms), 1)

    @staticmethod
    def _select_diverse_sources(items: list[dict], max_items: int, prefer_unique_doc: bool = True) -> list[dict]:
        if not items:
            return []

        selected: list[dict] = []
        seen_doc_ids: set[str] = set()
        seen_content: set[str] = set()

        # Pass 1: prioritize one strong chunk per paper.
        if prefer_unique_doc:
            for item in items:
                doc_id = str(item.get("metadata", {}).get("doc_id", ""))
                content_key = item.get("content", "")[:220].strip().lower()
                if content_key in seen_content:
                    continue
                if doc_id and doc_id in seen_doc_ids:
                    continue
                selected.append(item)
                seen_content.add(content_key)
                if doc_id:
                    seen_doc_ids.add(doc_id)
                if len(selected) >= max_items:
                    return selected

        # Pass 2: fill remaining with non-duplicate content.
        for item in items:
            content_key = item.get("content", "")[:220].strip().lower()
            if content_key in seen_content:
                continue
            selected.append(item)
            seen_content.add(content_key)
            if len(selected) >= max_items:
                return selected

        return selected

    @staticmethod
    def _extract_datasets_from_sources(sources: list[dict]) -> list[tuple[str, str, str]]:
        dataset_tokens = {
            "wmt",
            "glue",
            "squad",
            "multinli",
            "coco",
            "imagenet",
            "cifar-10",
            "higgs",
            "allstate",
            "bookscorpus",
            "wikipedia",
            "ieee-cis",
        }
        results: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()

        for item in sources:
            md = item.get("metadata", {})
            paper_title = md.get("title", "Unknown Paper")
            content = item.get("content", "")
            lower = content.lower()

            found: set[str] = set()
            for token in dataset_tokens:
                if token in lower:
                    found.add(token)

            # Generic pattern: "<Name> dataset" captures unseen datasets.
            for match in re.findall(r"\b([A-Z][A-Za-z0-9\-]{2,})\s+dataset\b", content):
                found.add(match.lower())
            # Generic pattern: "<descriptor> datasets" for broad mentions.
            for match in re.findall(r"\b([A-Za-z][A-Za-z0-9\-\s]{2,40})\s+datasets\b", content):
                cleaned = re.sub(r"\s+", " ", match.strip().lower())
                if "benchmark" in cleaned or "nlp" in cleaned or len(cleaned.split()) <= 3:
                    found.add(cleaned)

            for dataset in sorted(found):
                key = (dataset, paper_title)
                if key in seen:
                    continue
                seen.add(key)
                pretty = dataset.upper() if len(dataset) <= 5 else dataset.title()
                usage = "mentioned in experiments/evaluation context"
                results.append((pretty, paper_title, usage))

        return results

    def answer(
        self, query: str, top_k: int, query_mode: str = "general", section_filter: str | None = None
    ) -> AnswerResponse:
        results = self.embedding_service.search(query, top_k, section_filter)
        reranked = sorted(
            results,
            key=lambda item: item["distance"]
            - (self.settings.KEYWORD_RERANK_WEIGHT * self._keyword_overlap_score(query, item["content"])),
        )
        filtered = [item for item in reranked if item["distance"] < self.settings.RELEVANCE_THRESHOLD]
        used_fallback = False
        if not filtered and reranked and self.settings.ENABLE_FALLBACK_RETRIEVAL:
            # Fallback keeps demo and low-volume collections usable when distance scales vary by model/index.
            filtered = reranked[: min(self.settings.FALLBACK_TOP_N, len(reranked))]
            used_fallback = True
        filtered = self._select_diverse_sources(filtered, max_items=top_k, prefer_unique_doc=True)

        if not filtered:
            return AnswerResponse(
                answer=(
                    "I could not find relevant information in your paper library for this question. "
                    "Try uploading more papers or rephrasing your query."
                ),
                sources=[],
                confidence=0.0,
                has_answer=False,
                query=query,
                model_used=self.settings.LLM_MODEL,
                query_mode=query_mode,
                chunks_searched=len(reranked),
            )

        if query_mode == "datasets":
            extracted = self._extract_datasets_from_sources(filtered)
            if extracted:
                answer_lines = ["Datasets identified across your paper library:"]
                for dataset_name, paper_title, usage in extracted:
                    answer_lines.append(f"- {dataset_name} - {paper_title} - {usage}")
                answer_text = "\n".join(answer_lines)
            else:
                answer_text = "No explicit dataset names were found in the retrieved source passages."
        else:
            context_parts = []
            for i, item in enumerate(filtered):
                metadata = item["metadata"]
                context_parts.append(
                    f"[Source {i + 1}] Paper: {metadata.get('title', 'Unknown')} | "
                    f"Section: {metadata.get('section', 'body')} | "
                    f"Page: {metadata.get('page_number', 0)}\n{item['content']}\n\n"
                )
            context = "".join(context_parts)

            system_prompt = SYSTEM_PROMPTS.get(query_mode, SYSTEM_PROMPTS["general"])
            user_message = f"Context from research papers:\n\n{context}\n\nQuestion: {query}"
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
            answer_text = self.ollama_client.chat(messages)
        if used_fallback and query_mode != "datasets":
            answer_text = f"{answer_text}\n\nNote: best-available passages were used."

        sources = [
            SourceCitation(
                doc_id=item["metadata"].get("doc_id", ""),
                paper_title=item["metadata"].get("title", ""),
                authors=item["metadata"].get("authors", ""),
                year=item["metadata"].get("year", ""),
                section=item["metadata"].get("section", "body"),
                page_number=int(item["metadata"].get("page_number", 0) or 0),
                chunk_index=int(item["metadata"].get("chunk_index", 0) or 0),
                content_preview=item["content"][:250],
                distance=float(item["distance"]),
            )
            for item in filtered
        ]
        confidence = round(1.0 - (sum(item["distance"] for item in filtered) / len(filtered)), 2)

        return AnswerResponse(
            answer=answer_text,
            sources=sources,
            confidence=max(0.0, min(1.0, confidence)),
            has_answer=True,
            query=query,
            query_mode=query_mode,
            model_used=self.settings.LLM_MODEL,
            chunks_searched=len(reranked),
        )
