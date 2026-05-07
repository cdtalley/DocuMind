from __future__ import annotations

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

    def answer(
        self, query: str, top_k: int, query_mode: str = "general", section_filter: str | None = None
    ) -> AnswerResponse:
        results = self.embedding_service.search(query, top_k, section_filter)
        filtered = [item for item in results if item["distance"] < self.settings.RELEVANCE_THRESHOLD]

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
                chunks_searched=len(results),
            )

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
            chunks_searched=len(results),
        )
