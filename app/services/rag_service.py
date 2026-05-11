from __future__ import annotations

import logging
import re

from app.models.response_models import AnswerResponse, SourceCitation
from app.services.embedding_service import ChromaEmbeddingService
from app.utils.ollama_client import OllamaClient

logger = logging.getLogger("documind.rag")

# Substrings matched in lowercased chunk text for structured dataset extraction (datasets mode).
KNOWN_DATASET_HINTS: frozenset[str] = frozenset(
    {
        "wmt",
        "glue",
        "superglue",
        "squad",
        "multinli",
        "coco",
        "ms-coco",
        "imagenet",
        "cifar-10",
        "cifar-100",
        "mnist",
        "fashion-mnist",
        "higgs",
        "allstate",
        "bookscorpus",
        "wikipedia",
        "ieee-cis",
        "kaggle",
        "cora",
        "citeseer",
        "pubmed",
        "librispeech",
        "timit",
        "google news",
        "yahoo answers",
        "cnn/daily mail",
        "cnndm",
        "wikitext",
        "celeba",
        "lsun",
        "ytfcc100m",
        "yfcc100m",
        "open images",
        "cityscapes",
        "ade20k",
        "movielens",
        "criteo",
        "avazu",
        "uci",
        "electricity",
        "traffic",
        "retail",
        "bitcoin",
        "ethereum",
        "c4",
        "jft-300m",
        "imagenet-1k",
        "wordnet",
    }
)

# Shared rules: portfolio-grade depth must still be fully grounded.
_GROUNDING = (
    "You are DocuMind — a staff+ research synthesizer. Non-negotiable grounding:\n"
    "- Use ONLY the context blocks. Never invent papers, metrics, datasets, URLs, hardware, or hyperparameter values.\n"
    "- Every substantive claim needs a **Paper title** (exact from context) in the same bullet/paragraph or the adjacent one.\n"
    "- When the text supports it, go deeper: 2–4 short paragraphs per ### subsection, nested bullets for mechanisms and ablations, "
    "and optional blockquotes for ≤25-word verbatim fragments that appear exactly in the excerpt (quote marks in blockquote).\n"
    "- If evidence is thin, say so and list gaps — never pad with speculation. Skip generic filler words.\n"
    "- You may place a line containing only --- between major ## sections for readability.\n"
)

SYSTEM_PROMPTS = {
    "general": _GROUNDING
    + (
        "Write a thorough, publication-style note. Follow this ## outline in order:\n"
        "## Executive briefing\n"
        "4–7 bullets. Each: crisp claim + why it matters + **Paper title**.\n"
        "## Deep synthesis\n"
        "Several ### themed subsections (expect multiple paragraphs and nested bullet lists). "
        "Trace mechanisms, training/eval choices, and how papers relate when the passages allow.\n"
        "## Empirical anchors\n"
        "If the context states numbers (accuracy, scaling, dataset sizes, loss values), list them here in a small table or bullets with **Paper title**. "
        "If none, write *No quantitative anchors in excerpts.*\n"
        "## Open questions & coverage limits\n"
        "Bullets: what the user cannot conclude from these excerpts alone.\n"
    ),
    "compare": _GROUNDING
    + (
        "Produce a deep comparative analysis. Outline:\n"
        "## At a glance\n"
        "4–6 bullets: sharpest contrasts, shared assumptions, or ranking hints — each tied to **Paper title**.\n"
        "## Narrative overview\n"
        "Two short paragraphs (8–14 sentences total) weaving the story the papers support.\n"
        "## Comparison table\n"
        "Full GFM table. Columns: Method / paradigm | **Paper (exact title)** | Datasets / benchmarks | "
        "Reported claim or metric | Limitation or scope | Why a practitioner would care\n"
        "Add one row per distinct paper/method the context covers (merge duplicates).\n"
        "## Mechanism & objective contrast\n"
        "### Losses, objectives, inductive biases\n"
        "### Data & evaluation protocol\n"
        "Nested bullets; cite **Paper title** at least once per bullet cluster.\n"
        "## Trade-offs & decision guide\n"
        "When to pick which line of work; each bullet names papers.\n"
        "## Single-paper fallback\n"
        "If the corpus only supports one work, say it once, then mine that paper deeply.\n"
    ),
    "methodology": _GROUNDING
    + (
        "Extract implementation detail for someone about to code a replication. Outline:\n"
        "## TL;DR for implementers\n"
        "6–10 bullets covering objective, blocks/modules, optimizer, schedule hooks, regularization, batching tricks — each with **Paper title**.\n"
        "## Architecture\n"
        "## Training & optimization\n"
        "## Data pipeline & preprocessing\n"
        "## Hyperparameters & compute\n"
        "## Failure modes called out in text\n"
        "Use nested bullets. Missing detail → 'Not stated in excerpt.'\n"
    ),
    "datasets": (
        _GROUNDING
        + "List datasets or benchmarks using ONLY the context. "
        "Start with ## Dataset inventory then ### At a glance (3–5 bullets summarizing coverage). "
        "Then ### Entries as bullets: `**Dataset** — **Paper title** — usage from passage.` "
        "If none, explain what to ingest next."
    ),
    "reproduce": _GROUNDING
    + (
        "Build a serious reproducibility blueprint. Outline:\n"
        "## Repro snapshot\n"
        "2–3 short paragraphs on what can be re-run vs approximated from these excerpts.\n"
        "## Environment assumptions\n"
        "Bullets — hardware/software only when stated; else *Not stated in excerpt.*\n"
        "## Checklists\n"
        "Task lists (`- [ ]`, `- [x]` only if explicitly confirmed). Subsections:\n"
        "### Data & splits\n### Code & model artifacts\n### Training setup\n### Evaluation & metrics\n### Blockers & missing artifacts\n"
        "Under Blockers, separate *hard* (private data, undisclosed architecture width) from *soft* (missing seed).\n"
    ),
}

# Forward-looking active retrieval (FLARE, Jiang et al. EMNLP 2023). Full FLARE uses token-level
# confidence; local Ollama chat does not expose logprobs, so we use explicit uncertainty markers
# (???) and phrase hedges in a short draft to decide on a second retrieval pass.
FLARE_DRAFT_SYSTEM = (
    "You simulate the next sentences an expert would write while answering the user. "
    "This preview is used ONLY to decide whether more library search is needed.\n"
    "Rules:\n"
    "- Use ONLY information implied by the short excerpt previews below.\n"
    "- Write 2-4 sentences of plain prose (no markdown headers, no bullet lists).\n"
    "- Where the previews do not support a concrete fact, write the exact marker ??? instead of guessing.\n"
    "- If everything needed is already clear, write a confident continuation with no ???.\n"
)


def flare_triggers_follow_up(draft: str) -> bool:
    if "???" in draft:
        return True
    d = draft.lower()
    phrases = (
        "not stated in excerpt",
        "not mentioned in excerpt",
        "cannot determine from",
        "unclear from the excerpt",
        "unknown in excerpt",
        "no evidence in excerpt",
        "not in the excerpt",
    )
    return any(p in d for p in phrases)


class RAGService:
    def __init__(
        self, embedding_service: ChromaEmbeddingService, ollama_client: OllamaClient, settings
    ) -> None:
        self.embedding_service = embedding_service
        self.ollama_client = ollama_client
        self.settings = settings

    @staticmethod
    def _chunk_identity(item: dict) -> tuple:
        md = item.get("metadata") or {}
        doc = str(md.get("doc_id", ""))
        try:
            ci = int(md.get("chunk_index", -1) or -1)
        except (TypeError, ValueError):
            ci = -1
        if doc and ci >= 0:
            return ("chunk", doc, ci)
        return ("hash", hash((item.get("content") or "")[:240]))

    def _retrieve_k_budget(self, top_k: int, query_mode: str) -> int:
        if query_mode in ("compare", "general"):
            return min(64, max(top_k * 4, 20))
        if query_mode in ("datasets", "reproduce", "methodology"):
            return min(56, max(top_k * 3, 16))
        return min(64, max(top_k * 4, 20))

    def _context_slots_budget(self, top_k: int, query_mode: str) -> int:
        if query_mode in ("general", "compare"):
            return min(24, top_k + 6)
        if query_mode in ("methodology", "reproduce"):
            return min(22, top_k + 4)
        return min(24, top_k + 6)

    def _search_rerank(
        self,
        embed_query: str,
        overlap_query: str,
        top_k: int,
        query_mode: str,
        section_filter: str | None,
    ) -> list[dict]:
        retrieve_k = self._retrieve_k_budget(top_k, query_mode)
        results = self.embedding_service.search(embed_query, retrieve_k, section_filter)
        w = self.settings.KEYWORD_RERANK_WEIGHT
        return sorted(
            results,
            key=lambda item: item["distance"] - (w * self._keyword_overlap_score(overlap_query, item["content"])),
        )

    def _pick_context_from_reranked(
        self, reranked: list[dict], overlap_query: str, query_mode: str, top_k: int
    ) -> tuple[list[dict], bool]:
        # Compare synthesis needs multiple papers; a single chunk can pass the strict cosine
        # gate while other relevant near-misses sit just above it — then context collapses to one doc.
        thr = float(self.settings.RELEVANCE_THRESHOLD)
        if query_mode == "compare":
            thr = min(0.62, thr + 0.12)
        filtered = [item for item in reranked if item["distance"] < thr]
        used_fallback = False
        if not filtered and reranked and self.settings.ENABLE_FALLBACK_RETRIEVAL:
            filtered = reranked[: min(self.settings.FALLBACK_TOP_N, len(reranked))]
            used_fallback = True
        slots = self._context_slots_budget(top_k, query_mode)
        filtered = self._select_diverse_sources(filtered, max_items=slots, prefer_unique_doc=True)
        return filtered, used_fallback

    def _merge_reranked_passes(self, rer_a: list[dict], rer_b: list[dict], overlap_query: str) -> list[dict]:
        best: dict[tuple, dict] = {}
        for item in rer_a + rer_b:
            k = self._chunk_identity(item)
            cur = best.get(k)
            if cur is None or float(item["distance"]) < float(cur["distance"]):
                best[k] = item
        merged = list(best.values())
        w = self.settings.KEYWORD_RERANK_WEIGHT
        return sorted(
            merged,
            key=lambda item: item["distance"] - (w * self._keyword_overlap_score(overlap_query, item["content"])),
        )

    def _flare_mini_context(self, filtered: list[dict]) -> str:
        budget = max(400, self.settings.FLARE_DRAFT_MAX_CONTEXT_CHARS)
        parts: list[str] = []
        used = 0
        for i, item in enumerate(filtered):
            md = item.get("metadata") or {}
            title = md.get("title", "Unknown")
            sec = md.get("section", "body")
            snippet = (item.get("content") or "").strip()
            block = f"[{i + 1}] {title} ({sec})\n{snippet}\n\n"
            if used + len(block) > budget:
                remain = budget - used - 50
                if remain > 120:
                    parts.append(f"[{i + 1}] {title} ({sec})\n{snippet[:remain]}...\n\n")
                break
            parts.append(block)
            used += len(block)
        return "".join(parts).strip()

    def _flare_forward_looking_draft(self, user_query: str, mini_context: str) -> str:
        user_message = (
            f"User question:\n{user_query}\n\n"
            f"Excerpt previews from the current retrieval pass:\n{mini_context}\n\n"
            "Write the forward-looking preview now."
        )
        messages = [
            {"role": "system", "content": FLARE_DRAFT_SYSTEM},
            {"role": "user", "content": user_message},
        ]
        return self.ollama_client.chat(messages, temperature=0.12)

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
    def _usage_snippet(content: str, needle: str) -> str:
        lower = content.lower()
        idx = lower.find(needle.lower())
        if idx < 0:
            return "Evaluation or training context in the cited passage."
        line_start = content.rfind("\n", 0, idx) + 1
        line_end = content.find("\n", idx)
        if line_end < 0:
            line_end = min(len(content), idx + 220)
        line = content[line_start:line_end].strip()
        if len(line) > 180:
            line = line[:177] + "..."
        return line if line else "Evaluation or training context in the cited passage."

    @staticmethod
    def _extract_datasets_from_sources(sources: list[dict]) -> list[tuple[str, str, str]]:
        results: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()

        for item in sources:
            md = item.get("metadata", {})
            paper_title = md.get("title", "Unknown Paper")
            content = item.get("content", "")
            lower = content.lower()

            found: set[str] = set()
            for hint in sorted(KNOWN_DATASET_HINTS, key=len, reverse=True):
                if hint in lower:
                    found.add(hint)

            for match in re.findall(r"\b([A-Z][A-Za-z0-9\-]{2,30})\s+dataset\b", content):
                found.add(match.lower())

            for dataset in sorted(found):
                key = (dataset, paper_title)
                if key in seen:
                    continue
                seen.add(key)
                pretty = dataset.upper() if len(dataset) <= 6 and " " not in dataset else dataset.title()
                usage = RAGService._usage_snippet(content, dataset)
                results.append((pretty, paper_title, usage))

        results.sort(key=lambda row: (row[0].lower(), row[1].lower()))
        return results

    def answer(
        self,
        query: str,
        top_k: int,
        query_mode: str = "general",
        section_filter: str | None = None,
        use_flare: bool = False,
    ) -> AnswerResponse:
        """Retrieve, optionally merge FLARE follow-up pass, then answer (or dataset inventory)."""
        flare_on = bool(use_flare or self.settings.FLARE_ACTIVE_RETRIEVAL)
        flare_follow_up = False
        rer1 = self._search_rerank(query, query, top_k, query_mode, section_filter)
        filtered, used_fallback = self._pick_context_from_reranked(rer1, query, query_mode, top_k)
        chunks_searched = len(rer1)

        if flare_on and query_mode != "datasets" and filtered:
            mini = self._flare_mini_context(filtered)
            if mini:
                try:
                    draft = self._flare_forward_looking_draft(query, mini)
                except Exception as exc:
                    logger.warning("FLARE draft call failed; continuing with single pass: %s", exc)
                    draft = ""
                if draft.strip() and flare_triggers_follow_up(draft):
                    follow_q = (
                        f"{query.strip()}\n\nRetrieval focus from model lookahead:\n{draft.strip()}"[:2000]
                    )
                    rer2 = self._search_rerank(follow_q, query, top_k, query_mode, section_filter)
                    merged = self._merge_reranked_passes(rer1, rer2, query)
                    filtered, used_fallback = self._pick_context_from_reranked(merged, query, query_mode, top_k)
                    flare_follow_up = True
                    chunks_searched = len(rer1) + len(rer2)

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
                chunks_searched=chunks_searched,
                flare_enabled=flare_on,
                flare_followup_retrieval=flare_follow_up,
            )

        if query_mode == "datasets":
            extracted = self._extract_datasets_from_sources(filtered)
            if extracted:
                unique_datasets = {row[0] for row in extracted}
                unique_papers = {row[1] for row in extracted}
                answer_lines = [
                    "## Dataset inventory",
                    f"*Library-scoped scan — **{len(unique_datasets)}** dataset labels across **{len(unique_papers)}** papers "
                    f"({len(extracted)} mentions in retrieved chunks).*",
                    "",
                    "### At a glance",
                    f"- **{len(unique_datasets)}** distinct dataset or benchmark names detected",
                    f"- **{len(unique_papers)}** contributing papers in this answer",
                    f"- **{len(extracted)}** total dataset–paper mention rows (sorted below)",
                    "",
                    "### Entries",
                ]
                for dataset_name, paper_title, usage in extracted:
                    answer_lines.append(f"- **{dataset_name}** — **{paper_title}** — _{usage}_")
                answer_text = "\n".join(answer_lines)
            else:
                answer_text = (
                    "## Dataset inventory\n\n"
                    "No named datasets or benchmarks were detected in the retrieved passages. "
                    "Try a broader **Top K**, another mode, or ingest papers whose *experiments* sections mention benchmarks."
                )
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
            user_message = (
                f"Context from research papers:\n\n{context}\n"
                f"Question:\n{query}\n\n"
                "Produce the full structured answer. Be thorough where the passages allow: multi-paragraph ### sections, "
                "nested bullets, and a rich comparison table when in compare mode. "
                "Bold **Paper title** throughout. If a section has little evidence, keep it short and label the gap."
            )
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
            temp = 0.1
            if query_mode in ("general", "compare"):
                temp = 0.28
            elif query_mode in ("methodology", "reproduce"):
                temp = 0.16
            answer_text = self.ollama_client.chat(messages, temperature=temp)
        if used_fallback and query_mode != "datasets":
            answer_text = f"{answer_text}\n\n*Retrieval: using best-matching passages (strict distance threshold not met).*"

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
            chunks_searched=chunks_searched,
            flare_enabled=flare_on,
            flare_followup_retrieval=flare_follow_up,
        )
