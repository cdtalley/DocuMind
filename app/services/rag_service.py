from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any, Literal

from app.models.library import LibraryId
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

# Shared rules: depth stays grounded to retrieved text only.
_GROUNDING = (
    "You are DocuMind — a research synthesizer. Non-negotiable grounding:\n"
    "- Use ONLY the context blocks. Never invent papers, metrics, datasets, URLs, hardware, or hyperparameter values.\n"
    "- Every substantive claim needs a **Paper title** (exact from context) in the same bullet/paragraph or the adjacent one.\n"
    "- Use research-corpus vocabulary (papers, methods, benchmarks) only when the excerpts support it.\n"
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

_PUBLIC_GROUNDING = (
    "You are DocuMind — a careful encyclopedia-grounded assistant. Non-negotiable grounding:\n"
    "- Use ONLY the context blocks. Never invent facts, dates, people, places, or sources not supported by the excerpts.\n"
    "- Every substantive claim should name the **Article title** (exact string from context metadata) in the same "
    "bullet/paragraph or the adjacent one.\n"
    "- Prefer neutral encyclopedia tone (articles, passages, topics). If excerpts conflict, say so briefly. "
    "Do not frame answers as peer-reviewed paper reviews, GLUE/SuperGLUE leaderboards, or arXiv-style contributions "
    "unless the excerpt text explicitly does.\n"
    "- If evidence is thin, say so — never pad. Short verbatim quotes (≤25 words) only when they appear exactly in the excerpt.\n"
)

PUBLIC_SYSTEM_PROMPTS = {
    "general": _PUBLIC_GROUNDING
    + (
        "## Summary\n"
        "Direct answer to the question in 1–3 short paragraphs, citing **Article title** where relevant.\n"
        "## What the excerpts support\n"
        "Bullets tied to **Article title**.\n"
        "## Limits\n"
        "What cannot be concluded from these excerpts alone.\n"
    ),
    "compare": _PUBLIC_GROUNDING
    + (
        "## Comparison\n"
        "Contrast themes across articles using a markdown table when helpful: Topic | **Article title** | Supported claim | Caveat.\n"
        "## Narrative\n"
        "One or two paragraphs weaving only what the excerpts state.\n"
    ),
    "methodology": _PUBLIC_GROUNDING
    + (
        "## Extracted detail\n"
        "Implementation or process notes exactly as described, with **Article title** on each cluster of bullets.\n"
        "## Missing steps\n"
        "Use *Not stated in excerpt.* where the text is silent.\n"
    ),
    "datasets": _PUBLIC_GROUNDING
    + (
        "List dataset or benchmark names using ONLY the context. "
        "Start with ## Dataset inventory then ### At a glance. "
        "Then ### Entries as bullets: `**Dataset** — **Article title** — usage from passage.` "
        "If none, say so."
    ),
    "reproduce": _PUBLIC_GROUNDING
    + (
        "## What can be reproduced from excerpts\n"
        "Bullets with **Article title**; use checklists only for steps explicitly described.\n"
        "## Blockers\n"
        "What is missing from excerpts for a full reproduction.\n"
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


RetrievalStrategyName = Literal["baseline", "flare", "hyde", "multi_query"]

HYDE_SYSTEM = (
    "Write a short hypothetical passage (120-220 words) that would appear in a reference article "
    "and help answer the user's question. Plain prose only — no markdown, no bullet lists, no citations. "
    "Use plausible domain vocabulary; do not invent specific statistics unless typical for the topic."
)

MULTI_QUERY_SYSTEM = (
    "Given a user question for a document library, output exactly 3 diverse search queries "
    "that would retrieve complementary passages. One query per line. No numbering, prefixes, or extra text."
)

RRF_K = 60


@dataclass(frozen=True)
class _RetrievalOutcome:
    filtered: list[dict]
    used_fallback: bool
    chunks_searched: int
    strategy: RetrievalStrategyName
    retrieval_passes: int
    flare_follow_up: bool


class RAGService:
    def __init__(
        self,
        embedding_service: ChromaEmbeddingService,
        ollama_client: OllamaClient,
        settings,
        *,
        content_library: LibraryId = "papers",
    ) -> None:
        self.embedding_service = embedding_service
        self.ollama_client = ollama_client
        self.settings = settings
        self._content_library: LibraryId = content_library

    def _system_prompts(self) -> dict[str, str]:
        return PUBLIC_SYSTEM_PROMPTS if self._content_library == "public" else SYSTEM_PROMPTS

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

    def _keyword_rerank_weight(self) -> float:
        if self._content_library == "public":
            return float(self.settings.PUBLIC_KEYWORD_RERANK_WEIGHT)
        return float(self.settings.KEYWORD_RERANK_WEIGHT)

    def _retrieve_k_budget(self, top_k: int, query_mode: str) -> int:
        if self._content_library == "public":
            cap = 96
            mult = 5
        else:
            cap = 64
            mult = 4
        if query_mode in ("compare", "general"):
            return min(cap, max(top_k * mult, 20))
        if query_mode in ("datasets", "reproduce", "methodology"):
            return min(min(cap, 72), max(top_k * (mult - 1), 16))
        return min(cap, max(top_k * mult, 20))

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
        w = self._keyword_rerank_weight()
        return sorted(
            results,
            key=lambda item: item["distance"] - (w * self._keyword_overlap_score(overlap_query, item["content"])),
        )

    def _pick_context_from_reranked(
        self, reranked: list[dict], overlap_query: str, query_mode: str, top_k: int
    ) -> tuple[list[dict], bool]:
        # Compare synthesis needs multiple papers; a single chunk can pass the strict cosine
        # gate while other relevant near-misses sit just above it — then context collapses to one doc.
        thr = (
            float(self.settings.PUBLIC_RELEVANCE_THRESHOLD)
            if self._content_library == "public"
            else float(self.settings.RELEVANCE_THRESHOLD)
        )
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
        w = self._keyword_rerank_weight()
        return sorted(
            merged,
            key=lambda item: item["distance"] - (w * self._keyword_overlap_score(overlap_query, item["content"])),
        )

    @staticmethod
    def _rrf_fuse_ranked_lists(ranked_lists: list[list[dict]], overlap_query: str, keyword_weight: float) -> list[dict]:
        """Reciprocal rank fusion across multiple retrieval lists (RAG-Fusion style)."""
        scores: dict[tuple, float] = {}
        best_item: dict[tuple, dict] = {}
        for rlist in ranked_lists:
            for rank, item in enumerate(rlist, start=1):
                key = RAGService._chunk_identity(item)
                scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
                cur = best_item.get(key)
                if cur is None or float(item["distance"]) < float(cur["distance"]):
                    best_item[key] = item
        merged = list(best_item.values())

        def sort_key(item: dict) -> tuple:
            key = RAGService._chunk_identity(item)
            kw = RAGService._keyword_overlap_score(overlap_query, item["content"])
            return (-scores.get(key, 0.0), float(item["distance"]) - keyword_weight * kw)

        return sorted(merged, key=sort_key)

    @staticmethod
    def _effective_retrieval_strategy(
        retrieval_strategy: str,
        *,
        use_flare: bool,
        flare_active_default: bool,
        query_mode: str,
    ) -> RetrievalStrategyName:
        if query_mode == "datasets":
            return "baseline"
        if retrieval_strategy != "baseline":
            return retrieval_strategy  # type: ignore[return-value]
        if use_flare or flare_active_default:
            return "flare"
        return "baseline"

    def _hyde_hypothetical_passage(self, user_query: str) -> str:
        messages = [
            {"role": "system", "content": HYDE_SYSTEM},
            {"role": "user", "content": f"User question:\n{user_query.strip()}\n\nHypothetical passage:"},
        ]
        return self.ollama_client.chat(messages, temperature=0.35).strip()

    def _multi_query_variants(self, user_query: str) -> list[str]:
        messages = [
            {"role": "system", "content": MULTI_QUERY_SYSTEM},
            {"role": "user", "content": user_query.strip()},
        ]
        raw = self.ollama_client.chat(messages, temperature=0.2).strip()
        variants = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        seen: set[str] = {user_query.strip().lower()}
        out = [user_query.strip()]
        for v in variants:
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
            if len(out) >= 4:
                break
        return out

    def _run_retrieval(
        self,
        query: str,
        top_k: int,
        query_mode: str,
        section_filter: str | None,
        strategy: RetrievalStrategyName,
    ) -> _RetrievalOutcome:
        overlap = query
        flare_follow_up = False
        passes = 1

        if strategy == "hyde":
            try:
                hypo = self._hyde_hypothetical_passage(query)
            except Exception as exc:
                logger.warning("HyDE hypothetical call failed; falling back to baseline: %s", exc)
                hypo = ""
            embed_q = hypo if hypo else query
            reranked = self._search_rerank(embed_q, overlap, top_k, query_mode, section_filter)
            chunks_searched = len(reranked)
            filtered, used_fallback = self._pick_context_from_reranked(reranked, overlap, query_mode, top_k)
            return _RetrievalOutcome(
                filtered=filtered,
                used_fallback=used_fallback,
                chunks_searched=chunks_searched,
                strategy="hyde" if hypo else "baseline",
                retrieval_passes=1,
                flare_follow_up=False,
            )

        if strategy == "multi_query":
            try:
                sub_queries = self._multi_query_variants(query)
            except Exception as exc:
                logger.warning("Multi-query expansion failed; falling back to baseline: %s", exc)
                sub_queries = [query]
            lists = [
                self._search_rerank(sq, overlap, top_k, query_mode, section_filter) for sq in sub_queries
            ]
            chunks_searched = sum(len(lst) for lst in lists)
            w = self._keyword_rerank_weight()
            reranked = self._rrf_fuse_ranked_lists(lists, overlap, w)
            passes = len(sub_queries)
            filtered, used_fallback = self._pick_context_from_reranked(reranked, overlap, query_mode, top_k)
            return _RetrievalOutcome(
                filtered=filtered,
                used_fallback=used_fallback,
                chunks_searched=chunks_searched,
                strategy="multi_query",
                retrieval_passes=passes,
                flare_follow_up=False,
            )

        rer1 = self._search_rerank(query, overlap, top_k, query_mode, section_filter)
        filtered, used_fallback = self._pick_context_from_reranked(rer1, overlap, query_mode, top_k)
        chunks_searched = len(rer1)

        if strategy == "flare" and query_mode != "datasets" and filtered:
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
                    rer2 = self._search_rerank(follow_q, overlap, top_k, query_mode, section_filter)
                    merged = self._merge_reranked_passes(rer1, rer2, overlap)
                    filtered, used_fallback = self._pick_context_from_reranked(merged, overlap, query_mode, top_k)
                    flare_follow_up = True
                    chunks_searched = len(rer1) + len(rer2)
                    passes = 2

        return _RetrievalOutcome(
            filtered=filtered,
            used_fallback=used_fallback,
            chunks_searched=chunks_searched,
            strategy=strategy,
            retrieval_passes=passes,
            flare_follow_up=flare_follow_up,
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

    @staticmethod
    def _sources_from_chunks(filtered: list[dict]) -> list[SourceCitation]:
        return [
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

    def _empty_answer_message(self) -> str:
        if self._content_library == "public":
            return (
                "I could not find relevant information in the indexed public corpus for this question. "
                "Bulk-index Wikipedia text (see scripts/bulk_index_public.py) or ingest .txt files with library=public."
            )
        return (
            "I could not find relevant information in your paper library for this question. "
            "Try uploading more papers or rephrasing your query."
        )

    @staticmethod
    def _confidence_from_chunks(filtered: list[dict]) -> float:
        confidence = round(1.0 - (sum(item["distance"] for item in filtered) / len(filtered)), 2)
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _temperature_for_mode(query_mode: str) -> float:
        if query_mode in ("general", "compare"):
            return 0.28
        if query_mode in ("methodology", "reproduce"):
            return 0.16
        return 0.1

    def _compose_chat_request(self, query: str, query_mode: str, filtered: list[dict]) -> tuple[list[dict[str, str]], float]:
        context_parts = []
        src_label = "Article" if self._content_library == "public" else "Paper"
        corpus = "encyclopedia articles" if self._content_library == "public" else "research papers"
        for i, item in enumerate(filtered):
            metadata = item["metadata"]
            context_parts.append(
                f"[Source {i + 1}] {src_label}: {metadata.get('title', 'Unknown')} | "
                f"Section: {metadata.get('section', 'body')} | "
                f"Page: {metadata.get('page_number', 0)}\n{item['content']}\n\n"
            )
        context = "".join(context_parts)
        system_prompt = self._system_prompts().get(query_mode, self._system_prompts()["general"])
        cite = "**Article title**" if self._content_library == "public" else "**Paper title**"
        user_message = (
            f"Context from {corpus}:\n\n{context}\n"
            f"Question:\n{query}\n\n"
            "Produce the full structured answer. Be thorough where the passages allow: multi-paragraph ### sections, "
            "nested bullets, and a rich comparison table when in compare mode. "
            f"Bold {cite} throughout. If a section has little evidence, keep it short and label the gap."
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        return messages, self._temperature_for_mode(query_mode)

    def _build_datasets_answer(self, filtered: list[dict]) -> str:
        extracted = self._extract_datasets_from_sources(filtered)
        if not extracted:
            return (
                "## Dataset inventory\n\n"
                "No named datasets or benchmarks were detected in the retrieved passages. "
                "Try a broader **Top K**, another mode, or add content whose passages mention benchmarks."
            )
        unique_datasets = {row[0] for row in extracted}
        unique_papers = {row[1] for row in extracted}
        scope = "articles" if self._content_library == "public" else "papers"
        row_unit = "article" if self._content_library == "public" else "paper"
        answer_lines = [
            "## Dataset inventory",
            f"*Library-scoped scan — **{len(unique_datasets)}** dataset labels across **{len(unique_papers)}** {scope} "
            f"({len(extracted)} mentions in retrieved chunks).*",
            "",
            "### At a glance",
            f"- **{len(unique_datasets)}** distinct dataset or benchmark names detected",
            f"- **{len(unique_papers)}** contributing {scope} in this answer",
            f"- **{len(extracted)}** total dataset–{row_unit} mention rows (sorted below)",
            "",
            "### Entries",
        ]
        for dataset_name, paper_title, usage in extracted:
            answer_lines.append(f"- **{dataset_name}** — **{paper_title}** — _{usage}_")
        return "\n".join(answer_lines)

    def _answer_response(
        self,
        *,
        answer: str,
        sources: list[SourceCitation],
        confidence: float,
        has_answer: bool,
        query: str,
        query_mode: str,
        chunks_searched: int,
        flare_on: bool,
        flare_follow_up: bool,
        strategy: str,
        retrieval_passes: int,
    ) -> AnswerResponse:
        return AnswerResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            has_answer=has_answer,
            query=query,
            query_mode=query_mode,
            model_used=self.settings.LLM_MODEL,
            chunks_searched=chunks_searched,
            flare_enabled=flare_on,
            flare_followup_retrieval=flare_follow_up,
            retrieval_strategy=strategy,
            retrieval_passes=retrieval_passes,
            library=self._content_library,
        )

    def answer_stream(
        self,
        query: str,
        top_k: int,
        query_mode: str = "general",
        section_filter: str | None = None,
        use_flare: bool = False,
        retrieval_strategy: str = "baseline",
        retrieve_only: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield SSE-friendly events: retrieval (sources first), token chunks, then done."""
        strategy = self._effective_retrieval_strategy(
            retrieval_strategy,
            use_flare=use_flare,
            flare_active_default=bool(self.settings.FLARE_ACTIVE_RETRIEVAL),
            query_mode=query_mode,
        )
        flare_on = strategy == "flare"
        outcome = self._run_retrieval(query, top_k, query_mode, section_filter, strategy)
        filtered = outcome.filtered
        used_fallback = outcome.used_fallback
        chunks_searched = outcome.chunks_searched
        flare_follow_up = outcome.flare_follow_up
        strategy = outcome.strategy
        retrieval_passes = outcome.retrieval_passes

        if not filtered:
            payload = self._answer_response(
                answer=self._empty_answer_message(),
                sources=[],
                confidence=0.0,
                has_answer=False,
                query=query,
                query_mode=query_mode,
                chunks_searched=chunks_searched,
                flare_on=flare_on,
                flare_follow_up=flare_follow_up,
                strategy=strategy,
                retrieval_passes=retrieval_passes,
            ).model_dump(mode="json")
            yield {"event": "done", "data": payload}
            return

        sources = self._sources_from_chunks(filtered)
        confidence = self._confidence_from_chunks(filtered)
        meta = {
            "sources": [s.model_dump(mode="json") for s in sources],
            "confidence": confidence,
            "has_answer": True,
            "query": query,
            "query_mode": query_mode,
            "model_used": self.settings.LLM_MODEL,
            "chunks_searched": chunks_searched,
            "flare_enabled": flare_on,
            "flare_followup_retrieval": flare_follow_up,
            "retrieval_strategy": strategy,
            "retrieval_passes": retrieval_passes,
            "library": self._content_library,
        }
        yield {"event": "retrieval", "data": meta}

        if retrieve_only and query_mode != "datasets":
            payload = self._answer_response(
                answer="",
                sources=sources,
                confidence=confidence,
                has_answer=True,
                query=query,
                query_mode=query_mode,
                chunks_searched=chunks_searched,
                flare_on=flare_on,
                flare_follow_up=flare_follow_up,
                strategy=strategy,
                retrieval_passes=retrieval_passes,
            ).model_dump(mode="json")
            yield {"event": "done", "data": payload}
            return

        if query_mode == "datasets":
            answer_text = self._build_datasets_answer(filtered)
        else:
            messages, temp = self._compose_chat_request(query, query_mode, filtered)
            parts: list[str] = []
            for piece in self.ollama_client.chat_stream(messages, temperature=temp):
                parts.append(piece)
                yield {"event": "token", "data": {"text": piece}}
            answer_text = "".join(parts)

        if used_fallback and query_mode != "datasets":
            answer_text = f"{answer_text}\n\n*Retrieval: using best-matching passages (strict distance threshold not met).*"

        payload = self._answer_response(
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            has_answer=True,
            query=query,
            query_mode=query_mode,
            chunks_searched=chunks_searched,
            flare_on=flare_on,
            flare_follow_up=flare_follow_up,
            strategy=strategy,
            retrieval_passes=retrieval_passes,
        ).model_dump(mode="json")
        yield {"event": "done", "data": payload}

    def answer(
        self,
        query: str,
        top_k: int,
        query_mode: str = "general",
        section_filter: str | None = None,
        use_flare: bool = False,
        retrieval_strategy: str = "baseline",
        retrieve_only: bool = False,
    ) -> AnswerResponse:
        """Retrieve with the selected strategy, then answer (or dataset inventory)."""
        strategy = self._effective_retrieval_strategy(
            retrieval_strategy,
            use_flare=use_flare,
            flare_active_default=bool(self.settings.FLARE_ACTIVE_RETRIEVAL),
            query_mode=query_mode,
        )
        flare_on = strategy == "flare"
        outcome = self._run_retrieval(query, top_k, query_mode, section_filter, strategy)
        filtered = outcome.filtered
        used_fallback = outcome.used_fallback
        chunks_searched = outcome.chunks_searched
        flare_follow_up = outcome.flare_follow_up
        strategy = outcome.strategy
        retrieval_passes = outcome.retrieval_passes

        if not filtered:
            return self._answer_response(
                answer=self._empty_answer_message(),
                sources=[],
                confidence=0.0,
                has_answer=False,
                query=query,
                query_mode=query_mode,
                chunks_searched=chunks_searched,
                flare_on=flare_on,
                flare_follow_up=flare_follow_up,
                strategy=strategy,
                retrieval_passes=retrieval_passes,
            )

        sources = self._sources_from_chunks(filtered)
        confidence = self._confidence_from_chunks(filtered)

        if retrieve_only and query_mode != "datasets":
            return self._answer_response(
                answer="",
                sources=sources,
                confidence=confidence,
                has_answer=True,
                query=query,
                query_mode=query_mode,
                chunks_searched=chunks_searched,
                flare_on=flare_on,
                flare_follow_up=flare_follow_up,
                strategy=strategy,
                retrieval_passes=retrieval_passes,
            )

        if query_mode == "datasets":
            answer_text = self._build_datasets_answer(filtered)
        else:
            messages, temp = self._compose_chat_request(query, query_mode, filtered)
            answer_text = self.ollama_client.chat(messages, temperature=temp)
        if used_fallback and query_mode != "datasets":
            answer_text = f"{answer_text}\n\n*Retrieval: using best-matching passages (strict distance threshold not met).*"

        return self._answer_response(
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            has_answer=True,
            query=query,
            query_mode=query_mode,
            chunks_searched=chunks_searched,
            flare_on=flare_on,
            flare_follow_up=flare_follow_up,
            strategy=strategy,
            retrieval_passes=retrieval_passes,
        )
