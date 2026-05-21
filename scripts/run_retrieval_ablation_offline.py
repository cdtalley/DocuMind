#!/usr/bin/env python3
"""Offline retrieval ablation: real RAGService + ranking-fake Chroma + deterministic Ollama."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.services.rag_service import RAGService  # noqa: E402
from tests.ranking_fake_embedding import RankingFakeEmbeddingService  # noqa: E402

DEFAULT_BENCH = ROOT / "evaluation" / "retrieval_ablation.json"
ALL_STRATEGIES = ("baseline", "flare", "hyde", "multi_query")


def _percentile_ms(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] * (1.0 - (k - lo)) + xs[hi] * (k - lo)


def _chunk_keys(sources: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for s in sources:
        doc, ci = str(s.get("doc_id") or ""), s.get("chunk_index")
        if doc and ci is not None:
            keys.add(f"{doc}:{ci}")
    return keys


def _jaccard(a: set[str], b: set[str]) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _unique_doc_ids(sources: list[dict[str, Any]]) -> int:
    return len({str(s.get("doc_id") or "") for s in sources if s.get("doc_id")})


def _load_bench(path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data.get("library", "public")), data["cases"]


def seed_ablation_corpus(emb: RankingFakeEmbeddingService) -> None:
    """Multi-doc corpus tuned for ablation benchmark themes."""
    chunks = [
        (
            "wiki_ml",
            "Transformer models are evaluated on GLUE and SuperGLUE. Training uses AdamW; "
            "results report accuracy from 2018 through 2022 on benchmark splits.",
            "Machine Learning Benchmarks",
            "body",
        ),
        (
            "wiki_vision",
            "ImageNet classification and CIFAR-100 experiments compare CNN and ViT architectures. "
            "Populations and image counts appear in the results tables.",
            "Computer Vision Survey",
            "body",
        ),
        (
            "wiki_geo",
            "Geographic coverage includes Paris, France, the Rhine river, and regions of Central Europe. "
            "Cities such as Berlin and countries like Germany are discussed in historical context.",
            "European Geography Overview",
            "body",
        ),
        (
            "wiki_process",
            "The workflow proceeds in steps: collect data, preprocess features, train the model, "
            "validate on a holdout set, then deploy inference pipelines.",
            "ML Production Workflow",
            "body",
        ),
        (
            "wiki_conflict",
            "Some sources describe rapid urban growth while others emphasize conservation policy; "
            "viewpoints differ on the same metropolitan region.",
            "Urban Policy Debate",
            "body",
        ),
        (
            "wiki_numbers",
            "Reported values include 42.5 million population, distance 320 km, year 1991, and 12.4% growth rate.",
            "Demographic Statistics",
            "body",
        ),
    ]
    for i, (doc_id, content, title, section) in enumerate(chunks):
        emb.add_chunk(
            doc_id,
            content,
            title=title,
            section=section,
            chunk_index=0,
            authors="Corpus Seed",
            year="2024",
        )


class DeterministicOllama:
    def health_check(self) -> dict[str, Any]:
        return {"available": True, "models": ["llama3"]}

    def embed(self, text: str) -> list[float]:
        return [0.01] * 8

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.1) -> str:
        system = (messages[0].get("content") or "") if messages else ""
        user = (messages[-1].get("content") or "") if messages else ""
        if "hypothetical passage" in system.lower():
            return (
                "This reference article discusses transformer benchmarks on GLUE, ImageNet vision tasks, "
                "European cities and rivers, workflow steps for ML deployment, and demographic statistics "
                "with dates from 1991 to 2022."
            )
        if "3 diverse search queries" in system:
            return (
                "transformer GLUE SuperGLUE benchmark evaluation\n"
                "geographic cities countries Europe Paris Berlin\n"
                "numerical population statistics years dates workflow steps"
            )
        if "forward-looking preview" in user or "Write the forward-looking preview" in user:
            return "The excerpts mention benchmarks but ??? holdout protocol details are missing."
        return "Structured synthesis for stakeholder demo."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--strategies", default=",".join(ALL_STRATEGIES))
    ap.add_argument("--json-out", type=Path, default=ROOT / "evaluation" / "reports" / "ablation_results.json")
    ap.add_argument("--max-cases", type=int, default=0)
    ns = ap.parse_args()

    strategies = [s.strip() for s in ns.strategies.split(",") if s.strip()]
    library, cases = _load_bench(ns.bench)
    if ns.max_cases:
        cases = cases[: ns.max_cases]

    emb = RankingFakeEmbeddingService()
    seed_ablation_corpus(emb)
    ollama = DeterministicOllama()
    settings = get_settings()
    rag = RAGService(emb, ollama, settings, content_library=library)  # type: ignore[arg-type]

    rows: list[dict[str, Any]] = []
    keys_by_case_strategy: dict[tuple[str, str], set[str]] = {}

    for case in cases:
        case_id = case["id"]
        for strat in strategies:
            t0 = time.perf_counter()
            resp = rag.answer(
                query=case["query"],
                top_k=int(case.get("top_k", 10)),
                query_mode=case.get("query_mode", "general"),
                section_filter=case.get("section_filter"),
                retrieval_strategy=strat,
                retrieve_only=True,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            sources = [s.model_dump() for s in resp.sources]
            keys = _chunk_keys(sources)
            grounded = resp.has_answer and len(sources) > 0
            row = {
                "case_id": case_id,
                "strategy": strat,
                "grounded": grounded,
                "n_sources": len(sources),
                "unique_docs": _unique_doc_ids(sources),
                "confidence": resp.confidence,
                "chunks_searched": resp.chunks_searched,
                "retrieval_passes": resp.retrieval_passes,
                "flare_followup": resp.flare_followup_retrieval,
                "elapsed_ms": round(elapsed, 2),
                "query_mode": case.get("query_mode", "general"),
            }
            rows.append(row)
            keys_by_case_strategy[(case_id, strat)] = keys

    by_strategy: dict[str, dict[str, Any]] = {}
    for strat in strategies:
        sr = [r for r in rows if r["strategy"] == strat]
        lat = [float(r["elapsed_ms"]) for r in sr]
        n = len(sr) or 1
        by_strategy[strat] = {
            "grounded_rate_pct": 100.0 * sum(1 for r in sr if r["grounded"]) / n,
            "avg_sources": sum(r["n_sources"] for r in sr) / n,
            "avg_unique_docs": sum(r["unique_docs"] for r in sr) / n,
            "avg_confidence": sum(float(r["confidence"] or 0) for r in sr) / n,
            "avg_chunks_searched": sum(float(r["chunks_searched"] or 0) for r in sr) / n,
            "latency_ms_p50": round(_percentile_ms(lat, 50.0), 2),
            "latency_ms_p95": round(_percentile_ms(lat, 95.0), 2),
            "followup_rate_pct": 100.0 * sum(1 for r in sr if r.get("flare_followup")) / n,
        }

    pairwise: dict[str, float] = {}
    if "baseline" in strategies:
        for strat in strategies:
            if strat == "baseline":
                continue
            scores = []
            for case in cases:
                b = keys_by_case_strategy.get((case["id"], "baseline"))
                o = keys_by_case_strategy.get((case["id"], strat))
                if b is not None and o is not None:
                    scores.append(_jaccard(b, o))
            pairwise[strat] = sum(scores) / len(scores) if scores else 0.0

    payload = {
        "mode": "offline_ranking_fake",
        "bench": str(ns.bench.relative_to(ROOT)),
        "library": library,
        "cases": len(cases),
        "strategies": strategies,
        "corpus_docs": emb.collection_stats()["paper_count"],
        "corpus_chunks": emb.collection_stats()["total_chunks"],
        "by_strategy": by_strategy,
        "pairwise_jaccard_vs_baseline": pairwise,
        "rows": rows,
    }

    ns.json_out.parent.mkdir(parents=True, exist_ok=True)
    ns.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {ns.json_out}")
    for strat in strategies:
        s = by_strategy[strat]
        print(
            f"{strat:12} grounded={s['grounded_rate_pct']:.0f}% "
            f"avg_src={s['avg_sources']:.1f} p50={s['latency_ms_p50']:.0f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
