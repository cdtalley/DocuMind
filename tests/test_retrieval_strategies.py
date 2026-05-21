"""Unit tests for retrieval strategy selection and RRF fusion (no Ollama)."""

from app.services.rag_service import RAGService, flare_triggers_follow_up


def _item(doc_id: str, chunk_index: int, distance: float) -> dict:
    return {
        "content": f"chunk {doc_id}-{chunk_index} about transformers and datasets",
        "distance": distance,
        "metadata": {"doc_id": doc_id, "chunk_index": chunk_index, "title": doc_id},
    }


def test_effective_strategy_legacy_use_flare() -> None:
    assert (
        RAGService._effective_retrieval_strategy(
            "baseline", use_flare=True, flare_active_default=False, query_mode="general"
        )
        == "flare"
    )


def test_effective_strategy_explicit_hyde_overrides_flare_flag() -> None:
    assert (
        RAGService._effective_retrieval_strategy(
            "hyde", use_flare=True, flare_active_default=True, query_mode="general"
        )
        == "hyde"
    )


def test_datasets_mode_forces_baseline() -> None:
    assert (
        RAGService._effective_retrieval_strategy(
            "flare", use_flare=True, flare_active_default=True, query_mode="datasets"
        )
        == "baseline"
    )


def test_rrf_promotes_chunk_seen_in_multiple_lists() -> None:
    list_a = [_item("a", 0, 0.2), _item("b", 0, 0.5)]
    list_b = [_item("a", 0, 0.25), _item("c", 0, 0.4)]
    fused = RAGService._rrf_fuse_ranked_lists([list_a, list_b], "transformers datasets", 0.0)
    assert fused[0]["metadata"]["doc_id"] == "a"


def test_flare_triggers_unchanged() -> None:
    assert flare_triggers_follow_up("The rate is ??? in the excerpt.") is True
    assert flare_triggers_follow_up("Clear answer from the passage.") is False
