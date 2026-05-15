"""Contract: default operator showcase query validates against QueryRequest (sync with web/app/page.tsx)."""

from __future__ import annotations

from app.models.request_models import QueryRequest

# Duplicated from SHOWCASE_SCENARIOS[0].query — update both when changing the baseline prompt.
BASELINE_OPERATOR_QUERY = """Using ONLY the retrieved encyclopedia-style passages, write a concise answer for a general reader.

Rules:
- Every non-trivial claim must be traceable to a cited **Article title** from the context.
- If the passages disagree or omit a subtopic, say so explicitly — do not invent facts.
- End with a short "Coverage" line: what themes the excerpts did and did not support."""


def test_baseline_operator_query_validates_public_general() -> None:
    req = QueryRequest(
        query=BASELINE_OPERATOR_QUERY,
        library="public",
        top_k=10,
        query_mode="general",
        section_filter=None,
        use_flare=False,
    )
    assert len(req.query) <= 2000
    assert req.query_mode == "general"
    assert req.top_k == 10
    assert req.library == "public"
    assert req.section_filter is None
    assert req.use_flare is False
