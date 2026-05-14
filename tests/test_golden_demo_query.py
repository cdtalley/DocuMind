"""Contract: Gold showcase query must validate against QueryRequest (sync with web/app/page.tsx)."""

from __future__ import annotations

from app.models.request_models import QueryRequest

# Duplicated from SHOWCASE_SCENARIOS[0].query — update both when changing the demo prompt.
GOLD_DEMO_QUERY = """DEMO — Cross-paper benchmark audit (grounded synthesis only).

Retrieval anchors: GLUE SuperGLUE SQuAD ImageNet CIFAR MNIST Cora Citeseer PubMed LibriSpeech C4 Open Images IEEE-CIS MovieLens transformers vision CNN tabular gradient boosting graph convolution time series drift calibration ECE AdamW LoRA retrieval augmentation.

Using ONLY the retrieved excerpts, produce the full compare-mode outline: ## At a glance; ## Narrative overview; ## Comparison table as a GitHub-flavored markdown table with columns: Method / paradigm | Paper (exact title from excerpt) | Datasets or benchmarks named in text | Reported claim or metric if stated | Limitation or scope | Why a practitioner would care; ## Mechanism & objective contrast (subsections for losses/objectives and for data & evaluation); ## Trade-offs & decision guide.

Rules: No invented paper titles or datasets. If a name appears in a Keywords: line in the excerpt, you may treat it as in-scope. Where the library has no chunk for a theme, write explicitly that the excerpt set does not cover it — do not speculate."""


def test_gold_demo_query_validates_with_compare_24_and_flare() -> None:
    req = QueryRequest(
        query=GOLD_DEMO_QUERY,
        library="papers",
        top_k=24,
        query_mode="compare",
        use_flare=True,
    )
    assert len(req.query) <= 2000
    assert req.query_mode == "compare"
    assert req.top_k == 24
    assert req.use_flare is True
