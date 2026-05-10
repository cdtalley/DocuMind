#!/usr/bin/env python3
"""
Materialize a large bundled text corpus under data/sample_docs/ for production-scale RAG demos.

Each file is a multi-section synthetic technical brief (not a real arXiv paper). Content is
deterministic from the index so re-runs are stable. Pair with SAMPLE_CORPUS_VERSION bump in
app/config.py so startup re-seeds sample_* documents.

Usage (from repo root):
  python scripts/generate_production_corpus.py
  python scripts/generate_production_corpus.py --count 500 --force

Requires: Python 3.11+ (stdlib only).
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent.parent / "data" / "sample_docs"

TOPICS = [
    "contrastive representation learning for tabular defaults",
    "graph neural operators on transaction hypergraphs",
    "diffusion priors for probabilistic forecasting of spreads",
    "instruction-tuned retrievers for policy clause grounding",
    "neural bandits under delayed reward in recommender systems",
    "self-supervised pretraining for multivariate clinical time series",
    "equivariant CNNs for cryo-EM particle picking",
    "large-context transformers with sparse attention for logs",
    "Bayesian optimization for hyperparameter budgets in deep nets",
    "causal forests for heterogeneous treatment effects in marketing",
    "streaming PCA for high-dimensional sensor drift detection",
    "multi-task learning for joint NER and relation extraction",
    "neural SDEs for limit order book simulation",
    "federated fine-tuning with LoRA under communication caps",
    "knowledge distillation from ensemble teachers for latency SLAs",
    "spectral normalization for GAN stability in synthetic tabular data",
    "active learning with batch diversity for defect inspection",
    "hierarchical VAEs for structured e-commerce demand",
    "Transformer-XL style memory for long document QA",
    "graph contrastive learning for molecular property regression",
    "robust losses under label noise in credit scoring",
    "neural ODEs for irregularly sampled ICU vitals",
    "retrieval-augmented code generation with static analysis filters",
    "quantile regression forests for tail risk in PFE",
    "Wasserstein barycenters for domain adaptation in NLP",
    "deep sets for permutation-invariant portfolio encodings",
    "neural architecture search with hardware-aware rewards",
    "meta-learning few-shot adaptation for cold-start users",
    "conformal prediction intervals for probabilistic load forecasting",
    "energy-based models for anomaly detection in telemetry",
]

DATASETS = [
    "MIMIC-III de-identified cohorts",
    "CIFAR-100 with long-tail splits",
    "MS MARCO passage ranking",
    "IEEE-CIS fraud tabular benchmark",
    "LibriSpeech 960h",
    "QM9 molecular graphs",
    "Elliptic Bitcoin transaction graph",
    "UCI Electricity load panel",
    "S&P 500 earnings call transcripts (synthetic subset)",
    "Open Images detection subset",
    "GLUE and SuperGLUE dev sets",
    "Common Crawl-derived C4 slice",
    "FINBEN proprietary stress tape (synthetic)",
    "NASA MODIS tiles (public)",
    "Waymo Open motion slices",
]

METRICS = [
    ("AUROC", 0.72, 0.94),
    ("F1 at 10% review budget", 0.41, 0.88),
    ("RMSE vs seasonal naive", 0.12, 0.45),
    ("calibrated ECE", 0.02, 0.11),
    ("mean reciprocal rank@20", 0.35, 0.71),
    ("BLEU on held-out targets", 18.2, 41.7),
    ("nDCG@10 on enterprise search logs", 0.44, 0.82),
]


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:60] or "topic"


def build_document(idx: int, rng: random.Random) -> str:
    topic = TOPICS[idx % len(TOPICS)]
    spin = TOPICS[(idx * 7 + 3) % len(TOPICS)]
    ds1, ds2 = rng.choice(DATASETS), rng.choice(DATASETS)
    mname, mlow, mhigh = METRICS[idx % len(METRICS)]
    mval = round(mlow + (mhigh - mlow) * rng.random(), 3)
    year = 2016 + (idx % 9)
    authors = f"Synthetic Authors Collective {idx % 200:03d}"
    title = f"Production Corpus Brief {idx:05d}: {topic.title()} — extensions toward {spin.split()[0]} systems"

    return f"""{title}
{authors}
{year}

Abstract
We present empirical results on {topic}, emphasizing reproducibility constraints, leakage-safe evaluation, and deployment considerations for regulated or high-stakes environments. We connect design choices to {spin} and report where claims are supported only by synthetic or anonymized corpora.

Methodology
Our pipeline combines standard baselines with a proposed module trained with AdamW, cosine decay, gradient clipping at one unit, and early stopping on a patience of seven epochs. Ablations remove auxiliary losses and measure degradation on held-out slices stratified by time or entity clusters. Where applicable we report confidence intervals via paired bootstrap over evaluation queries.

Datasets
Primary training and validation use {ds1}; secondary stress evaluation uses {ds2}. We enforce strict temporal splits for any time-ordered data and document any relaxed splits in appendix-style notes within this brief. Class imbalance is handled with stratified sampling and focal loss when base rates fall below five percent positives.

Experiments
We track {mname} as the headline metric (reported value {mval} on the primary slice). We additionally log latency p95, GPU memory residency, and failure cases where retrieval or OCR quality limits conclusions. Hyperparameters include batch sizes in {{32, 64, 128}}, learning rates in the interval [1e-5, 3e-4], and weight decay in [0.01, 0.1] unless otherwise noted for stability.

Results
The proposed approach improves {mname} versus the strongest baseline in seven of twelve configuration cells, with largest gains when auxiliary metadata is available. Gains shrink under covariate shift between train and validation vintages; we highlight segments where the model reverts to safe abstention. Error analysis shows confusion between near-duplicate entity strings and sensitivity to long-tail vocabulary introduced after the training cutoff.

Conclusion and limitations
We summarize practical takeaways for engineering teams integrating similar models behind retrieval layers: maintain provenance on chunks, monitor embedding drift weekly, and budget for periodic re-embedding after material vocabulary shift. This document is synthetic and intended for retrieval benchmarking only; it must not be cited as peer-reviewed literature.

Keywords: {_slug(topic)}, benchmarking, retrieval, evaluation, synthetic brief id-{idx:05d}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Write synthetic sample_docs/*.txt for large bundled corpus.")
    parser.add_argument("--count", type=int, default=400, help="Number of new files to create (default 400).")
    parser.add_argument(
        "--prefix",
        type=str,
        default="sample_corpus_p7",
        help="Filename prefix before _NNNNN.txt (default sample_corpus_p7).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files for this prefix.",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    for i in range(args.count):
        name = f"{args.prefix}_{i:05d}.txt"
        path = OUT / name
        if path.exists() and not args.force:
            continue
        rng = random.Random(20260207 + i * 1009)
        path.write_text(build_document(i, rng).strip() + "\n", encoding="utf-8")
        written += 1
    print(f"Wrote {written} files under {OUT} (prefix={args.prefix}_*.txt).")
    total = len(list(OUT.glob("*.txt")))
    print(f"Total .txt files in sample_docs: {total}")


if __name__ == "__main__":
    main()
