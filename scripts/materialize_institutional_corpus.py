#!/usr/bin/env python3
"""Write additional sample_docs/*.txt for a larger bundled corpus (run from repo root)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample_docs"

# Dense, citation-style summaries: finance / risk / NLP / ops — plausible in regulated enterprise R&D.
CORPUS: list[tuple[str, str]] = [
    (
        "market_microstructure_deep_limit_order_book.txt",
        """Deep Limit Order Book Representations for Short-Horizon Price Movement Prediction
Zhang, Zohren, Roberts (synthetic brief for retrieval benchmarking)
2021

Abstract
We study convolutional and attention architectures over normalized L2 order book tensors for predicting mid-price changes at the next tick, emphasizing leakage-safe splits by trading session and instrument.

Methodology
Architectures include deep CNNs over multi-level bid/ask stacks, dilated temporal convolutions, and Transformer encoders with relative time embeddings. Training uses cross-entropy and Huber losses on directional labels derived from future mid quotes.

Datasets
Experiments use FI-2010 and proprietary LOB snapshots anonymized to five price levels; evaluation reports accuracy, F1, and Matthews correlation on stratified time splits.

Results
Attention models modestly outperform CNNs when depth exceeds eight levels; calibration via temperature scaling improves probability quality for downstream risk limits.
""",
    ),
    (
        "nlp_sec_filings_financial_sentiment.txt",
        """FinBERT and Hierarchical Attention for Multi-Section SEC Filing Analysis
Internal NLP Research Note (summarized)
2020

Abstract
We fine-tune transformer encoders on 10-K and 10-Q sections to predict credit migration indicators and earnings surprises, with section-aware pooling to respect document structure.

Methodology
Models concatenate MD&A, Risk Factors, and Financial Statements sub-spans; we apply Longformer-style attention windows for documents exceeding 4096 subword tokens.

Datasets
Training draws from EDGAR 2008–2019 filings paired with quarterly outcomes; development uses time-based splits only.

Experiments
Metrics include AUROC for downgrade within four quarters and RMSE for EPS surprise; ablations show Risk Factors contribute disproportionately to tail-risk prediction.
""",
    ),
    (
        "portfolio_robust_estimation_high_dim.txt",
        """Robust Covariance Shrinkage and Graphical Lasso for High-Dimensional Portfolio Construction
Quant Research Methods Survey
2019

Abstract
We compare Ledoit-Wolf shrinkage, factor-model structured estimators, and sparse inverse covariance selection under non-stationary equity return panels relevant to global equity books.

Methodology
Stability selection on graphical lasso paths identifies sparse precision matrices; turnover penalties integrate with mean-variance optimizers under long-only and sector-neutral constraints.

Datasets
Experiments use MSCI World constituents and Russell 3000 subsets with monthly rebalancing; stress tests apply historical 2008 and 2020 windows.

Results
Shrinkage dominates sample covariance in out-of-sample Sharpe; sparse models reduce concentration in idiosyncratic names during correlation spikes.
""",
    ),
    (
        "federated_risk_models_privacy_aware.txt",
        """Federated Learning for Anti-Money Laundering Scores with Differential Privacy Guarantees
Privacy Engineering Working Paper (summarized)
2022

Abstract
We train graph-augmented tabular models across regional silos using federated averaging with gradient clipping and Gaussian noise, targeting epsilon-delta DP budgets compatible with internal audit.

Methodology
Each silo computes local updates on transaction graphs with temporal neighborhood features; secure aggregation reduces single-party visibility of raw gradients.

Datasets
Synthetic mixtures mirror SWIFT-like message features and Kaggle IEEE-CIS style tabular attributes; labels are highly imbalanced.

Results
Utility loss versus centralized training remains under five AUC points at epsilon=8; larger graphs benefit from knowledge distillation from a public teacher model.
""",
    ),
    (
        "streaming_cep_latency_compliance.txt",
        """Complex Event Processing for Real-Time Limit Breach Detection Under Sub-10ms Latency
Systems Architecture Note
2021

Abstract
We describe a CEP pipeline combining sliding windows, temporal operators, and compiled finite-state monitors for position and Greeks limits across equities and listed derivatives.

Methodology
Rules compile to deterministic automata; hot paths avoid GC pressure via pre-allocated ring buffers; back-pressure policies shed non-critical analytics under overload.

Datasets
Replay uses captured market data feeds and internal risk snapshots from 2019 stress week.

Results
p99 end-to-end latency stays below eight milliseconds on commodity hardware when co-located with matching engines; false positives drop with two-stage confirmation filters.
""",
    ),
    (
        "execution_shortfall_transformer_tca.txt",
        """Sequence-to-Sequence Models for Execution Shortfall Prediction in Algorithmic Trading
TCA Machine Learning Brief
2023

Abstract
We forecast implementation shortfall using encoder-decoder Transformers over order and market microstructure event sequences, conditioning on strategy type and urgency.

Methodology
Inputs include child order arrivals, partial fills, spread, and depth imbalance time series; training minimizes quantile loss for tail-aware scheduling.

Datasets
Labels come from historical TCA databases across US and European cash equities; train/validate splits are by quarter.

Results
Models improve pinball loss versus gradient boosting on sparse features; calibration helps desk-level budget setting.
""",
    ),
    (
        "model_risk_governance_llm_evaluation.txt",
        """Model Risk Management for Large Language Models in Document-QA Workflows
Governance Framework Draft (summarized)
2024

Abstract
We extend SR 11-7 style lifecycle controls to LLM-augmented retrieval systems: conceptual soundness, ongoing monitoring, and independent validation of grounding and refusal behavior.

Methodology
Checklists cover training data provenance, prompt injection tests, citation fidelity audits, and drift monitors on embedding spaces; red teams probe policy violations.

Datasets
Evaluation suites mix public retrieval benchmarks and synthetic policy documents with labeled gold answers.

Results
Retrieval-augmented setups reduce hallucination rates versus pure generation but require chunk-level provenance logging for audit replay.
""",
    ),
    (
        "causal_uplift_neural_networks_marketing.txt",
        """Neural Networks for Heterogeneous Treatment Effect Estimation in Client Campaigns
Applied Causal ML Note
2020

Abstract
We compare T-learner, X-learner, and DragonNet-style architectures for uplift modeling on digital marketing experiments with high-dimensional client features.

Methodology
Propensity models use gradient boosting; outcome heads are multi-layer perceptrons with targeted regularization; cross-fitting reduces bias in doubly robust scores.

Datasets
Semi-synthetic data built from CRM features and randomized holdouts; outcomes include conversion and subsequent engagement.

Results
DragonNet variants stabilize value estimates in small treatment arms; calibration plots guide budget allocation curves.
""",
    ),
    (
        "knowledge_graph_compliance_retrieval.txt",
        """Retrieval over Enterprise Knowledge Graphs for Regulatory Interpretation Assistants
Knowledge Engineering Brief
2023

Abstract
We combine graph neural networks with sparse BM25 and dense vector retrieval over policy nodes, preserving lineage edges for explainability in compliance Q&A.

Methodology
Subgraph sampling around seed entities feeds a message-passing encoder; retrieved nodes align with transformer rerankers for final answer composition.

Datasets
Graphs encode cross-references between policy clauses, interpretations, and control standards; evaluation measures answer correctness with human review.

Results
Hybrid retrieval outperforms pure vector search on multi-hop questions requiring citation chains.
""",
    ),
    (
        "interest_rate_curve_neural_sde.txt",
        """Neural Stochastic Differential Equations for Yield Curve Simulation and Scenario Generation
Rates Research Summary
2022

Abstract
We parameterize drift and diffusion of forward rates with neural SDEs calibrated to historical term structure dynamics and caps/floors implied vols.

Methodology
Training minimizes Wasserstein distance between simulated and historical increments; Euler-Maruyama discretization supports GPU batch simulation for PFE-style metrics.

Datasets
USD and EUR OIS curves 2010–2022; validation compares to Hull-White and affine benchmark models.

Results
Neural SDEs capture volatility smile migration better in stress windows but require careful regularization to avoid arbitrage in long horizons.
""",
    ),
    (
        "aml_graph_attention_transactions.txt",
        """Graph Attention Networks for Suspicious Subgraph Detection in Transaction Networks
Financial Crime Analytics Paper (summarized)
2021

Abstract
We detect anomalous communities in bipartite customer-merchant graphs using attention layers with temporal decay on edge timestamps.

Methodology
Mini-batch training samples ego-networks; class imbalance handled via focal loss; explanations aggregate attention weights on adjacent accounts.

Datasets
Labeled alerts from internal case management are paired with public graph benchmarks (Elliptic-style topology, anonymized features).

Results
GAT variants improve precision at top-decile review versus logistic features alone; false positives concentrate in seasonal merchant categories.
""",
    ),
    (
        "multimodal_earnings_calls_audio_text.txt",
        """Multimodal Transformers Fusing Earnings Call Audio Prosody with Transcript Text
Equity Research ML Note
2023

Abstract
We predict post-earnings drift using wav2vec-style encoders aligned to ASR transcripts with cross-modal contrastive pretraining.

Methodology
Audio segments align to sentence boundaries; fusion uses co-attention; training optimizes multi-task objectives on direction and volatility labels.

Datasets
S&P 500 historical calls 2015–2022 with price reactions; held-out sectors test generalization.

Results
Prosody features add incremental R-squared beyond text alone for guidance-heavy names; data governance restricts external sharing.
""",
    ),
    (
        "reinforcement_learning_smart_order_routing.txt",
        """Deep Reinforcement Learning for Smart Order Routing with Latency-Aware Rewards
Execution Research (summarized)
2022

Abstract
Agents choose venue sequences under stochastic fill models; rewards blend shortfall, fees, and SLA penalties with entropy bonuses for exploration.

Methodology
Proximal policy optimization on simulated environments calibrated from historical venue performance; safe policy updates constrain divergence from baseline routers.

Datasets
Venue-level fills and quote updates from US equities; simulation replays order book snapshots.

Results
Policies reduce median shortfall versus static rules in volatile opens; sim-to-real gaps addressed via domain randomization on latency.
""",
    ),
    (
        "bayesian_structural_time_series_macro_nowcast.txt",
        """Bayesian Structural Time Series for Macro Nowcasting with Mixed-Frequency Indicators
Econometrics Desk Note
2018

Abstract
We extend BSTS with stochastic volatility and dynamic regression on weekly and monthly predictors to nowcast GDP and inflation surprises.

Methodology
State-space MCMC and variational approximations trade accuracy for runtime; spike-and-slab priors perform feature selection on hundreds of series.

Datasets
FRED-MD style macro panels; evaluation uses real-time vintages to avoid lookahead.

Results
BSTS ensembles beat naive AR in RMSE during regime shifts; uncertainty bands improve scenario narrative quality.
""",
    ),
    (
        "tabular_contrastive_representations_credit.txt",
        """Contrastive Learning for Self-Supervised Tabular Representations in Credit Modeling
Retail Risk ML Brief
2023

Abstract
We learn embeddings for loan application tables using column-wise masking and Siamese objectives, then fine-tune for default and loss-given-default prediction.

Methodology
Encoder combines piecewise linear embeddings for numeric fields and entity embeddings for categoricals; negatives drawn within mini-batches across time cohorts.

Datasets
Prime and subprime mortgage vintages with regulatory stress labels; strict temporal splits.

Results
Self-supervised pretraining improves AUC in low-label regimes; fairness audits monitor disparities across protected proxy features.
""",
    ),
    (
        "layout_lm_document_understanding_filings.txt",
        """LayoutLMv3 for Structured Table Extraction from Annual Reports
Document AI Summary
2022

Abstract
We fine-tune multimodal transformers on PDF page images and token coordinates to extract balance sheet line items for fundamental databases.

Methodology
Detection heads predict cell spans; reading order modules reduce errors on multi-column layouts; human-in-the-loop corrects low-confidence extractions.

Datasets
Internal annotations on 10-K PDFs; evaluation uses tree-edit distance versus analyst gold tables.

Results
Model reduces manual keying effort by forty percent on dense financial tables; OCR noise remains dominant error mode.
""",
    ),
    (
        "uncertainty_deep_ensembles_market_risk.txt",
        """Deep Ensembles and MC Dropout for Value-at-Risk Backtesting Under Heavy Tails
Market Risk Methods
2020

Abstract
We compare frequentist VaR models with neural distributional regression heads producing predictive quantiles, using Bernoulli and independence tests on violation series.

Methodology
Ensembles of MLPs trained with heteroscedastic Gaussian outputs; alternative uses Student-t emission with learned degrees of freedom.

Datasets
Multi-asset portfolio P&L windows including COVID shock; rolling 250-day estimation.

Results
Deep ensembles reduce clustering of violations versus historical simulation in stress periods; regulators prefer transparent benchmark overlays.
""",
    ),
    (
        "llm_guardrails_red_teaming_financial_qa.txt",
        """Red-Teaming Retrieval-Augmented LLMs for Financial Question Answering
Safety & Controls Note
2024

Abstract
We systematize adversarial probes for prompt injection, policy leakage, and ungrounded advice in internal document QA assistants grounded on retrieval.

Methodology
Automated tests inject malicious chunks into synthetic corpora; human reviewers score harmfulness; mitigation layers include citation-required templates and tool allowlists.

Datasets
Mix of public finance FAQs and synthetic policy corpora with planted contradictions.

Results
Grounding cuts unverified claims by half but does not eliminate social engineering via instruction overrides; layered defenses required.
""",
    ),
    (
        "survival_analysis_churn_credit_lines.txt",
        """Discrete-Time Survival Models with Neural Hazards for Revolving Credit Attrition
Consumer Risk Paper (summarized)
2019

Abstract
We estimate monthly churn hazards using partial likelihood extensions with embedding layers for behavioral utilization sequences.

Methodology
Networks output logits for each discrete interval; time-varying covariates include macro indices; penalization discourages unstable hazards.

Datasets
Millions of anonymized accounts with multi-year horizons; censored observations handled correctly.

Results
Neural hazards improve integrated Brier score versus Cox with handcrafted splines; explainability uses time-local SHAP aggregates.
""",
    ),
    (
        "volatility_forecasting_neural_garch_hybrid.txt",
        """Hybrid Neural-GARCH Models for Intraday Volatility Forecasting
Volatility Research Brief
2021

Abstract
We combine GARCH structure on daily variance with neural networks modeling intraday seasonalities and news surprise embeddings.

Methodology
Two-stage estimation with backprop through volatility recursion approximations; regularization anchors parameters near econometric baselines.

Datasets
US equity futures and FX spot at five-minute bars; evaluation uses QLIKE and VaR error.

Results
Hybrids outperform pure neural nets on long horizons where mean reversion dominates; training stability benefits from warm-starting from GARCH fits.
""",
    ),
    (
        "graph_neural_counterparty_exposure.txt",
        """Message Passing Neural Networks for Counterparty Network Exposure and Contagion Stress
Credit Portfolio Analytics
2022

Abstract
We embed bilateral exposure graphs to estimate incremental default risk under correlated shocks, comparing GNN outputs to analytical netting approximations.

Methodology
Layers propagate nominal and collateral-adjusted exposures; node features include ratings and sector; global readouts feed scenario loss distributions.

Datasets
Simulated networks calibrated to public filings topology; stress parameters align with CCAR-style shocks.

Results
GNNs capture nonlinearities from cyclical connectivity missed by independent defaults; runtime scales linearly in edges with batching.
""",
    ),
    (
        "synthetic_data_generation_privacy_tabular.txt",
        """Generative Adversarial Networks for Privacy-Preserving Synthetic Tabular Banking Data
Synthetic Data Workshop Paper
2023

Abstract
We train CTGAN and diffusion-based tabular models under DP-SGD to release synthetic loan datasets preserving marginal and correlation structure for vendor collaboration.

Methodology
Evaluation uses propensity MMD, correlation error, and downstream model utility on synthetics-only training; membership inference attacks quantify privacy.

Datasets
Retail mortgage and card transaction features anonymized; rare category handling uses mode collapse guards.

Results
Diffusion models reduce correlation error versus GANs at moderate epsilon; utility gap remains for tail quantiles.
""",
    ),
    (
        "real_time_fraud_detection_streaming_ml.txt",
        """Streaming Machine Learning for Payment Fraud with Concept Drift Adaptation
Payments Risk Engineering
2022

Abstract
We deploy online logistic and tree models updated via incremental learning on Kafka streams, with drift detectors triggering full retrains.

Methodology
Features computed in Flink windows; model server supports shadow mode and canary releases; feedback latency monitored for label delay bias.

Datasets
Card-not-present transactions in EU and US; severe class imbalance.

Results
Online updates reduce false positives after merchant rule changes; governance requires immutable model versioning and audit logs.
""",
    ),
    (
        "credit_spread_prediction_sequence_models.txt",
        """Sequence Models for Corporate Bond Spread Changes Using Fundamental and Macro Sequences
Fixed Income ML Note
2020

Abstract
LSTMs and Transformers ingest quarterly fundamentals, rating actions, and curve factors to predict monthly spread changes by issuer.

Methodology
Entity embeddings combine with temporal encoders; training uses huber loss on cross-sectional panels with issuer fixed effects ablations.

Datasets
ICE BofA indices constituents 2005–2019; liquidity filters exclude distressed names below price thresholds.

Results
Transformers marginally beat LSTMs when fundamentals are dense; macro-only baselines weaker in flight-to-quality episodes.
""",
    ),
    (
        "operational_resilience_ml_systems_dr.txt",
        """Disaster Recovery and Active-Active Patterns for Low-Latency ML Inference Services
SRE / MLOps Playbook Excerpt
2023

Abstract
We document RTO/RPO targets, health probes, traffic shadowing, and embedding store replication for retrieval services supporting document QA.

Methodology
Kubernetes readiness gates on /health/ready; blue-green releases; chaos experiments on vector database partitions.

Datasets
Synthetic load tests reproduce peak query rates; failover drills measure staleness of embeddings.

Results
Active-active cuts failover time below one minute when object storage replication lags are bounded; cost tradeoffs favor regional pairs over triple-active.
""",
    ),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in CORPUS:
        path = OUT / name
        if path.exists():
            continue
        text = body.strip() + "\n"
        path.write_text(text, encoding="utf-8")
        print("wrote", path.relative_to(ROOT))
    print("done. Existing files were skipped; delete a file to regenerate.")


if __name__ == "__main__":
    main()
