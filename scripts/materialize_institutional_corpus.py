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
    # --- v6 expansion: additional ML + applied briefs (synthetic, for retrieval stress) ---
    (
        "protein_language_models_structure_prediction.txt",
        """Protein Language Models for Secondary Structure and Contact Prediction
Structure Bioinformatics Brief (synthetic)
2023

Abstract
We fine-tune large protein language models on residue sequences with auxiliary heads for DSSP-derived labels and distogram-based contact maps, comparing frozen-backbone adapters to full fine-tunes.

Methodology
Training uses crop windows of 512 residues with paired MSA dropout; losses combine cross-entropy for secondary structure and binary cross-entropy on top-L contact pairs.

Datasets
CASP14 targets and PDB chains filtered to 40 percent identity clusters; evaluation reports Q8 accuracy and long-range precision on L over 24.

Results
Adapter-only training reaches within one point Q8 of full fine-tune at four times lower GPU hours; calibration on rare amino acid contexts remains the main failure mode.
""",
    ),
    (
        "equivariant_neural_networks_3d_molecules.txt",
        """E(3)-Equivariant Networks for Molecular Property Regression
Geometric Deep Learning Note (synthetic)
2022

Abstract
We build tensor field networks that respect rotations and translations on 3D atomic coordinates, predicting energy and dipole moments from conformer ensembles.

Methodology
Message passing uses relative position vectors transformed through irreducible representations; pooling is invariant to global rigid motions.

Datasets
QM9 and GEOM-Drugs subsets; train and test splits are by molecular scaffold to reduce leakage.

Results
Equivariant models outperform MLPs on coordinates alone by large margins on energy MAE; inference cost scales with neighbor count K.
""",
    ),
    (
        "diffusion_models_text_conditional_image.txt",
        """Classifier-Free Guidance for Text-Conditional Image Diffusion at Scale
Generative Modeling Summary (synthetic)
2022

Abstract
We train latent diffusion models with cross-attention to CLIP text embeddings and study guidance scale tradeoffs between fidelity and diversity.

Methodology
U-Net backbones operate in a VAE latent space; training drops classifier labels for a portion of steps to enable guidance at sampling time.

Datasets
LAION subsets and internal captioned photo data; human ratings on a 1-5 alignment rubric.

Results
Guidance above seven sharpens text alignment but collapses diversity; dynamic thresholding stabilizes sampling at high guidance.
""",
    ),
    (
        "sparse_mixture_of_experts_language_scaling.txt",
        """Sparse Mixture-of-Experts Transformers for Compute-Efficient Language Model Scaling
Systems ML Brief (synthetic)
2021

Abstract
We route tokens to expert feed-forward blocks with load-balancing auxiliary losses and evaluate wall-clock throughput versus dense models at matched parameter counts.

Methodology
Top-2 routing with capacity factor limits; all-to-all communication patterns optimized on TPU slices; gradient clipping and expert dropout.

Datasets
C4 and Wikipedia-derived corpora; downstream GLUE and SuperGLUE fine-tunes from frozen checkpoints.

Results
MoE models achieve lower training FLOPs per token at fixed quality but increase serving complexity; expert imbalance remains sensitive to batch size.
""",
    ),
    (
        "test_time_compute_chain_of_thought.txt",
        """Test-Time Compute Scaling via Self-Consistency and Best-of-N Sampling
Reasoning Systems Note (synthetic)
2024

Abstract
We study repeated sampling with majority vote and verifier reranking on math word problems and code generation benchmarks.

Methodology
Temperature and nucleus sampling generate N candidates; a lightweight verifier scores partial correctness on unit tests for code tasks.

Datasets
GSM8K-style arithmetic sets and HumanEval-style function synthesis prompts.

Results
Accuracy grows sublinearly with N; verifier quality dominates naive self-consistency when distractors look plausible.
""",
    ),
    (
        "neural_operators_pde_surrogates.txt",
        """Fourier Neural Operators as Surrogates for Parametric PDE Families
Scientific ML Summary (synthetic)
2021

Abstract
We learn mappings from coefficient fields to solution fields for Darcy flow and Navier-Stokes snapshots with mesh-independent evaluation.

Methodology
Fourier layers in the latent grid with periodic padding; super-resolution on coarser grids at train time for robustness.

Datasets
PDE benchmark suites with varying viscosity and boundary conditions.

Results
FNOs generalize across resolutions not seen in training when boundary conditions stay in-distribution; out-of-distribution forcing breaks badly without physics priors.
""",
    ),
    (
        "hybrid_retrieval_bm25_dense_fusion.txt",
        """Hybrid BM25 and Dense Retrieval with Learned Fusion for Open-Domain QA
IR + RAG Brief (synthetic)
2020

Abstract
We combine lexical BM25 scores with dense retriever scores using a shallow ranker trained on click and relevance labels from enterprise search logs.

Methodology
Reciprocal rank fusion and a two-layer MLP reranker are compared; negatives mined from hard non-relevant documents.

Datasets
MS MARCO passage ranking and internal policy corpora with anonymized queries.

Results
Hybrid beats either channel alone on recall at 100; fusion weights drift when vocabulary shifts quarter to quarter without retraining.
""",
    ),
    (
        "kv_cache_optimization_long_context_inference.txt",
        """KV-Cache Quantization and Page-Based Attention for Long-Context Transformer Inference
Inference Optimization Note (synthetic)
2024

Abstract
We quantize past key-value tensors to int8 with per-head scales and implement paged attention to reduce fragmentation on variable-length batches.

Methodology
SmoothQuant-style calibration on representative prompts; microbenchmarks on A100 and H100 GPUs.

Datasets
Synthetic long prompts up to 128k tokens and real chat transcripts truncated to policy limits.

Results
Throughput improves up to two times at 32k context with under one point perplexity regression on held-out chats.
""",
    ),
    (
        "trade_surveillance_sequence_transformers.txt",
        """Sequence Transformers for Suspicious Order Pattern Detection in Equities Markets
Market Surveillance ML (synthetic)
2021

Abstract
We encode time-ordered order events with self-attention and contrast against hand-crafted rule alerts for escalation triage.

Methodology
Embeddings for order types, sizes, and resting time gaps; focal loss for rare positive alerts.

Datasets
Labeled alert outcomes from internal surveillance teams on US equities tape replays.

Results
Model reduces false escalations by a double-digit percent at fixed review budget; regulatory explainability requires attention rollout visualizations for auditors.
""",
    ),
    (
        "liquidity_coverage_ratio_forecasting_ml.txt",
        """Gradient Boosting for Liquidity Coverage Ratio Shortfall Forecasting
Treasury Risk Brief (synthetic)
2019

Abstract
We forecast end-of-month LCR components using cash inflows, HQLA balances, and wholesale funding spreads with monotonic constraints on selected features.

Methodology
LightGBM with quantile objectives for tail risk; SHAP for committee-ready explanations.

Datasets
Regional bank internal treasury time series 2015-2018 with stress overlays.

Results
Quantile models outperform point forecasts for worst-week liquidity; data quality on intraday sweeps dominates model choice.
""",
    ),
    (
        "esg_narrative_mining_annual_reports.txt",
        """Weakly Supervised ESG Theme Detection in Annual Report Narratives
Sustainable Finance NLP (synthetic)
2022

Abstract
We tag paragraphs with ESG themes using seed keywords expanded through embedding neighborhoods and human validation rounds.

Methodology
Hierarchical attention over sections; abstain class for non-applicable text.

Datasets
European and US annual reports 2010-2021 with sparse theme labels.

Results
Precision on climate-related paragraphs exceeds eighty percent after two annotation rounds; greenwashing language remains a blind spot without external data.
""",
    ),
    (
        "legal_clause_classification_transformers.txt",
        """Transformer Encoders for Material Adverse Change Clause Classification
Legal NLP Brief (synthetic)
2021

Abstract
We classify MAC clauses in merger agreements into standard vs highly negotiated language using span-level annotations from paralegal review.

Methodology
Longformer windows with sliding overlap; class imbalance handled with weighted loss.

Datasets
Hundreds of agreements under attorney-client redaction patterns.

Results
Macro-F1 improves over bag-of-words baselines; errors cluster on cross-border definitions referencing foreign statutes.
""",
    ),
    (
        "contrastive_learning_remote_sensing_change.txt",
        """Self-Supervised Contrastive Learning for Satellite Image Change Detection
Remote Sensing ML (synthetic)
2023

Abstract
We learn representations from bi-temporal image pairs without pixel labels, then fine-tune a light decoder for binary change masks.

Methodology
MoCo-style queues on patch crops; augmentations respect sensor noise statistics.

Datasets
Sentinel-2 tiles over urban and forest regions with OSM-derived weak labels for evaluation only.

Results
SSL pretrain cuts labeled pixel requirements roughly in half versus random init; cloud shadows still dominate false positives.
""",
    ),
    (
        "spectral_clustering_large_graph_embeddings.txt",
        """Spectral Embeddings and k-NN Graph Sparsification for Billion-Edge Clustering
Graph Algorithms Note (synthetic)
2018

Abstract
We approximate Laplacian eigenvectors via randomized range finders on sparsified k-NN graphs built from vector embeddings of entities.

Methodology
Landmark sampling for approximate nearest neighbors; normalized cuts on lower-dimensional embeddings.

Datasets
Social and transaction graphs with degree skew; evaluation uses held-out link prediction sanity checks.

Results
Runtime drops orders of magnitude versus exact spectral methods with modest community detection quality loss.
""",
    ),
    (
        "biocreative_ner_transformers_chemical.txt",
        """Transformer Taggers for Chemical Named Entity Recognition in Patents
Biomedical NLP Brief (synthetic)
2020

Abstract
We compare CRF heads vs span classifiers on BERT encoders for chemical entity spans in patent text.

Methodology
Subword alignment heuristics for span boundaries; stratified CV by technology class.

Datasets
BioCreative-style chemical mention corpora augmented with patent snippets.

Results
Span classifiers win on long compound names; inference latency doubles versus linear-chain CRF at same backbone.
""",
    ),
    (
        "speech_enhancement_complex_mask_network.txt",
        """Complex-Valued Mask Estimation Networks for Single-Channel Speech Enhancement
Speech Processing Note (synthetic)
2019

Abstract
We predict complex ideal ratio masks in the STFT domain with a convolutional encoder-decoder and SI-SNR loss.

Methodology
Mixed real-imaginary channel formulation; causal convolutions for streaming deployment.

Datasets
VoiceBank-DEMAND and internal telephony noise corpora.

Results
SI-SNR gains of roughly three dB over magnitude-only masks; musical noise artifacts persist under extreme SNR.
""",
    ),
    (
        "optimal_transport_domain_adaptation_tabular.txt",
        """Entropic Optimal Transport for Covariate Shift Correction in Credit Scoring
Domain Adaptation Brief (synthetic)
2021

Abstract
We align source and target feature distributions via entropic OT maps before training logistic and tree models on labeled source data only.

Methodology
Sinkhorn iterations with stabilization; small regularization to avoid degenerate couplings.

Datasets
Vintage credit applications with time-based target shift.

Results
AUC improves on the target quarter when shift is smooth; performance collapses when macro regimes jump outside support of the map.
""",
    ),
    (
        "instruction_backtranslation_low_resource_nmt.txt",
        """Back-Translation with Instruction-Tuned Models for Low-Resource Machine Translation
NMT Note (synthetic)
2023

Abstract
We generate synthetic parallel data from monolingual target text using a multilingual instruction-tuned model and filter pairs with round-trip consistency.

Methodology
Quality estimation with COMET-style scores; vocabulary overlap filters.

Datasets
Low-resource pairs from WMT tiny tracks.

Results
BLEU gains of several points over pivot translation; hallucinated entities remain a risk for domain-specific monolingual text.
""",
    ),
    (
        "document_layout_detection_transformers.txt",
        """Transformer Detectors for Reading Order and Layout Region Classification in Scanned Forms
Document AI Brief (synthetic)
2022

Abstract
We detect text blocks, tables, and checkboxes on scanned government forms with a DETR-style detector and resolve reading order with a lightweight solver.

Methodology
Synthetic distortions for augmentation; bipartite matching loss at train time.

Datasets
Thousands of form images with polygon annotations under license.

Results
mAP improves over classical layout pipelines on skewed scans; handwritten fields remain out of distribution.
""",
    ),
    (
        "calibration_temperature_platt_scaling_credit.txt",
        """Post-Hoc Calibration of Tree Ensembles for Probability of Default Scores
Risk Modeling Note (synthetic)
2018

Abstract
We apply temperature scaling and Platt scaling on out-of-fold predictions from gradient boosted trees for PD models subject to monotonicity constraints.

Methodology
Isotonic regression where constraints allow; Brier score and reliability diagrams for monitoring.

Datasets
Retail mortgage vintages with multi-year outcomes.

Results
Expected calibration error drops materially on the test horizon; recalibration needed after macro shocks even when AUC is stable.
""",
    ),
    (
        "active_learning_batch_selection_deep_models.txt",
        """Batch Mode Active Learning with Diversity for Deep Image Classifiers
Active Learning Brief (synthetic)
2019

Abstract
We select labeling batches using uncertainty scores tempered with k-means++ diversity in embedding space to avoid redundant images.

Methodology
Core-set approximations compared to BADGE embeddings from penultimate layer features.

Datasets
CIFAR-100 and internal defect inspection imagery.

Results
Label budget to reach target accuracy drops twenty to thirty percent versus naive uncertainty sampling; batch size interacts strongly with diversity gains.
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
