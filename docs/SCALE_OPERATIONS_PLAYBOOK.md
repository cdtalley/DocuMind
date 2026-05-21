# DocuMind — scale and operations

Internal runbook: capacity math, eval-driven tuning, and cloud migration stages. Maps to paths and scripts in this repo.

## 1. Vocabulary

| Term | What it means here |
|------|---------------------|
| **Document / article** | One logical unit: a PDF, a `.txt` shard, or one Wikipedia page written as a file. |
| **Chunk** | A segment produced by `DocumentChunker` (`CHUNK_SIZE` / `CHUNK_OVERLAP`). Stored as one row in Chroma with id `{doc_id}_{i}`. |
| **Vector** | One embedding vector attached to a chunk. “Millions of vectors” is normal industry phrasing. |
| **Collection** | Chroma namespace: **papers** vs **public** (`CHROMA_COLLECTION_NAME` / `CHROMA_COLLECTION_PUBLIC`). |
| **QPS** | Queries per second — user-visible *query* completions (often measured at the gateway after auth). |
| **RPS** | Requests per second — raw HTTP hits (health checks, retries, preflight can inflate RPS vs QPS). |
| **p95 / p99 latency** | Tail latency percentiles; production SLOs are usually written on p95 or p99, not averages. |

**Scale statement you can defend:** Ingestion supports **checkpointed bulk upserts** into a dedicated public collection, **resume** after failure, and capacity visibility via `GET /api/v1/libraries`.

**Bundled `data/sample_docs/`** is a small papers-only demo set — see [`data/sample_docs/README.md`](../data/sample_docs/README.md) and README §14 for counts vs a large public build.

## 2. Growing the public corpus (bulk path)

1. **Stream text to disk** (no embeddings yet):  
   `pip install datasets`  
   `python scripts/stream_wikipedia_to_txt.py --out-dir data/wiki_txt_build --max-articles 50000`  
   Streaming avoids loading Wikipedia into RAM.

2. **Bulk embed + Chroma write** (Ollama must be up):  
   `python scripts/bulk_index_public.py --txt-dir data/wiki_txt_build --checkpoint data/.bulk_public_checkpoint.json --workers 8`  
   Checkpoint JSON lists completed filenames so **Ctrl+C or Ollama restart does not lose progress**.

3. **One-shot chain** (optional):  
   `python scripts/build_public_corpus.py --articles 20000 --workers 8`

**Tuning `--workers`:** Ollama often serializes GPU work; more workers help when embedding is CPU-bound or when multiple Ollama instances sit behind a load balancer (out of scope for this repo).

## 3. Capacity and back-of-napkin math

- **Chunks ≈ total characters / (CHUNK_SIZE − overlap)** per document, with floor/ceiling from splitter behavior.  
- **Embed throughput** = your bottleneck. Measure: `bulk_index_public` logs `progress files=… chunks=… elapsed_s=…`.  
- **Disk:** Chroma stores vectors + full document text per chunk + HNSW graph. Order-of-magnitude: **multiple bytes per float × dims × chunks**, plus stored text. For millions of chunks, **plan TB-class** storage before you promise a number.

## 4. Query-path tuning (when answers feel empty or noisy)

| Symptom | Knob | Direction |
|---------|------|-----------|
| Misses relevant passages | `TOP_K_RESULTS`, retrieval budget in `rag_service` | Increase top_k / widen pre-rerank pool |
| Too much irrelevant context | `RELEVANCE_THRESHOLD` | Tighten (lower distance cutoff—see README for distance semantics) |
| Lexical mismatch | `KEYWORD_RERANK_WEIGHT` | Increase slightly |
| Long articles dominate | diversity + `top_k` | Already capped per doc in code; tighten or add reranker in a fork |

**Large-chunk embed stalls:** raise `OLLAMA_REQUEST_TIMEOUT_SEC` in `.env` (wired into `OllamaClient`).

## 5. What you operate in production (even if this repo is single-node)

- **Health:** `/health/live` vs `/health/ready` — liveness “process up”, readiness “can serve traffic”.  
- **Dual index:** `GET /api/v1/libraries` — capacity, split-brain checks, growth dashboards.  
- **Logs:** `request_id` on every response; `LOG_JSON=true` for aggregation.  
- **Corrupt vector store:** Chroma 1.x can throw Rust/SQLite errors on bad disks or version skew; development path **quarantines** the persist dir and forces **process restart** (see `app/main.py` and README §12). Production = **restore from backup** or **fail closed**, not silent delete.

## 6. Example delivery narrative (short)

Two corpora (public text + papers) behind one API with separate collections. Ingestion is checkpointed and observable; queries return citations. Typical implementation path: stream text to disk, `bulk_index_public` with resume, one shared Chroma client for SQLite safety, `GET /api/v1/libraries` for counts. Report **measured** chunks/hour from indexer logs for the slice you actually ran.

### Example measured run (this repo, one session)

- Streamed **1,500** `wikimedia/wikipedia` articles to `data/wiki_txt_build/` (streaming; large local dirs are gitignored).  
- `bulk_index_public.py --max-files 25 --workers 4` → **~1,123** vectors in `documind_wikipedia` in **~10 min** wall clock (`GET /api/v1/libraries`).

## 7. Scripts (pre-release or audit)

```bash
pytest -q
python scripts/report_corpus_scale.py --api-base http://127.0.0.1:8001
python scripts/bulk_index_public.py --txt-dir data/wiki_txt_build --dry-run
python scripts/run_wiki_public_bench.py --base-url http://127.0.0.1:8001 --snapshot-only
```

### 7.1 Eval and tuning (CI vs live bench)

**A — Regression cases (CI):** `tests/query_eval_cases.py` + `tests/test_rag_query_suite.py` — twenty `QueryEvalCase` rows covering empty index, section filters, compare + high `top_k`, FLARE, long query near cap, diversity (`min_sources=2`), unicode, `top_k=1`, etc. CI runs **real `RAGService`** with a ranking-aware fake embedding layer and a deterministic stub LLM (`has_answer`, source counts, answer substrings, `chunks_searched`, wall time).

**B — Live public probes:** `evaluation/wiki_public_bench.json` + `scripts/run_wiki_public_bench.py`. Outcomes depend on the indexed slice; use for `PUBLIC_RELEVANCE_THRESHOLD` / `PUBLIC_KEYWORD_RERANK_WEIGHT` iteration.

**Merge gate (retrieval changes):** CI case list + `run_query_eval.py --tier structural` on candidate builds; `--tier full` on a pinned corpus; wiki bench deltas reviewed for public `PUBLIC_*` knobs before merge.

## 8. “What we’d do next” at real enterprise scale

- Cross-encoder **rerank** after ANN retrieval.  
- **Hosted** vector DB + separate ingest workers (Kubernetes Jobs).  
- **Embedding cache** (content-hash → vector) to skip duplicate pages.  
- **Evaluation harness** (Ragas / custom JSONL) on frozen question sets per release.

---

## 9. Cloud production: migration ladder

Three long-lived concerns: **ingest/index**, **query path**, **eval and governance**. Below is a staged way to move this architecture to typical cloud primitives without changing the HTTP contracts upfront.

### 9.1 Non-negotiables before you call anything “prod”

| Area | What “real” looks like |
|------|-------------------------|
| **Identity** | OAuth2/OIDC or API keys in a **secret manager** (not `.env` on disk in the image). |
| **Network** | TLS everywhere; private subnets for vector DB and model endpoints; **egress** controlled. |
| **Data** | Encryption at rest on volumes; **backup + restore tested** (vector index + object store for raw docs). |
| **SLOs** | Written targets: e.g. p95 retrieve **under** 300 ms, p95 end-to-end **under** 8 s, availability 99.9% — with **error budget** policy. |
| **Observability** | **Trace IDs** across gateway → API → embed → vector → LLM; metrics (QPS, saturation, queue depth); logs in a central sink. |
| **Safety** | PII scan on ingest; **tenant isolation** in metadata filters; prompt-injection playbooks (rate limit + output policy). |
| **Release** | Canary or blue/green; **frozen eval set** gates promotion (see §8). |

If you skip the table above, you have a **hosted demo**, not production RAG.

### 9.2 Stage ladder (what you build in order)

Treat each stage as **shippable**: measurable SLOs, rollback, and a runbook before you add the next.

**Stage 0 — Artifact and contract freeze**  
- Pin **embedding model id + dimension** and **LLM id** in config; record them in your design doc. Any change = **re-embed or dual-write** plan.  
- Export a **corpus version** string (git SHA + index build id); attach to every chunk metadata for cache invalidation.

**Stage 1 — Stateless API on cloud compute**  
- Run the existing FastAPI container on **ECS Fargate**, **Cloud Run**, **AKS**, or a **single VM** behind a load balancer.  
- Move `CHROMA_PERSIST_DIR` to a **network-attached volume** (EBS, PD, Azure Disk) with **snapshots**; wire `/health/ready` to the LB **only after** Chroma + model dependency checks pass.  
- **Horizontal scale:** N identical API replicas **only if** the vector store and model tier are **networked shared services** (not local disk per replica).

**Stage 2 — Split “orchestration” from “inference”**  
- Replace colocated **Ollama** with **managed inference** (OpenAI / Azure OpenAI / Bedrock / Vertex) **or** self-hosted **vLLM / TGI** on GPU autoscaling groups behind an **internal** load balancer.  
- Implement **timeouts, retries with jitter, and circuit breakers** on every outbound call (you already have timeout knobs toward Ollama; production adds **bulkhead** limits per tenant).  
- **Cost control:** per-tenant token budgets, max `top_k`, max context chars enforced **server-side** (never trust the UI).

**Stage 3 — Ingest as a job system (not HTTP upload at scale)**  
- Raw files land in **object storage** (S3 / GCS / Azure Blob); an event triggers **workers** (SQS + ECS, Pub/Sub + Cloud Run Jobs, Azure Queue + Container Apps).  
- Workers run the same **chunk + embed** logic as `bulk_index_public.py` / `DocumentService`, writing to the vector tier in **batches** with idempotent keys (`doc_id`, `chunk_index`).  
- **Checkpointing** stays mandatory at TB scale; the pattern you have (JSON checkpoint file) becomes a **DynamoDB / Firestore / SQL** “ingest cursor” table with lease locks for multi-worker safety.

**Stage 4 — Vector tier that matches your SLA**  
- Embedded **Chroma on one volume** is valid for many **single-region** products up to team-sized corpora; for **HA, multi-region, or ops-heavy** teams, move to **managed vector** (Pinecone, Weaviate Cloud, Zilliz) or **pgvector / OpenSearch k-NN** with replication.  
- Preserve a thin **`VectorStore` interface** in code so DocuMind’s `RAGService` stays the orchestration brain while storage swaps.  
- **ANN parameters** (ef, M, replicas) and **distance metric** are part of your **capacity review**, not afterthoughts.

**Stage 5 — Read path under load (QPS is won here, not in Python loops)**  
- **Semantic cache:** Redis / Memcached keyed by `(tenant_id, hash(normalized_query), corpus_version)` with short TTL for hot repeated questions.  
- **Optional precomputed retrieval** for top navigational queries (not generic RAG, but real products mix both).  
- **Async generation** for long answers: return `202` + `job_id` + WebSocket/SSE for staff consoles; keep synchronous path only for strict SLAs you can meet.  
- **Rate limiting** at API gateway (Kong, AWS API Gateway, Envoy); **WAF** in front of public endpoints.

**Stage 6 — Multi-tenant and compliance**  
- Every Chroma/Pinecone row carries **`tenant_id`**; every query applies `where` / namespace filter **server-side** after authZ.  
- **Data residency:** index region pinned to contract; **cross-border** LLM calls documented and blocked if required.

### 9.3 Where cloud primitives usually land (mapping table)

| Concern | AWS-shaped | GCP-shaped | Azure-shaped |
|---------|------------|------------|--------------|
| API + LB | ALB + ECS Fargate / EKS | Cloud Load Balancing + Cloud Run / GKE | App Gateway + Container Apps / AKS |
| Secrets | Secrets Manager | Secret Manager | Key Vault |
| Object store | S3 | GCS | Blob Storage |
| Async work | SQS + Lambda/ECS | Pub/Sub + Cloud Run Jobs | Service Bus + Functions |
| GPU LLM | EKS + p4/p5, or Bedrock | GKE + A2/H100, or Vertex | AKS + NC-series, or Azure OpenAI |
| Metrics/logs/traces | CloudWatch + X-Ray | Cloud Monitoring + Cloud Trace | Monitor + App Insights |
| IaC | Terraform / CDK | Terraform / Deployment Manager | Bicep / Terraform |

You do **not** need all three clouds; one column done deeply beats three columns slideware.

### 9.4 How you work week-to-week (solo or small team)

1. **Week 1:** Container + LB + secrets + `/health/ready` gate + structured logs + **one** dashboard (QPS, p95, 5xx).  
2. **Week 2:** Move embeddings + chat to **one** managed vendor API; load-test **retrieve-only** vs **full RAG** to find the split latency.  
3. **Week 3:** Ingest path: S3 drop → worker → vector upsert; **replay** test (re-run job idempotent).  
4. **Week 4:** Eval JSONL in CI for **structure** (citations present, modes); nightly **human spot-check** bucket for quality.  
5. **Ongoing:** Error budget reviews; **index rebuild** runbooks; **embedding model migration** runbooks (the highest-risk recurring project in RAG).

### 9.5 What stays “serious” from this repo vs what you swap

| Keep (design / code ideas) | Swap or extend for cloud scale |
|----------------------------|----------------------------------|
| **RAGService** decomposition: retrieve → assemble context → mode prompts → optional second pass | Same flow; **swap** `ChromaEmbeddingService` for vendor SDK |
| **Library / collection** split and `GET /api/v1/libraries` observability | Same pattern; **collection per tenant** or namespace strategy |
| **Checkpointed bulk ingest** semantics | Same semantics; **cursor store** + distributed workers |
| **Tests that isolate LLM** (`conftest` overrides) | Add **contract tests** against recorded HTTP fixtures for vendor APIs |
| **Ollama** for dev | **Not** the long-term production backbone at high QPS unless you operate GPU like a platform team |

### 9.6 How you prove it (so it is not “tutorial BS”)

- Publish **numbers**: p50/p95 for retrieve and E2E on a **defined hardware skew**; QPS at **saturated** vs **comfortable** CPU.  
- Publish **failure drills**: kill vector pod → graceful degradation; revoke API key → safe error surface.  
- Publish **eval delta** between releases on a **frozen** question set.  
- Keep a **one-page architecture diagram** and a **one-page on-call** (who gets paged, what to check first: LB 5xx vs embed timeouts vs vector latency).

---

*Runbook paths reference this repository’s layout.*
