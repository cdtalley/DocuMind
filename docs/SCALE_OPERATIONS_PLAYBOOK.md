# DocuMind at scale — operations and interview playbook

Use this doc when you need to **sound like you have run and maintained** large retrieval systems. Everything below maps to **real code and scripts in this repository**; adapt numbers to what you have actually run on your machine.

## 1. Vocabulary (say it precisely in interviews)

| Term | What it means here |
|------|---------------------|
| **Document / article** | One logical unit: a PDF, a `.txt` shard, or one Wikipedia page written as a file. |
| **Chunk** | A segment produced by `DocumentChunker` (`CHUNK_SIZE` / `CHUNK_OVERLAP`). Stored as one row in Chroma with id `{doc_id}_{i}`. |
| **Vector** | One embedding vector attached to a chunk. “Millions of vectors” is normal industry phrasing. |
| **Collection** | Chroma namespace: **papers** vs **public** (`CHROMA_COLLECTION_NAME` / `CHROMA_COLLECTION_PUBLIC`). |

**Honest scale claim:** “We designed the ingestion path for **checkpointed bulk upserts** into a dedicated public collection, with **resume** after failure, and **operator visibility** via `GET /api/v1/libraries`.” That is true regardless of whether you have 10k or 10M chunks today.

## 2. How you grow the public corpus (the real pipeline)

1. **Stream text to disk** (no embeddings yet):  
   `pip install datasets`  
   `python scripts/stream_wikipedia_to_txt.py --out-dir data/wiki_txt_build --max-articles 50000`  
   Streaming avoids loading Wikipedia into RAM.

2. **Bulk embed + Chroma write** (Ollama must be up):  
   `python scripts/bulk_index_public.py --txt-dir data/wiki_txt_build --checkpoint data/.bulk_public_checkpoint.json --workers 8`  
   Checkpoint JSON lists completed filenames so **Ctrl+C or Ollama restart does not lose progress**.

3. **One-shot chain** (optional):  
   `python scripts/build_public_corpus.py --articles 20000 --workers 8`

**Tuning `--workers`:** Ollama often serializes GPU work; raising workers helps when embedding is CPU-bound or when you run multiple Ollama replicas behind a load balancer (not in this repo—your interview story for “how we’d scale horizontally”).

## 3. Capacity and back-of-napkin math (credible, not magic)

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

## 6. Interview STAR snippets (copy ideas, not lies)

- **S — Situation:** “Two corpora—encyclopedia-scale public text and research PDFs—with one API and separate collections.”  
- **T — Task:** “Ingestion had to be resumable and observable; queries had to stay grounded with citations.”  
- **A — Action:** “Streaming HF Wikipedia to `.txt`, checkpointed `bulk_index_public`, shared single `PersistentClient` for SQLite safety, libraries endpoint for ops.”  
- **R — Result:** “Measured chunks/hour from indexer logs; tuned workers and timeouts; tracked public vs papers growth in the dashboard.”

**If you have not run millions of chunks:** Say you **validated the pipeline architecture** for that scale (streaming, checkpointing, batch upsert, timeouts) and would **load-test** on a larger slice before production cutover— that is senior-level honesty.

### Example measured run (this repo, one session)

- Streamed **1,500** `wikimedia/wikipedia` articles to `data/wiki_txt_build/` (streaming; directory may contain **100k+** `.txt` files after repeated operator runs—**all gitignored**).  
- `bulk_index_public.py --max-files 25 --workers 4` → **~1,123** vectors in `documind_wikipedia` in **~10 min** wall clock (`GET /api/v1/libraries`).

## 7. Scripts you can run before an interview loop

```bash
pytest -q
python scripts/report_corpus_scale.py --api-base http://127.0.0.1:8001
python scripts/bulk_index_public.py --txt-dir data/wiki_txt_build --dry-run
```

## 8. “What we’d do next” at real enterprise scale

- Cross-encoder **rerank** after ANN retrieval.  
- **Hosted** vector DB + separate ingest workers (Kubernetes Jobs).  
- **Embedding cache** (content-hash → vector) to skip duplicate pages.  
- **Evaluation harness** (Ragas / custom JSONL) on frozen question sets per release.

---

*This playbook is intentionally tied to DocuMind’s layout so you can open the repo in an interview and walk the files.*
