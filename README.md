# DocuMind

**Local-first RAG over your research paper library** — upload PDFs, DOCX, or TXT, pull papers from arXiv, and query across everything with modes built for how data scientists and ML engineers actually work: methodology dives, dataset inventory, cross-paper comparison, and reproducibility checklists. **No OpenAI, no Anthropic:** inference and embeddings run on **Ollama** (`llama3`, `nomic-embed-text`) on your machine.

I built DocuMind to demonstrate that I can ship an **end-to-end applied AI product** — not a notebook — with a real API surface, persistent retrieval, honest citations, and a UI people can click through.

---

## What I bring (and what this repo shows)

- **Product sense:** The API is organized around workflows researchers care about (modes, section filters, library management) rather than a single generic chat box.
- **Backend discipline:** FastAPI with **Pydantic v2** schemas everywhere, **pydantic-settings** for config, clear separation between ingestion, embedding, retrieval, and generation.
- **RAG engineering:** Chunking with **LangChain** splitters, section-aware metadata, vector search in **ChromaDB** (cosine, persistent store), reranking and diversity logic so answers are not dominated by one noisy chunk or one paper when you have a library.
- **Local LLM operations:** Ollama-only stack — good for **privacy**, **cost**, and **repeatable demos** without API keys.
- **Reliability:** PowerShell bootstrap and health scripts, graceful behavior when Ollama is down, request logging with **X-Request-ID** for tracing.
- **Frontend delivery:** **Next.js 15** dashboard (primary) plus **Streamlit** for fast iteration — both talk to the same API.
- **Quality bar:** **pytest** coverage for ingest, query paths, arXiv validation, and evaluation fixtures.

---

## The problem I’m solving

Research libraries grow fast: PDFs from arXiv, notes, half-read papers. Answering “which paper used which dataset?”, “how do these two methods differ?”, or “what would I need to reproduce this?” usually means re-skimming files or trusting memory. DocuMind is a **searchable, cited layer** on top of that library — still grounded in your documents, with explicit source cards.

---

## Architecture (how it fits together)

| Layer | What I implemented |
|--------|---------------------|
| **Ingestion** | File type + size validation; **PyPDF2** / **python-docx** / plain text; heuristic metadata (title, authors, year, arXiv id); LangChain **RecursiveCharacterTextSplitter** with section hints. |
| **Indexing** | **ChromaDB** persistent collection; **nomic-embed-text** via Ollama; metadata-rich chunks (doc id, page, section, chunk index). |
| **Retrieval** | Top-k semantic search, optional **section filter**, distance threshold plus **keyword overlap rerank** and **source deduplication / cross-paper diversity** so UI answers aren’t repetitive garbage. |
| **Generation** | Mode-specific system prompts; **llama3** via Ollama chat API; structured responses with **citations** and **confidence**; **datasets** mode uses structured extraction from retrieved text for cleaner, more reliable lists. |
| **API** | REST under `/api/v1`, OpenAPI at `/docs`, lifespan-managed singletons, `GET /health` for ops. |
| **UI** | Next.js app (`web/`) — dashboard, library, ingest/fetch flows; Streamlit (`frontend/app.py`) optional. |

---

## Feature set

- Ingest **`.pdf`**, **`.docx`**, **`.txt`**
- **Fetch from arXiv** by ID (PDF download + ingest)
- **Five query modes:** `general`, `compare`, `methodology`, `datasets`, `reproduce`
- **Section filter** (optional) for focused retrieval
- **Paper library:** list, detail, delete
- **Collection stats** (chunks, paper count)
- **Large starter corpus** in `data/sample_docs/` (**40+** text briefs: core ML, vision, NLP, graphs, time series, alignment, plus **institutional-style** summaries spanning risk, markets, compliance-adjacent ML, and ops — useful for stress-testing retrieval breadth). Bundled papers use stable ids `sample_*`. When `SAMPLE_CORPUS_VERSION` in settings changes, those rows are purged and re-indexed on API startup (requires Ollama).
- **Batch backfill:** `scripts/bulk_ingest_arxiv.py` + `data/arxiv_seed_list.txt` to grow the library toward **hundreds of papers** (respect arXiv rate limits; run against a live API with Ollama).

---

## Production operations & scale

Patterns here match what serious teams ship behind a real RAG product: **observability hooks**, **dependency-aware readiness**, **tight CORS**, optional **host allowlists**, and **container packaging**.

| Concern | What to use |
|--------|-------------|
| **Liveness** | `GET /health/live` — process up (orchestrator keep-alive). |
| **Readiness** | `GET /health/ready` — **200** when Ollama + Chroma are usable; **503** otherwise (Kubernetes / load balancer drain). |
| **Full status** | `GET /health` — models, collection stats, degraded vs ok. |
| **Tracing** | Every response includes **`X-Request-ID`**; logs include `request_id`. |
| **Hardening** | `.env`: `APP_ENV`, `LOG_LEVEL`, `TRUSTED_HOSTS`, `DISABLE_OPENAPI`, `CORS_ORIGINS` / `CORS_ALLOW_ALL`. |
| **Docker** | `docker compose up --build` — API on **8001**, persistent Chroma volume; set `OLLAMA_BASE_URL` to reach Ollama (defaults to `host.docker.internal` on Docker Desktop). |
| **Corpus growth** | Add `.txt` under `data/sample_docs/` and bump `SAMPLE_CORPUS_VERSION`, or bulk-ingest arXiv IDs. |

**Interview narrative:** In conversations about “production RAG,” anchor on **concrete** decisions: retrieval depth, chunking, evaluation, failure behavior when the LLM or index is down, auditability of citations, and how you refreshed the index. This repository is a **technical reference implementation** you can walk through line-by-line — it does not substitute for articulating *your* ownership and metrics at past employers.

---

## Tech stack

| Area | Choices |
|------|---------|
| API | **FastAPI**, **Uvicorn**, async where it matters |
| Data | **Pydantic v2**, **pydantic-settings** |
| Vectors | **ChromaDB** (persistent), cosine space |
| Chunking | **LangChain** text splitters |
| Models | **Ollama** — `llama3`, `nomic-embed-text` |
| HTTP client | **httpx** (arXiv), **requests** (Ollama sync calls in embedding path) |
| Web | **Next.js 15**, **React 18**, **TypeScript** |
| Alt UI | **Streamlit** |
| Tests | **pytest**, **pytest-asyncio** |
| Ops | **Docker** / **docker-compose**; PowerShell **start / stop / health / demo** scripts |

---

## Run it (Windows — recommended)

I standardize on **fixed ports** so the UI and API never argue: API **8001**, dashboard **3002**.

**First time (pulls models if needed):**

```powershell
.\start_documind.ps1
```

**Day-to-day (models already present):**

```powershell
.\start_documind.ps1 -SkipModelPull
```

**Stop:**

```powershell
.\stop_documind.ps1
```

**Sanity check (health + sample query):**

```powershell
.\demo_healthcheck.ps1
```

**Full stack smoke (boot + health + library + query preview):**

```powershell
.\interview_demo.ps1
```

**Prerequisites:** Python 3.11+, Node 18+, [Ollama](https://ollama.com), ~8GB+ RAM comfortable for `llama3`.

---

## Manual setup (cross-platform)

**Backend**

```bash
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3
ollama pull nomic-embed-text
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

- API: `http://127.0.0.1:8001`  
- Docs: `http://127.0.0.1:8001/docs`

**Next.js**

```bash
cd web
npm install
npm run dev -- -p 3002
```

Set `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8001` if the UI and API run on different hosts.

**Streamlit (optional)**

```bash
streamlit run frontend/app.py
```

**Next.js dev error `Cannot find module './NNN.js'`:** stop `next dev`, then from `web/` run `npm run clean` and restart. On Windows, stop the dev server before `npm install` if `@next/swc*` reports `EBUSY`. OneDrive-synced projects can stale `.next` — `npm run clean` fixes it.

---

## API (summary)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Ollama + collection stats |
| POST | `/api/v1/ingest` | Upload paper |
| DELETE | `/api/v1/ingest/{doc_id}` | Remove by ingest id |
| POST | `/api/v1/fetch-arxiv` | Fetch PDF by arXiv id |
| POST | `/api/v1/query` | RAG query (mode + optional section) |
| GET | `/api/v1/papers` | Library list |
| GET | `/api/v1/papers/{doc_id}` | One paper |
| DELETE | `/api/v1/papers/{doc_id}` | Delete paper |
| GET | `/api/v1/collection/stats` | Chunk / paper counts |

---

## Query modes (intent)

- **`general`** — Q&A with citations  
- **`compare`** — Cross-paper comparison framing  
- **`methodology`** — architectures, training, hyperparameters  
- **`datasets`** — datasets and how they’re used (structured extraction from retrieved context)  
- **`reproduce`** — reproducibility checklist style answers  

---

## Testing & evaluation

```bash
pytest -q
```

`evaluation/test_cases.json` holds scenario prompts + keywords for regression-style checks; `evaluation/test_rag_pipeline.py` guards fixture shape.

---

## Docker

```bash
docker-compose up --build
```

*(Host networking and Ollama are environment-specific — for a fully local demo, the PowerShell scripts are the most predictable on Windows.)*

---

## Configuration

See `.env.example`: Ollama URL, models, Chroma paths, chunk sizes, `TOP_K_RESULTS`, `RELEVANCE_THRESHOLD`, retrieval tuning (`ENABLE_FALLBACK_RETRIEVAL`, `FALLBACK_TOP_N`, `KEYWORD_RERANK_WEIGHT`), and upload limits.

---

## License / use

This repo is intended as a **portfolio-grade reference implementation**. Extend auth, multi-tenant namespaces, stronger PDF layout parsing, and formal RAG eval (precision/recall @k) as the next production steps — the current codebase is deliberately understandable and demo-safe.

I’m comfortable owning the full path from **problem framing → API design → retrieval tuning → UI → ops scripts**, which is what I want this project to communicate at a glance.
