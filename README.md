# DocuMind

DocuMind is a local-first RAG app over a research paper library. You ingest PDFs, Word files, or plain text, optionally pull PDFs from arXiv by ID, then query the whole library with modes that match how I actually work: quick Q&A, side-by-side method comparison, methodology extraction, a dataset pass, and a “what would it take to reproduce this?” style checklist.

Inference and embeddings run on **Ollama** (`llama3`, `nomic-embed-text`) on your machine. No OpenAI or Anthropic keys required for the default setup, which keeps demos cheap and keeps everything on disk you control.

I built this because I wanted something I could hand to someone and they could click through—not a notebook buried in a repo. There is a real FastAPI surface, a persistent vector index, source cards on answers, and a Next.js dashboard. Streamlit is there if I want to iterate faster on the same API.

---

## Why I bothered

My reading pile is always half PDFs and half “I’ll come back to this.” When I need to remember which paper used which benchmark, or how two approaches differ, I’m usually re-skimming or trusting memory. DocuMind is the layer I wanted on top of that: search grounded in what I actually indexed, with citations so I can sanity-check the model.

---

## How it’s put together

| Piece | What I used |
|-------|-------------|
| Ingest | Type and size checks; PyPDF2 / python-docx / raw text; rough metadata (title, authors, year, arXiv id when the text cooperates). |
| Chunking | LangChain `RecursiveCharacterTextSplitter` plus a light section label on each chunk from the first lines of text. |
| Index | ChromaDB, persistent on disk, cosine distance; `nomic-embed-text` through Ollama for vectors. |
| Retrieval | Top‑k search, optional section filter, a distance cutoff, keyword overlap rerank, and logic so one paper doesn’t eat the whole context window. |
| Answers | Mode-specific prompts to `llama3`; response includes sources and a simple confidence figure. Datasets mode pulls structured hints from retrieved text so lists aren’t entirely free‑form generation. |
| API | REST under `/api/v1`, OpenAPI at `/docs`, singletons over FastAPI lifespan. |
| UI | `web/` Next.js app is what I use day to day; `frontend/app.py` Streamlit hits the same endpoints. |

---

## What you can do with it

Ingest `.pdf`, `.docx`, `.txt`. Fetch from arXiv by ID. Query in five modes: `general`, `compare`, `methodology`, `datasets`, `reproduce`. Optional section filter. List papers, fetch one, delete. Collection stats endpoint.

Bundled text “papers” live under `data/sample_docs/`—40‑ish summaries spanning core ML topics plus a bunch of finance‑ and ops‑flavored briefs I added so retrieval isn’t trivial on a two‑document toy set. They index as `sample_*` doc ids. If you change `SAMPLE_CORPUS_VERSION` in settings, startup purges those sample rows and re‑ingests (Ollama has to be up).

To grow the library for real: `scripts/bulk_ingest_arxiv.py` with `data/arxiv_seed_list.txt` (I throttle between requests so I’m not hammering arXiv).

---

## Ops and “production-shaped” bits

I’m not claiming this is a bank deployment. I am claiming the boring parts are thought through enough that I wouldn’t be embarrassed wiring it behind a reverse proxy for a serious demo.

- `GET /health/live` — process is up.
- `GET /health/ready` — returns 200 when Ollama and Chroma are usable, 503 otherwise (same idea as Kubernetes readiness).
- `GET /health` — models, collection counts, degraded vs ok.
- Every response gets an `X-Request-ID`; logs include it.
- `.env` covers CORS allowlist (or open CORS for local only), optional trusted Host headers, optional `API_KEY` on `/api/v1` (send `X-API-Key`; leave blank for the Next.js dev UI), gzip on responses, JSON log lines if you want them, turning OpenAPI off in locked‑down environments. Chroma’s anonymized telemetry defaults off unless you opt in.
- Docker: `docker compose up --build`, API on 8001, Chroma in a named volume. Ollama is assumed on the host (`host.docker.internal` on Docker Desktop). On Windows I still reach for the PowerShell scripts first.

If I’m talking about production RAG in an interview, I point to specifics: how I chunk, how I handle bad retrieval scores, what happens when the model host is down, how citations are surfaced, how I’d reindex on a schedule. This repo is something I can walk file by file. It doesn’t replace talking about scope and outcomes from paid work.

---

## Upwork / portfolio

I keep a one‑pager for proposals:

- `portfolio/DocuMind_Upwork_Catalog.html` — open in a browser, print to PDF if I want to tweak layout.
- `portfolio/DocuMind_Upwork_Catalog.pdf` — regenerate with `pip install -r scripts/portfolio_requirements.txt` then `python scripts/generate_portfolio_pdf.py`.

---

## Stack (quick)

Python 3.11+, FastAPI, Pydantic v2, pydantic-settings, Uvicorn, ChromaDB, LangChain text splitters, Ollama, httpx, requests (Ollama calls from the embedding path), Next.js 15, React 18, TypeScript, pytest. Optional Streamlit.

---

## How I run it on Windows

I fixed ports so nothing fights: API **8001**, dashboard **3002**.

First boot (pulls models if they’re missing):

```powershell
.\start_documind.ps1
```

After that, when models already exist:

```powershell
.\start_documind.ps1 -SkipModelPull
```

Stop listeners on 3002 / 8001 / 11434:

```powershell
.\stop_documind.ps1
```

Health + sample query:

```powershell
.\demo_healthcheck.ps1
```

Heavier smoke:

```powershell
.\interview_demo.ps1
```

You’ll want Python 3.11+, Node 18+, Ollama installed, and enough RAM that `llama3` isn’t miserable (I treat ~8GB as a soft floor).

---

## Manual setup without the scripts

Backend:

```bash
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3
ollama pull nomic-embed-text
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

API: `http://127.0.0.1:8001` — docs at `/docs`.

Frontend:

```bash
cd web
npm install
npm run dev -- -p 3002
```

If the UI isn’t on the same machine, set `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8001` (or whatever host the API listens on).

Streamlit (optional):

```bash
streamlit run frontend/app.py
```

If Next throws `Cannot find module './NNN.js'`, I kill dev, run `npm run clean` from `web/`, and start again. On Windows, if `npm install` complains about `EBUSY` on `@next/swc*`, the dev server is still holding the file—stop it first. OneDrive can leave `.next` in a weird state; `npm run clean` usually fixes it.

---

## API routes I actually use

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | Full status |
| GET | `/health/live` | Liveness |
| GET | `/health/ready` | Readiness |
| POST | `/api/v1/ingest` | Multipart upload |
| DELETE | `/api/v1/ingest/{doc_id}` | 404 if nothing was indexed for that id |
| POST | `/api/v1/fetch-arxiv` | Body: `{ "arxiv_id": "..." }` |
| POST | `/api/v1/query` | Body: `query`, `top_k`, `query_mode`, optional `section_filter` |
| GET | `/api/v1/papers` | Library |
| GET | `/api/v1/papers/{doc_id}` | One row |
| DELETE | `/api/v1/papers/{doc_id}` | Same 404 behavior as ingest delete |
| GET | `/api/v1/collection/stats` | Counts |

---

## Query modes

`general` — straight Q&A with citations.  
`compare` — framed for comparing lines of work across papers.  
`methodology` — training, architecture, hyperparameters where the text supports it.  
`datasets` — dataset names and usage, with structured extraction from hits.  
`reproduce` — checklist-style “what you’d need to rerun this” grounded in context.

---

## Tests

```bash
pytest -q
```

`evaluation/test_cases.json` plus `evaluation/test_rag_pipeline.py` are there so I don’t accidentally break the shape of the eval fixtures.

---

## Docker

```bash
docker compose up --build
```

For a laptop-only loop on Windows, the PowerShell scripts are still what I trust most; Docker is for “same API in a container” demos and CI-ish runs.

---

## Config

Everything important is in `.env.example`: Ollama URL, model names, Chroma paths, chunk size and overlap, relevance threshold, retrieval fallback knobs, upload cap, CORS, optional API key, logging shape, gzip toggle, corpus version for the bundled samples.

---

## License / intent

I’m shipping this as a portfolio reference: readable, demo-safe, something I can extend with real auth, tenant isolation, and heavier PDF layout handling when a client actually needs it.

If you’re reading this because you’re hiring or contracting: I care about owning the path from problem to API to retrieval behavior to UI to how it runs in prod. That’s what DocuMind is here to show.
