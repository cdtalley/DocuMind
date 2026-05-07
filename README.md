# DocuMind — Data Science Research Paper Intelligence API

I built DocuMind to solve a real research workflow bottleneck: reading dozens of data science papers, then trying to remember which paper discussed a specific dataset, training setup, metric, or implementation detail.

DocuMind turns a personal paper library into a searchable research intelligence system. I can upload papers (PDF, DOCX, TXT), fetch directly from arXiv, and run specialized RAG queries such as:
- cross-paper methodology comparison,
- dataset extraction,
- reproducibility checklists,
- implementation-focused deep dives.

The entire stack runs locally with Ollama (`llama3` + `nomic-embed-text`), so there are zero paid LLM API costs and full data privacy.

## Why This Project Matters

- **Production-oriented architecture**: async FastAPI backend, persistent ChromaDB, strict schema validation with Pydantic v2, modular service layer.
- **Research-specific retrieval design**: chunk metadata includes section labels, page mapping, and source citations for explainability.
- **Portfolio-ready UX**: polished Next.js dashboard for live demos, plus Streamlit app for quick experimentation.
- **Operational robustness**: startup scripts, health checks, and graceful fallback behavior for reliable demos.

## Core Features

- Upload and ingest papers in `.pdf`, `.docx`, and `.txt`
- Fetch papers directly from arXiv by ID
- Local semantic retrieval with Chroma + Ollama embeddings
- Five expert query modes:
  - `general`
  - `compare`
  - `methodology`
  - `datasets`
  - `reproduce`
- Source-grounded answers with confidence scoring and citations
- Paper library management (list, inspect, delete)
- Health and collection telemetry for operations visibility

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic v2, pydantic-settings
- **RAG**: LangChain text splitters, ChromaDB, Ollama (`llama3`, `nomic-embed-text`)
- **Parsing**: PyPDF2, python-docx
- **External fetch**: httpx (async arXiv client)
- **Frontends**: Next.js (showcase dashboard), Streamlit (rapid demo UI)
- **Quality**: pytest, pytest-asyncio
- **Containerization**: Docker, docker-compose

## System Design Overview

1. **Ingestion Layer**
   - Validates file type and size
   - Extracts text + paper metadata (title/authors/year/arXiv ID heuristics)
   - Splits into semantically useful chunks with section detection

2. **Retrieval Layer**
   - Embeds chunks with local Ollama embeddings
   - Stores vectors and metadata in persistent Chroma
   - Supports top-k retrieval with optional section filtering

3. **Generation Layer**
   - Applies query-mode-specific system prompts
   - Sends context-grounded messages to local `llama3`
   - Returns structured answer object with source citations and confidence

4. **Experience Layer**
   - Next.js dashboard for portfolio-grade interactive demos
   - Streamlit UI for quick analyst workflows

## Preset Demo Paper Library

I included a starter set of data science / ML landmark paper summaries in `data/sample_docs/`.
On API startup, DocuMind auto-seeds them into Chroma if missing, so the system is demo-ready immediately.

Included starter docs:
- Transformer / attention paper summary
- XGBoost systems paper summary
- BERT pretraining paper summary

## Local Setup

### One-Command Reliable Boot (Recommended)

This is the default way I run demos so Ollama is always started first and the stack comes up in a stable order.

```powershell
.\start_documind.ps1
```

What this script does:
- Starts Ollama automatically if not already running
- Ensures required models exist (`llama3`, `nomic-embed-text`)
- Boots FastAPI on `http://127.0.0.1:8001`
- Boots Next.js on `http://localhost:3002`
- Sets frontend API target automatically to the backend above

Fast repeat boot:

```powershell
.\start_documind.ps1 -SkipModelPull
```

Stop everything:

```powershell
.\stop_documind.ps1
```

Pre-demo readiness check:

```powershell
.\demo_healthcheck.ps1
```

This validates frontend reachability, API health, Ollama availability, indexed paper count, and a real dataset-mode RAG query.

End-to-end demo run:

```powershell
.\interview_demo.ps1
```

This runs a complete workflow: reliable boot, health verification, indexed library check, and a grounded RAG answer preview.

### Prerequisites

- Python 3.11+
- Node.js 18+
- Ollama installed: [https://ollama.com](https://ollama.com)
- Recommended RAM: 8GB+ for smooth local inference

### 1) Backend

```bash
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3
ollama pull nomic-embed-text
uvicorn app.main:app --reload
```

API runs at `http://127.0.0.1:8001`  
Swagger docs: `http://127.0.0.1:8001/docs`

### 2) Next.js Showcase Frontend

```bash
cd web
npm install
npm run dev
```

Dashboard runs at `http://localhost:3002`

### 3) Streamlit Frontend (Optional)

```bash
streamlit run frontend/app.py
```

## API Surface

- `GET /health` — Ollama connectivity + collection stats
- `POST /api/v1/ingest` — upload paper file
- `DELETE /api/v1/ingest/{doc_id}` — delete ingested document
- `POST /api/v1/fetch-arxiv` — fetch and ingest paper from arXiv
- `POST /api/v1/query` — run RAG query in selected mode
- `GET /api/v1/papers` — list paper library
- `GET /api/v1/papers/{doc_id}` — get paper details
- `DELETE /api/v1/papers/{doc_id}` — delete paper
- `GET /api/v1/collection/stats` — chunk/paper counts

## Query Mode Intent

- `general`: standard question answering across your library
- `compare`: compare approaches across papers
- `methodology`: implementation details, architecture, training setup
- `datasets`: extract dataset mentions and usage
- `reproduce`: enumerate reproducibility requirements and blockers

## Testing

```bash
pytest -q
```

Current test suite validates ingest, query behavior, invalid arXiv handling, collection stats, and RAG evaluation fixture integrity.

## Docker

```bash
docker-compose up --build
```

## Project Highlights

DocuMind is an end-to-end AI engineering project that combines:
- API engineering with typed contracts and modular service boundaries
- retrieval system design for research-focused workflows
- local-first LLM operations for privacy and cost control
- frontend productization for interactive stakeholder demos
- testing and deployment readiness for production evolution
