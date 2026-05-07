# DocuMind — Data Science Research Paper Intelligence

Ask questions across your entire data science paper library — compare methodologies, find datasets, and extract reproducibility details — powered entirely by local Ollama with zero API costs.

## Prerequisites
- Python 3.11+
- Ollama installed and running ([https://ollama.com](https://ollama.com))
- At least 8GB RAM recommended for `llama3`
- Node.js 18+ (for the Next.js showcase frontend)

## Quick Start (Backend + Streamlit)
```bash
pip install -r requirements.txt
ollama pull llama3
ollama pull nomic-embed-text
cp .env.example .env
uvicorn app.main:app --reload
# in second terminal
streamlit run frontend/app.py
```

## Quick Start (Next.js Showcase Frontend)
```bash
cd web
npm install
npm run dev
```

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Ollama status + collection stats |
| POST | /api/v1/ingest | Upload PDF/DOCX/TXT paper |
| DELETE | /api/v1/ingest/{doc_id} | Remove paper from library |
| POST | /api/v1/fetch-arxiv | Fetch paper by arXiv ID |
| POST | /api/v1/query | Ask a question (5 query modes) |
| GET | /api/v1/papers | List all papers in library |
| GET | /api/v1/papers/{doc_id} | Get one paper |
| DELETE | /api/v1/papers/{doc_id} | Delete paper |
| GET | /api/v1/collection/stats | Chunk and paper counts |

## Query Modes
| Mode | Use Case |
|------|----------|
| general | Standard Q&A over papers |
| compare | Compare methods across papers |
| methodology | Implementation and architecture details |
| datasets | Extract all datasets mentioned |
| reproduce | What you need to reproduce results |

## Architecture
- **Ingestion**: PDF/DOCX/TXT parsing via `PyPDF2` and `python-docx`
- **Chunking**: LangChain `RecursiveCharacterTextSplitter` with section detection
- **Embeddings**: local Ollama `nomic-embed-text`
- **Vector Store**: persistent local ChromaDB (`./chroma_db`)
- **Generation**: local Ollama `llama3` with mode-specific prompts
- **Frontend**: Streamlit demo + polished Next.js app in `web/`

## Preset Demo Library
On startup, sample research docs in `data/sample_docs/` are auto-seeded into Chroma if missing. This gives you instant demo content for interviews.

## Testing
```bash
pytest -q
```
