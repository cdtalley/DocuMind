#!/usr/bin/env python3
"""Build portfolio/DocuMind_Upwork_Catalog.pdf (run from repo root after: pip install -r scripts/portfolio_requirements.txt)."""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "portfolio" / "DocuMind_Upwork_Catalog.pdf"


class PDF(FPDF):
    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def section(pdf: PDF, title: str) -> None:
    pdf.ln(4)
    w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(w, 7, title)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)


def body(pdf: PDF, text: str) -> None:
    w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(w, 5, text)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = PDF()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(37, 99, 235)
    pdf.multi_cell(w, 9, "DocuMind")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(w, 5, "Local-first RAG for research libraries - portfolio and Upwork catalog brief")
    pdf.ln(6)

    body(
        pdf,
        "Use this PDF on Upwork as: (1) a Project Catalog attachment, (2) a Portfolio item description "
        "supplement, or (3) a fixed-price scope handout. Replace bracketed fields with your rates and links.",
    )

    section(pdf, "One-line pitch")
    body(
        pdf,
        "End-to-end retrieval-augmented Q&A: ingest PDFs/DOCX/TXT and arXiv, chunk with section metadata, "
        "embed in ChromaDB, retrieve with rerank and diversity, answer with mode-specific prompts and "
        "citations - FastAPI backend + Next.js dashboard + Ollama (no cloud LLM keys required for demos).",
    )

    section(pdf, "Buyer outcomes")
    body(
        pdf,
        "- Grounded answers with source cards (paper title, section, chunk, distance).\n"
        "- Modes: general Q&A, compare methods, methodology extraction, dataset inventory, reproducibility checklist.\n"
        "- Operations: liveness and readiness HTTP probes, request IDs, optional API key on /api/v1, CORS allowlist, "
        "gzip responses, security headers, structured JSON logs, Docker volume for vectors.\n"
        "- Honest deletes (404 when no vectors removed) and validated section filters (422 on bad values).",
    )

    section(pdf, "Tech stack (keywords for search)")
    body(
        pdf,
        "Python 3.11+, FastAPI, Pydantic v2, Uvicorn, ChromaDB, LangChain text splitters, Ollama (llama3, "
        "nomic-embed-text), httpx, Next.js 15, React 18, TypeScript, react-markdown, pytest, Docker.",
    )

    section(pdf, "Suggested Upwork catalog titles")
    body(
        pdf,
        "A) 'Ship a local RAG document assistant (FastAPI + Chroma + Next.js)'\n"
        "B) 'Retrieval QA over your PDFs with citations, modes, and health checks'\n"
        "C) 'MVP: ingest + vector index + grounded answers - Ollama or swap to OpenAI'",
    )

    section(pdf, "Pricing guide (US-oriented anchors - edit for your tier)")
    body(
        pdf,
        "Hourly (integration + architecture): [ $85-$150+ / hr ] depending on profile and reviews.\n"
        "Fixed - discovery + written architecture (1-2 weeks): [ $2,500-$6,000 ].\n"
        "Fixed - MVP RAG (ingest, index, UI, deploy to one cloud, basic eval): [ $5,000-$12,000 ].\n"
        "Fixed - hardening pass (auth, rate limits, multi-tenant ACLs, formal eval harness): priced after scope.\n"
        "Tip: scope milestones (Ingest, Retrieval, Generation, Deploy, Eval) with acceptance criteria.",
    )

    section(pdf, "Demo script (5 minutes)")
    body(
        pdf,
        "1) Start API + UI (see README: start_documind.ps1 or docker compose).\n"
        "2) Open dashboard, run Flagship compare scenario, show synthesis + sources.\n"
        "3) Open /docs and /health/ready - explain readiness vs liveness for Kubernetes.\n"
        "4) Mention optional API_KEY and CORS allowlist for staging/production.",
    )

    section(pdf, "Repo & contact (fill in)")
    body(
        pdf,
        "Repository: [ your GitHub URL ]\n"
        "Live demo (if hosted): [ URL or 'available on request' ]\n"
        "Email / Upwork: [ your contact ]\n"
        "Note: Do not paste client confidential data into public demos.",
    )

    section(pdf, "How this PDF was generated")
    body(
        pdf,
        "Source: scripts/generate_portfolio_pdf.py (fpdf2). Regenerate after editing copy: "
        "pip install -r scripts/portfolio_requirements.txt && python scripts/generate_portfolio_pdf.py",
    )

    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
