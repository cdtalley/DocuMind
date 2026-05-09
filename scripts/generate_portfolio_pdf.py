#!/usr/bin/env python3
"""Build portfolio/DocuMind_Upwork_Catalog.pdf (requires: pip install -r scripts/portfolio_requirements.txt).

Prefer opening portfolio/DocuMind_Upwork_Catalog.html in Chrome and Print -> Save as PDF for best layout.
This script uses large type and wide line spacing so the PDF stays readable if you need a CLI-only file.
"""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "portfolio" / "DocuMind_Upwork_Catalog.pdf"


class PDF(FPDF):
    def __init__(self) -> None:
        super().__init__(format="letter", unit="mm")
        self.set_auto_page_break(auto=True, margin=18)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 80, 80)
        self.set_x(self.l_margin)
        self.cell(self.epw, 8, f"Page {self.page_no()}", align="C")


def text_width(pdf: PDF) -> float:
    return pdf.epw


def section(pdf: PDF, title: str) -> None:
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(text_width(pdf), 7, title)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)


def body(pdf: PDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(text_width(pdf), 6.2, text)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = PDF()
    pdf.set_margins(22, 22, 22)
    pdf.add_page()
    w = text_width(pdf)

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 64, 175)
    pdf.multi_cell(w, 11, "DocuMind")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(45, 45, 45)
    pdf.multi_cell(w, 7, "Local-first RAG - portfolio and Upwork catalog brief")
    pdf.ln(5)

    body(
        pdf,
        "Use this file on Upwork as a catalog attachment or scope handout. "
        "Replace bracketed fields with your rates and links. "
        "For best readability, also use the HTML version in a browser and Print to PDF.",
    )

    section(pdf, "One-line pitch")
    body(
        pdf,
        "End-to-end retrieval-augmented Q and A: ingest PDFs, DOCX, TXT and arXiv; chunk with section metadata; "
        "embed in ChromaDB; retrieve with rerank and diversity; answer with mode-specific prompts and citations - "
        "FastAPI plus Next.js plus Ollama (no cloud LLM keys required for demos).",
    )

    section(pdf, "Buyer outcomes")
    body(
        pdf,
        "- Grounded answers with source cards (paper title, section, chunk, distance).\n"
        "- Modes: general Q and A, compare, methodology, dataset inventory, reproducibility checklist.\n"
        "- Operations: liveness and readiness HTTP probes, request IDs, optional API key on /api/v1, CORS allowlist, "
        "gzip, security headers, optional JSON logs, Docker volume for vectors.\n"
        "- Honest deletes (404 when nothing removed) and validated section filters (422 on bad values).",
    )

    section(pdf, "Tech stack (keywords)")
    body(
        pdf,
        "Python 3.11+, FastAPI, Pydantic v2, Uvicorn, ChromaDB, LangChain text splitters, Ollama (llama3, "
        "nomic-embed-text), httpx, Next.js 15, React 18, TypeScript, pytest, Docker.",
    )

    section(pdf, "Suggested Upwork titles")
    body(
        pdf,
        "A) Ship a local RAG document assistant (FastAPI + Chroma + Next.js)\n"
        "B) Retrieval QA over your PDFs with citations, modes, and health checks\n"
        "C) MVP: ingest plus vector index plus grounded answers - Ollama or swap to OpenAI",
    )

    section(pdf, "Pricing anchors (edit for your tier)")
    body(
        pdf,
        "Hourly (integration and architecture): [ $85-$150+ / hr ] depending on profile and reviews.\n"
        "Fixed - discovery plus written architecture (1-2 weeks): [ $2,500-$6,000 ].\n"
        "Fixed - MVP RAG (ingest, index, UI, one cloud deploy, basic eval): [ $5,000-$12,000 ].\n"
        "Fixed - hardening (auth, rate limits, multi-tenant ACLs, formal eval): priced after scope.\n"
        "Tip: milestones (Ingest, Retrieval, Generation, Deploy, Eval) with acceptance criteria.",
    )

    section(pdf, "Demo script (5 minutes)")
    body(
        pdf,
        "1) Start API and UI (README: start_documind.ps1 or docker compose).\n"
        "2) Open dashboard, run Flagship compare scenario, show synthesis and sources.\n"
        "3) Open /docs and /health/ready - explain readiness versus liveness for Kubernetes.\n"
        "4) Mention optional API_KEY and CORS for staging and production.",
    )

    section(pdf, "Repo and contact (fill in)")
    body(
        pdf,
        "Repository: [ your GitHub URL ]\n"
        "Live demo: [ URL or available on request ]\n"
        "Email / Upwork: [ your contact ]\n"
        "Note: do not paste client confidential data into public demos.",
    )

    section(pdf, "Regenerate this PDF")
    body(
        pdf,
        "pip install -r scripts/portfolio_requirements.txt\n"
        "python scripts/generate_portfolio_pdf.py",
    )

    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
