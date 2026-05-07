from __future__ import annotations

import io
import re
from pathlib import Path

from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain.schema import Document

from app.utils.chunker import DocumentChunker


class DocumentService:
    def __init__(self, chunker: DocumentChunker) -> None:
        self.chunker = chunker

    def extract_paper_metadata(self, pages: list[dict]) -> dict:
        first_pages_text = "\n".join([p["text"] for p in pages[:2]])
        lines = [line.strip() for line in first_pages_text.splitlines() if line.strip()]

        title = ""
        title_idx = -1
        for idx, line in enumerate(lines):
            if len(line) > 20 and not line.isupper():
                title = line
                title_idx = idx
                break

        authors = ""
        if title_idx >= 0:
            for offset in (2, 3):
                candidate_idx = title_idx + offset
                if candidate_idx < len(lines):
                    authors = lines[candidate_idx]
                    break

        year_match = re.search(r"\b(20(?:0\d|1\d|2[0-6]))\b", first_pages_text)
        arxiv_match = re.search(r"arXiv[:\s]*(\d{4}\.\d{4,5})", first_pages_text, flags=re.IGNORECASE)

        return {
            "title": title or "",
            "authors": authors or "",
            "year": year_match.group(1) if year_match else "",
            "arxiv_id": arxiv_match.group(1) if arxiv_match else "",
        }

    def load_pdf(self, file_bytes: bytes, filename: str) -> list[dict]:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages: list[dict] = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if len(text) < 20:
                continue
            pages.append({"text": text, "page_number": i + 1, "filename": filename})
        return pages

    def load_docx(self, file_bytes: bytes, filename: str) -> list[dict]:
        doc = DocxDocument(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        pages: list[dict] = []
        page_number = 1
        for i in range(0, len(paragraphs), 5):
            block = "\n".join(paragraphs[i : i + 5]).strip()
            if len(block) < 20:
                continue
            pages.append({"text": block, "page_number": page_number, "filename": filename})
            page_number += 1
        return pages

    def load_txt(self, file_bytes: bytes, filename: str) -> list[dict]:
        text = file_bytes.decode("utf-8", errors="ignore")
        pages: list[dict] = []
        for i in range(0, len(text), 3000):
            chunk = text[i : i + 3000].strip()
            if len(chunk) < 20:
                continue
            pages.append({"text": chunk, "page_number": (i // 3000) + 1, "filename": filename})
        return pages

    def process(self, file_bytes: bytes, filename: str, doc_id: str) -> tuple[list[Document], dict]:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            pages = self.load_pdf(file_bytes, filename)
        elif ext == ".docx":
            pages = self.load_docx(file_bytes, filename)
        elif ext == ".txt":
            pages = self.load_txt(file_bytes, filename)
        else:
            raise ValueError("Unsupported file format. Only .pdf, .docx, .txt are allowed.")

        metadata = self.extract_paper_metadata(pages)
        base_metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "title": metadata["title"] or filename,
            "authors": metadata["authors"],
            "year": metadata["year"],
            "arxiv_id": metadata["arxiv_id"],
        }
        documents = self.chunker.split(pages, base_metadata)
        return documents, base_metadata
