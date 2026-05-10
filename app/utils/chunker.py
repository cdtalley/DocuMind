from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    PAPER_SECTIONS = [
        "abstract",
        "introduction",
        "related work",
        "background",
        "methodology",
        "method",
        "approach",
        "experiments",
        "experimental",
        "results",
        "evaluation",
        "discussion",
        "conclusion",
        "limitations",
        "future work",
        "references",
        "acknowledgments",
    ]

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

    def detect_section(self, text: str) -> str:
        prefix = text[:60].lower()
        for section in self.PAPER_SECTIONS:
            if section in prefix:
                return section
        return "body"

    def split(self, pages: list[dict], base_metadata: dict) -> list[Document]:
        if not pages:
            return []

        page_starts: list[tuple[int, int]] = []
        full_text_parts: list[str] = []
        running_index = 0

        for page in pages:
            page_text = page["text"].strip()
            page_starts.append((running_index, int(page["page_number"])))
            full_text_parts.append(page_text)
            running_index += len(page_text) + 2

        full_text = "\n\n".join(full_text_parts)
        chunks = self.splitter.create_documents([full_text])
        result: list[Document] = []

        for i, chunk in enumerate(chunks):
            start_char = chunk.metadata.get("start_index", 0)
            page_number = 1
            for start_idx, page_num in page_starts:
                if start_idx <= start_char:
                    page_number = page_num
                else:
                    break

            metadata = {
                **base_metadata,
                "page_number": page_number,
                "chunk_index": i,
                "section": self.detect_section(chunk.page_content),
                "char_count": len(chunk.page_content),
            }
            result.append(Document(page_content=chunk.page_content, metadata=metadata))

        return result
