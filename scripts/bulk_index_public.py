#!/usr/bin/env python3
"""
Offline bulk indexer for the **public** vector collection (Wikipedia .txt shards, etc.).

Production patterns:
- **Checkpoint / resume** — JSON list of completed basenames (survives Ollama restarts).
- **Parallel embeddings** — bounded `ThreadPoolExecutor` over chunk texts (Ollama may still serialize; tune `--workers`).
- **Single Chroma add per file** — `add_indexed_batch` after all vectors for that document are ready.
- **Chunk provenance** — `embedding_model`, `chroma_collection`, `indexed_at` on every row (see ChromaEmbeddingService).

Usage (from repo root, Ollama running):
  python scripts/bulk_index_public.py --txt-dir data/wiki_txt_demo --max-files 500 --workers 6
  python scripts/bulk_index_public.py --txt-dir data/wiki_txt --checkpoint data/.bulk_public_checkpoint.json
  python scripts/bulk_index_public.py --txt-dir data/wiki_txt_demo --dry-run
  python scripts/bulk_index_public.py ... --progress-every 100
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.services.document_service import DocumentService  # noqa: E402
from app.services.embedding_service import ChromaEmbeddingService  # noqa: E402
from app.utils.chunker import DocumentChunker  # noqa: E402
from app.utils.ollama_client import OllamaClient  # noqa: E402

logger = logging.getLogger("bulk_index_public")


def _load_done(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("done_files", []))
    except (json.JSONDecodeError, OSError):
        return set()


def _save_done(path: Path, done: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"done_files": sorted(done)}, indent=2), encoding="utf-8")


def _embed_parallel(client: OllamaClient, texts: list[str], workers: int) -> list[list[float]]:
    if not texts:
        return []
    if workers <= 1:
        return [client.embed(t) for t in texts]

    out: list[list[float] | None] = [None] * len(texts)

    def _job(i: int, text: str) -> tuple[int, list[float]]:
        return i, client.embed(text)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, i, texts[i]) for i in range(len(texts))]
        for fut in as_completed(futures):
            i, vec = fut.result()
            out[i] = vec

    if any(v is None for v in out):
        raise RuntimeError("parallel embedding produced incomplete results")
    return [v for v in out]


def _index_file(
    path: Path,
    doc_id: str,
    document_service: DocumentService,
    embedding: ChromaEmbeddingService,
    client: OllamaClient,
    workers: int,
) -> int:
    file_bytes = path.read_bytes()
    docs, _ = document_service.process(file_bytes, path.name, doc_id)
    if not docs:
        return 0
    texts = [d.page_content for d in docs]
    t0 = time.perf_counter()
    embeddings = _embed_parallel(client, texts, workers)
    elapsed = time.perf_counter() - t0
    ids = [f"{doc_id}_{i}" for i in range(len(docs))]
    metadatas = [dict(d.metadata or {}) for d in docs]
    embedding.add_indexed_batch(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.debug("embedded %s chunks in %.1fs (%s)", len(docs), elapsed, path.name)
    return len(docs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bulk-index .txt files into the public Chroma collection.")
    ap.add_argument("--txt-dir", type=Path, required=True, help="Directory of UTF-8 .txt files")
    ap.add_argument("--max-files", type=int, default=0, help="Stop after N new files (0 = no limit)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="JSON file listing completed basenames for resume")
    ap.add_argument("--workers", type=int, default=4, help="Parallel Ollama embed calls per file (>=1)")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=25,
        metavar="N",
        help="Emit INFO progress every N newly indexed files (raise for huge corpora to cut log spam)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Count files and chunks only (no Ollama / Chroma writes)",
    )
    ap.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    ns = ap.parse_args()

    logging.basicConfig(level=getattr(logging, ns.log_level), format="%(levelname)s %(message)s")

    txt_dir: Path = ns.txt_dir.resolve()
    if not txt_dir.is_dir():
        logger.error("Not a directory: %s", txt_dir)
        return 1

    settings = get_settings()
    files = sorted(p for p in txt_dir.glob("*.txt") if p.is_file() and not p.name.startswith("."))
    done = _load_done(ns.checkpoint)

    chunker = DocumentChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    document_service = DocumentService(chunker=chunker)

    if ns.dry_run:
        total_chunks = 0
        considered = 0
        for path in files:
            if path.name in done:
                continue
            if ns.max_files and considered >= ns.max_files:
                break
            considered += 1
            doc_id = f"wiki_txt_{path.stem}"[:120]
            docs, _ = document_service.process(path.read_bytes(), path.name, doc_id)
            total_chunks += len(docs)
        logger.info(
            "DRY-RUN: txt_files=%s pending_files=%s pending_chunks≈%s (checkpoint skips=%s)",
            len(files),
            considered,
            total_chunks,
            len(done),
        )
        return 0

    client = OllamaClient(
        base_url=settings.OLLAMA_BASE_URL,
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        request_timeout_sec=float(settings.OLLAMA_REQUEST_TIMEOUT_SEC),
    )
    if not client.health_check().get("available"):
        logger.error("Ollama not available at %s", settings.OLLAMA_BASE_URL)
        return 1

    embedding = ChromaEmbeddingService(
        settings.CHROMA_COLLECTION_PUBLIC,
        client,
        persist_dir=settings.CHROMA_PERSIST_DIR,
    )

    workers = max(1, int(ns.workers))
    indexed = 0
    skipped = 0
    chunk_total = 0
    t_start = time.perf_counter()

    for path in files:
        if path.name in done:
            skipped += 1
            continue
        if ns.max_files and indexed >= ns.max_files:
            break
        doc_id = f"wiki_txt_{path.stem}"[:120]
        try:
            n = _index_file(path, doc_id, document_service, embedding, client, workers)
            if n == 0:
                logger.warning("No chunks for %s — skipping checkpoint entry", path.name)
                continue
            done.add(path.name)
            chunk_total += n
            if ns.checkpoint:
                _save_done(ns.checkpoint, done)
            indexed += 1
            if indexed % max(1, ns.progress_every) == 0:
                elapsed = time.perf_counter() - t_start
                logger.info(
                    "progress files=%s chunks=%s elapsed_s=%.0f chkpt=%s",
                    indexed,
                    chunk_total,
                    elapsed,
                    ns.checkpoint or "(none)",
                )
        except Exception as exc:
            logger.exception("Failed %s: %s", path, exc)
            return 1

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Done: new_files=%s skipped_checkpoint=%s new_chunks=%s wall_s=%.1f collection=%s workers=%s",
        indexed,
        skipped,
        chunk_total,
        elapsed,
        settings.CHROMA_COLLECTION_PUBLIC,
        workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
