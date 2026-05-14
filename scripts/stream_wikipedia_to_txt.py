#!/usr/bin/env python3
"""
Stream public-domain encyclopedia text into .txt files for DocuMind ingest.

Source: Hugging Face `wikimedia/wikipedia` (CC BY-SA content; follow their attribution
requirements if you redistribute answers). Uses **streaming** so you can process millions
of articles without holding the full dataset in RAM.

This is the practical split from the "research PDF" domain: same chunk/embed pipeline
(`DocumentService.load_txt` + chunker), different corpus provenance and scale story.

Install once:
  pip install datasets

Examples:
  python scripts/stream_wikipedia_to_txt.py --out-dir data/wiki_txt_demo --max-articles 2000
  python scripts/stream_wikipedia_to_txt.py --out-dir data/wiki_txt --subset 20231101.en --max-articles 0

`--max-articles 0` means no cap (production batch); ensure disk and downstream ingest can cope.

Typical interview framing at "millions" scale:
- You stop ingesting through a single FastAPI upload: batch embed + bulk upsert, separate
  collection (`CHROMA_COLLECTION_PUBLIC`), monitor ANN recall/latency, plan shard or hosted
  vector (Qdrant, pgvector, managed Pinecone) when Chroma persistence becomes the bottleneck.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _safe_filename(title: str, article_id: str, idx: int) -> str:
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", title).strip("_")[:80] or "article"
    return f"wiki_{article_id}_{idx:08d}_{base}.txt"


def main() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing dependency: pip install datasets", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description="Stream Wikipedia (wikimedia/wikipedia) to .txt files.")
    p.add_argument(
        "--subset",
        default="20231101.en",
        help="wikimedia/wikipedia config, e.g. 20231101.en (see dataset card on HF if this fails).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "wiki_txt",
        help="Directory for .txt files (created if missing).",
    )
    p.add_argument(
        "--max-articles",
        type=int,
        default=500,
        help="Stop after N articles (0 = unlimited).",
    )
    p.add_argument(
        "--min-text-chars",
        type=int,
        default=400,
        help="Skip stubs shorter than this many body characters.",
    )
    p.add_argument(
        "--max-body-chars",
        type=int,
        default=400_000,
        help="Truncate article body to limit file size for very long pages.",
    )
    ns = p.parse_args()

    out: Path = ns.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("wikimedia/wikipedia", ns.subset, split="train", streaming=True)
    except Exception as exc:
        print(f"load_dataset failed: {exc}", file=sys.stderr)
        print("Try another --subset from https://huggingface.co/datasets/wikimedia/wikipedia", file=sys.stderr)
        return 1

    written = 0
    for i, row in enumerate(ds):
        if ns.max_articles and written >= ns.max_articles:
            break
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        aid = str(row.get("id", i))
        if len(text) < ns.min_text_chars:
            continue
        if len(text) > ns.max_body_chars:
            text = text[: ns.max_body_chars] + "\n\n[truncated]\n"
        # First non-trivial line is used as title hint by extract_paper_metadata; body follows.
        block = f"{title}\nWikipedia article id {aid}\n\n{text}\n"
        path = out / _safe_filename(title, aid, written)
        path.write_text(block, encoding="utf-8")
        written += 1
        if written % 500 == 0:
            print(f"... {written} articles written under {out}", flush=True)

    print(f"Done: wrote {written} files to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
