#!/usr/bin/env python3
"""
Build a large **public** corpus: stream Wikipedia (HF) to .txt, then bulk-index into Chroma.

The git repo stays small; **you** choose how big the index gets via `--articles`.

Requires: pip install datasets

Examples:
  python scripts/build_public_corpus.py --articles 5000
  python scripts/build_public_corpus.py --articles 50000 --workers 8
  python scripts/build_public_corpus.py --skip-stream --out-dir data/wiki_txt_demo

Safety: `--articles 0` (uncapped stream) requires `--allow-unbounded`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(argv: list[str]) -> int:
    r = subprocess.run([sys.executable, *argv], cwd=ROOT)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Stream Wikipedia shards then bulk-index public collection.")
    ap.add_argument("--articles", type=int, default=5000, help="Max HF articles to materialize (0=uncapped, needs --allow-unbounded)")
    ap.add_argument("--subset", default="20231101.en", help="wikimedia/wikipedia config name")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "wiki_txt_build",
        help="Directory for streamed .txt files",
    )
    ap.add_argument(
        "--checkpoint-bulk",
        type=Path,
        default=ROOT / "data" / ".bulk_public_checkpoint.json",
        help="Resume file for bulk_index_public",
    )
    ap.add_argument("--workers", type=int, default=4, help="Parallel embed workers per file (bulk_index_public)")
    ap.add_argument("--skip-stream", action="store_true", help="Only run bulk indexer (out-dir must exist)")
    ap.add_argument("--skip-index", action="store_true", help="Only stream to disk (no Ollama/Chroma)")
    ap.add_argument("--dry-run-index", action="store_true", help="Pass through to bulk_index_public --dry-run")
    ap.add_argument(
        "--allow-unbounded",
        action="store_true",
        help="Required when --articles 0 (can exhaust disk)",
    )
    ns = ap.parse_args()

    if ns.articles == 0 and not ns.allow_unbounded:
        print("Refusing --articles 0 without --allow-unbounded (risk of filling disk).", file=sys.stderr)
        return 2

    out = ns.out_dir.resolve()
    if not ns.skip_stream:
        stream_args = [
            str(ROOT / "scripts" / "stream_wikipedia_to_txt.py"),
            "--out-dir",
            str(out),
            "--subset",
            ns.subset,
            "--max-articles",
            str(ns.articles),
        ]
        rc = _run(stream_args)
        if rc != 0:
            return rc

    if ns.skip_index:
        print("Skip index: done after stream.", file=sys.stderr)
        return 0

    bulk_args = [
        str(ROOT / "scripts" / "bulk_index_public.py"),
        "--txt-dir",
        str(out),
        "--checkpoint",
        str(ns.checkpoint_bulk),
        "--workers",
        str(ns.workers),
    ]
    if ns.dry_run_index:
        bulk_args.append("--dry-run")
    return _run(bulk_args)


if __name__ == "__main__":
    raise SystemExit(main())
