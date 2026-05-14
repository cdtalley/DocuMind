#!/usr/bin/env python3
"""
Print live Chroma + API snapshot for capacity interviews and ops checks.

  python scripts/report_corpus_scale.py
  python scripts/report_corpus_scale.py --api-base http://127.0.0.1:8001

Requires a running DocuMind API (uses /health and /api/v1/libraries). Optional: local
CHROMA_PERSIST_DIR size when the directory exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _dir_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _human(n: int) -> str:
    for unit, label in ((1 << 40, "TiB"), (1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")):
        if n >= unit:
            return f"{n / unit:.2f} {label}"
    return f"{n} B"


def main() -> int:
    ap = argparse.ArgumentParser(description="Report DocuMind corpus scale from HTTP + optional disk.")
    ap.add_argument("--api-base", default=os.environ.get("DOCUMIND_API_BASE", "http://127.0.0.1:8001"))
    ap.add_argument("--chroma-dir", type=Path, default=None, help="Override persist dir for size scan")
    ns = ap.parse_args()

    base = ns.api_base.rstrip("/")
    chroma = (ns.chroma_dir or (ROOT / os.environ.get("CHROMA_PERSIST_DIR", "chroma_db"))).resolve()

    def get(path: str) -> dict:
        url = f"{base}{path}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        health = get("/health")
        libs = get("/api/v1/libraries")
    except urllib.error.URLError as exc:
        print(f"API unreachable at {base}: {exc}", file=sys.stderr)
        print("Start the API (e.g. uvicorn app.main:app --port 8001) and retry.", file=sys.stderr)
        return 1

    print("=== DocuMind corpus scale report ===\n")
    print(json.dumps({"health": health, "libraries": libs}, indent=2))
    print()

    b = _dir_bytes(chroma)
    if b:
        print(f"CHROMA_PERSIST_DIR on disk: {chroma}")
        print(f"Approximate size: {_human(b)} ({b} bytes)")
    else:
        print(f"No readable directory at {chroma} (set --chroma-dir or CHROMA_PERSIST_DIR).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
