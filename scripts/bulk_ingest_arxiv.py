#!/usr/bin/env python3
"""
Backfill the vector store from a list of arXiv IDs (production-style batch ingest).

Usage (API must be running with Ollama up):
  python scripts/bulk_ingest_arxiv.py --api http://127.0.0.1:8001 --file data/arxiv_seed_list.txt --delay 2

Requires: requests
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="POST /api/v1/fetch-arxiv for each ID in a text file.")
    p.add_argument("--api", default="http://127.0.0.1:8001", help="DocuMind API base URL")
    p.add_argument("--file", default="data/arxiv_seed_list.txt", help="One arXiv id per line (# comments ok)")
    p.add_argument("--delay", type=float, default=1.5, help="Seconds between requests (be polite to arXiv)")
    args = p.parse_args()

    base = args.api.rstrip("/")
    path = args.file
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line.split()[0])

    ok, fail = 0, 0
    for i, arxiv_id in enumerate(lines):
        url = f"{base}/api/v1/fetch-arxiv"
        try:
            r = requests.post(url, json={"arxiv_id": arxiv_id}, timeout=300)
            if r.status_code == 200:
                title = r.json().get("title", "")
                print(f"[{i+1}/{len(lines)}] OK {arxiv_id} — {title[:80]}")
                ok += 1
            else:
                print(f"[{i+1}/{len(lines)}] FAIL {arxiv_id} — {r.status_code} {r.text[:200]}")
                fail += 1
        except requests.RequestException as exc:
            print(f"[{i+1}/{len(lines)}] ERROR {arxiv_id} — {exc}")
            fail += 1
        if i + 1 < len(lines):
            time.sleep(max(0.0, args.delay))
    print(f"Finished: {ok} succeeded, {fail} failed")


if __name__ == "__main__":
    main()
