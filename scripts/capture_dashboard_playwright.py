#!/usr/bin/env python3
"""Drive the Next.js dashboard: run a showcase query, then full-page screenshot (Playwright)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "portfolio" / "screenshots" / "documind-dashboard.png"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3002/", help="Dashboard base URL")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PNG path")
    parser.add_argument(
        "--scenario",
        choices=("flagship", "datasets", "reproduce", "methodology"),
        default="datasets",
        help="Showcase card to click before Run query (default: datasets — fast, no LLM narrative).",
    )
    parser.add_argument("--timeout-ms", type=int, default=180_000, help="Max wait for synthesis panel")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install Playwright:  .venv\\Scripts\\pip install playwright", file=sys.stderr)
        print("Then browsers:        .venv\\Scripts\\playwright install chromium", file=sys.stderr)
        return 1

    label_map = {
        "flagship": "Flagship: research landscape",
        "datasets": "Dataset map",
        "reproduce": "Repro blueprint",
        "methodology": "Methods & training",
    }
    scenario_label = label_map[args.scenario]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
        page.get_by_role("button", name=scenario_label).click()
        page.get_by_role("button", name="Run query").first.click()
        # Synthesis heading appears after successful query
        page.get_by_role("heading", name="Synthesis").wait_for(state="visible", timeout=args.timeout_ms)
        page.wait_for_timeout(4000)
        page.screenshot(path=str(args.out), full_page=True)
        browser.close()

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
