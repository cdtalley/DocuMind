#!/usr/bin/env python3
"""
Drive the Next.js dashboard: wait until the API has indexed enough papers, run a showcase query,
wait until the synthesis body is non-trivial, then write a PNG that shows the system in action.

Viewport capture uses a fixed height cap (default 3200px) and scrolls the synthesis panel into view.
Never trust documentElement.scrollHeight for screenshots — long Markdown answers can exceed 90kpx and
produce an unusable vertical strip when scaled down.

First boot after SAMPLE_CORPUS_VERSION bump can take a long time (many embeds). Use --min-docs lower for a smaller
corpus, or raise --wait-index-ms for a full v7 seed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from catalog_thumb_art import render_catalog_thumbnail, write_plain_top_crop_thumbnail

DEFAULT_OUT = ROOT / "portfolio" / "screenshots" / "documind-dashboard.png"
# Upwork Project Catalog / portfolio: 4:3, designed for 1000×750 display (see Upwork Help).
DEFAULT_THUMB_OUT = ROOT / "portfolio" / "screenshots" / "documind-upwork-catalog-1000x750.png"


def write_catalog_thumbnail(src: Path, dst: Path, *, plain: bool) -> None:
    if plain:
        write_plain_top_crop_thumbnail(src, dst)
    else:
        render_catalog_thumbnail(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3002/", help="Dashboard base URL")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PNG path")
    parser.add_argument(
        "--scenario",
        choices=("flagship", "datasets", "reproduce", "methodology"),
        default="flagship",
        help="Showcase card before Run query (default: flagship — richer cross-paper answer for portfolio shots).",
    )
    parser.add_argument(
        "--min-docs",
        type=int,
        default=400,
        help="Wait until topbar text matches at least this many indexed papers (default 400 for full v7 corpus).",
    )
    parser.add_argument(
        "--wait-index-ms",
        type=int,
        default=3_600_000,
        help="Max time to wait for min-docs in UI (default 60 minutes).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=360_000,
        help="Max wait for synthesis body text after Run query (default 6 minutes for local LLM).",
    )
    parser.add_argument(
        "--min-answer-chars",
        type=int,
        default=120,
        help="Wait until .prose-answer has at least this many characters (proves a real model response).",
    )
    parser.add_argument(
        "--full-page",
        action="store_true",
        help="Capture the entire scrollable page (can be 50k+ px tall with long answers — poor for portfolios).",
    )
    parser.add_argument(
        "--viewport-height",
        type=int,
        default=3200,
        help="Browser viewport height for the final PNG when not using --full-page (default 3200, max 4096).",
    )
    parser.add_argument(
        "--thumb-out",
        type=Path,
        default=DEFAULT_THUMB_OUT,
        help="Write Upwork catalog 4:3 PNG (1000×750) here after the main capture (default beside dashboard).",
    )
    parser.add_argument(
        "--no-thumb",
        action="store_true",
        help="Skip writing the 1000×750 Upwork catalog thumbnail.",
    )
    parser.add_argument(
        "--thumb-only",
        action="store_true",
        help="Only build --thumb-out from existing --out PNG (no browser; use after manual edits).",
    )
    parser.add_argument(
        "--plain-catalog-thumb",
        action="store_true",
        help="Simple 4:3 top-crop instead of the branded attention layout (A/B or strict minimalism).",
    )
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install:  .venv\\Scripts\\pip install -r scripts/screenshot_requirements.txt", file=sys.stderr)
        print("Browsers: .venv\\Scripts\\playwright install chromium", file=sys.stderr)
        return 1

    label_map = {
        "flagship": "Flagship: research landscape",
        "datasets": "Dataset map",
        "reproduce": "Repro blueprint",
        "methodology": "Methods & training",
    }
    scenario_label = label_map[args.scenario]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.thumb_only:
        if not args.out.is_file():
            print(f"Missing source image: {args.out}", file=sys.stderr)
            return 1
        if args.no_thumb:
            print("--thumb-only conflicts with --no-thumb", file=sys.stderr)
            return 1
        try:
            write_catalog_thumbnail(args.out, args.thumb_out, plain=args.plain_catalog_thumb)
        except (OSError, ValueError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Wrote {args.thumb_out}")
        return 0

    min_docs = max(1, args.min_docs)
    # Top bar uses CSS text-transform; innerText is e.g. "469 DOCS …" not "469 docs ·".
    js = f"""
    () => {{
      const t = document.body.innerText || "";
      const m = t.match(/(\\d+)\\s+docs\\b/i);
      if (!m) return false;
      return parseInt(m[1], 10) >= {min_docs};
    }}
    """

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_function(js, timeout=args.wait_index_ms)
        page.get_by_role("button", name="Refresh status").click()
        page.wait_for_timeout(1500)
        page.get_by_role("button", name=scenario_label).click()
        page.wait_for_timeout(500)
        page.get_by_role("button", name="Run query").first.click()
        page.get_by_role("heading", name="Synthesis").wait_for(state="visible", timeout=args.timeout_ms)

        min_chars = max(20, args.min_answer_chars)
        page.wait_for_function(
            f"""() => {{
              const el = document.querySelector(".prose-answer");
              if (!el) return false;
              const t = (el.innerText || "").trim();
              return t.length >= {min_chars};
            }}""",
            timeout=args.timeout_ms,
        )
        # Let Markdown / layout settle; confidence bar and sources often paint shortly after.
        page.wait_for_timeout(3500)
        # Framing: sticky topbar stays visible; scroll so synthesis + sources dominate the frame.
        page.evaluate(
            """() => {
              const panel = document.querySelector(".answer-panel");
              const head = document.querySelector("#synthesis-heading");
              const tgt = head || panel;
              if (!tgt) return;
              const top = tgt.getBoundingClientRect().top + window.scrollY - 100;
              window.scrollTo({ top: Math.max(0, top), behavior: "instant" });
            }"""
        )
        page.wait_for_timeout(400)

        if args.full_page:
            page.screenshot(path=str(args.out), full_page=True)
        else:
            vh = max(900, min(int(args.viewport_height), 4096))
            page.set_viewport_size({"width": 1440, "height": vh})
            page.wait_for_timeout(350)
            page.screenshot(path=str(args.out), full_page=False)

        browser.close()

    print(f"Wrote {args.out}")
    if not args.no_thumb and args.out.is_file():
        if args.full_page:
            print(
                "Skipping catalog thumbnail: --full-page captures are often too tall for a clean 4:3 crop. "
                "Re-run without --full-page, then upload documind-upwork-catalog-1000x750.png.",
                file=sys.stderr,
            )
        else:
            try:
                write_catalog_thumbnail(args.out, args.thumb_out, plain=args.plain_catalog_thumb)
                kind = "plain 4:3" if args.plain_catalog_thumb else "branded attention 4:3"
                print(f"Wrote {args.thumb_out} (Upwork catalog — {kind})")
            except (OSError, ValueError, RuntimeError) as exc:
                print(str(exc), file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
