"""
1000×750 Upwork tile — Sentinel-style infographic (no scaled screenshots).

Dark navy grid, cyan corner glow, bold title + stack line, 2×2 metric cards, bottom tech pills.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 1000, 750
BG = (10, 16, 32)
GRID = (22, 34, 56)
ACCENT = (34, 211, 238)
ACCENT_SOFT = (125, 211, 252)
CARD_BG = (16, 26, 46)
CARD_EDGE = (48, 62, 88)
LABEL_GRAY = (148, 163, 184)
WHITE = (255, 255, 255)


def _windows_font(name: str) -> Path | None:
    windir = os.environ.get("WINDIR", "C:/Windows")
    p = Path(windir) / "Fonts" / name
    return p if p.is_file() else None


def _sentinel_fonts() -> dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    bold = _windows_font("segoeuib.ttf") or _windows_font("arialbd.ttf")
    reg = _windows_font("segoeui.ttf") or _windows_font("arial.ttf")
    try:
        if bold and reg:
            return {
                "kicker": ImageFont.truetype(str(bold), 12),
                "title": ImageFont.truetype(str(bold), 40),
                "stack_line": ImageFont.truetype(str(reg), 16),
                "metric_val": ImageFont.truetype(str(bold), 34),
                "metric_lbl": ImageFont.truetype(str(bold), 11),
                "pill": ImageFont.truetype(str(reg), 13),
            }
    except OSError:
        pass
    d = ImageFont.load_default()
    return {k: d for k in ("kicker", "title", "stack_line", "metric_val", "metric_lbl", "pill")}


def _draw_subtle_grid(draw: ImageDraw.ImageDraw, width: int, height: int, step: int = 44) -> None:
    for x in range(0, width + 1, step):
        draw.line((x, 0, x, height), fill=GRID, width=1)
    for y in range(0, height + 1, step):
        draw.line((0, y, width, y), fill=GRID, width=1)


def _composite_corner_glow(base_rgba: Image.Image) -> Image.Image:
    w, h = base_rgba.size
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = w - 72, h - 64
    for r, a in (
        (420, 5),
        (340, 7),
        (270, 9),
        (200, 11),
        (140, 13),
        (90, 12),
        (55, 9),
    ):
        gd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*ACCENT, a))
    return Image.alpha_composite(base_rgba, glow)


def _draw_metric_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    cw: int,
    ch: int,
    value: str,
    label: str,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    draw.rounded_rectangle((x, y, x + cw, y + ch), radius=10, fill=CARD_BG, outline=CARD_EDGE, width=1)
    draw.text((x + 20, y + 22), value, fill=WHITE, font=fonts["metric_val"])
    draw.text((x + 20, y + 78), label.upper(), fill=LABEL_GRAY, font=fonts["metric_lbl"])


def _draw_pill_row(
    draw: ImageDraw.ImageDraw,
    tags: list[str],
    font: ImageFont.ImageFont,
    x0: int,
    y0: int,
    gap: int = 10,
) -> None:
    x = x0
    for tag in tags:
        bbox = draw.textbbox((0, 0), tag, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pw, ph = tw + 22, max(28, th + 14)
        draw.rounded_rectangle((x, y0, x + pw, y0 + ph), radius=14, fill=(12, 22, 40), outline=ACCENT, width=1)
        draw.text((x + 11, y0 + (ph - th) // 2 - 1), tag, fill=ACCENT_SOFT, font=font)
        x += pw + gap


def render_documind_sentinel_tile(out_png: Path) -> None:
    """Typography-led catalog tile matching SentinelAI reference (readable at grid size)."""
    fonts = _sentinel_fonts()
    base = Image.new("RGBA", (W, H), (*BG, 255))
    draw = ImageDraw.Draw(base)
    _draw_subtle_grid(draw, W, H)
    base = _composite_corner_glow(base)
    draw = ImageDraw.Draw(base)

    margin_l = 40
    y = 28
    draw.text((margin_l, y), "SYSTEM STACK", fill=ACCENT, font=fonts["kicker"])
    y += 26

    t1 = "DocuMind — "
    t2 = "Grounded corpus RAG"
    draw.text((margin_l, y), t1, fill=WHITE, font=fonts["title"])
    b1 = draw.textbbox((margin_l, y), t1, font=fonts["title"])
    draw.text((b1[2], y), t2, fill=ACCENT, font=fonts["title"])

    stack = "FastAPI · Chroma cosine · Ollama · mode-aware K · keyword rerank · Next.js · Docker"
    y += 52
    draw.text((margin_l, y), stack, fill=LABEL_GRAY, font=fonts["stack_line"])

    # 2×2 cards — production signals (match README / config story)
    y_cards = 198
    cw, ch = 228, 118
    gap = 14
    _draw_metric_card(draw, margin_l, y_cards, cw, ch, "Chroma", "cosine · persist", fonts)
    _draw_metric_card(draw, margin_l + cw + gap, y_cards, cw, ch, "Rerank", "dense + keyword", fonts)
    _draw_metric_card(draw, margin_l, y_cards + ch + gap, cw, ch, "Health", "live · ready", fonts)
    _draw_metric_card(draw, margin_l + cw + gap, y_cards + ch + gap, cw, ch, "FastAPI", "openapi · v1", fonts)

    pills = ["FastAPI", "Chroma", "Ollama", "Next.js", "Docker", "LangChain"]
    _draw_pill_row(draw, pills, fonts["pill"], margin_l, H - 56, gap=10)

    draw.rectangle((0, H - 3, W, H), fill=ACCENT)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(out_png, "PNG", optimize=True)


def render_catalog_thumbnail(_master: Path, out_png: Path) -> None:
    """Composite Upwork tile (Sentinel-style infographic; master path unused)."""
    render_documind_sentinel_tile(out_png)


def write_plain_top_crop_thumbnail(src: Path, dst: Path, width: int = 1000, height: int = 750) -> None:
    Image.MAX_IMAGE_PIXELS = 200_000_000
    im = Image.open(src).convert("RGB")
    sw, sh = im.size
    if sw < 400 or sh < 300:
        raise ValueError(f"Source too small: {sw}×{sh}")
    scale = width / sw
    nh = max(height, int(round(sh * scale)))
    resized = im.resize((width, nh), Image.Resampling.LANCZOS)
    if nh <= height:
        canvas = Image.new("RGB", (width, height), (11, 15, 26))
        canvas.paste(resized, (0, (height - nh) // 2))
        out = canvas
    else:
        out = resized.crop((0, 0, width, height))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, "PNG", optimize=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build 1000×750 catalog thumbnail (Sentinel-style or plain crop)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plain", action="store_true", help="Top-crop from dashboard PNG instead")
    ap.add_argument("--src", type=Path, help="Dashboard PNG (required with --plain)")
    ns = ap.parse_args()
    if ns.plain:
        if not ns.src:
            ap.error("--plain requires --src")
        write_plain_top_crop_thumbnail(ns.src, ns.out)
    else:
        render_documind_sentinel_tile(ns.out)
    print(f"Wrote {ns.out}")
