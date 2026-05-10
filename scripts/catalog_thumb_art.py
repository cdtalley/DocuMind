"""
1000×750 (4:3) catalog image: short stack summary + cropped live UI from dashboard PNG.

Upwork suggests ~1000×750 for Project Catalog tiles; this keeps text factual and the UI legible.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _windows_font(name: str) -> Path | None:
    windir = os.environ.get("WINDIR", "C:/Windows")
    p = Path(windir) / "Fonts" / name
    return p if p.is_file() else None


def _load_fonts() -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ...]:
    title_p = _windows_font("segoeuib.ttf") or _windows_font("arialbd.ttf")
    body_p = _windows_font("segoeui.ttf") or _windows_font("arial.ttf")
    try:
        if title_p and body_p:
            return (
                ImageFont.truetype(str(title_p), 40),
                ImageFont.truetype(str(body_p), 18),
                ImageFont.truetype(str(body_p), 16),
                ImageFont.truetype(str(title_p), 12),
            )
    except OSError:
        pass
    d = ImageFont.load_default()
    return (d, d, d, d)


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    w, h = size
    mask = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(mask)
    dr.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    return mask


def render_catalog_thumbnail(master_png: Path, out_png: Path) -> None:
    """Composite 1000×750 catalog/portfolio thumbnail from master dashboard PNG."""
    Image.MAX_IMAGE_PIXELS = 200_000_000

    master = Image.open(master_png).convert("RGB")
    mw, mh = master.size
    if mw < 400 or mh < 400:
        raise ValueError(f"Master too small: {mw}×{mh}")

    W, H = 1000, 750
    LEFT_W = 292
    RIGHT_W = W - LEFT_W
    radius = 16
    pad = (6, 10, 22)

    canvas = Image.new("RGB", (W, H), pad)
    draw = ImageDraw.Draw(canvas)

    # Left gradient panel
    left = Image.new("RGB", (LEFT_W, H))
    ld = ImageDraw.Draw(left)
    for y in range(H):
        t = y / max(H - 1, 1)
        r = int(3 + t * 22)
        g = int(10 + t * 40)
        b = int(26 + t * 58)
        ld.line([(0, y), (LEFT_W, y)], fill=(r, g, b))

    ft_title, ft_sub, ft_li, ft_badge = _load_fonts()
    ld.text((24, 36), "DocuMind", fill=(248, 250, 252), font=ft_title)
    ld.text((24, 100), "RAG over a paper library", fill=(125, 211, 252), font=ft_sub)

    bullets = [
        "PDF / DOCX / TXT ingest",
        "Chroma + Ollama",
        "FastAPI + OpenAPI",
        "Answer + source cards",
    ]
    yb = 152
    for i, b in enumerate(bullets):
        ld.text((24, yb + i * 28), "· " + b, fill=(203, 213, 225), font=ft_li)

    pill = (24, H - 88, 24 + 178, H - 54)
    ld.rounded_rectangle(pill, radius=10, fill=(22, 101, 52))
    ld.text((36, H - 82), "LOCAL BY DEFAULT", fill=(187, 247, 208), font=ft_badge)

    canvas.paste(left, (0, 0))

    # Right: "cover" zoom on main column — drop narrow sidebar, cap vertical slab so synthesis/UI scale up.
    x0 = min(int(mw * 0.19), 340)
    slab_h = min(mh, 2200)
    slab = master.crop((x0, 0, mw, slab_h))
    sw_, sh_ = slab.size
    scale = max(RIGHT_W / sw_, H / sh_)
    nw, nh = max(1, int(sw_ * scale)), max(1, int(sh_ * scale))
    big = slab.resize((nw, nh), Image.Resampling.LANCZOS)
    left_px = max(0, (nw - RIGHT_W) // 2)
    top_px = max(0, (nh - H) // 2)
    panel_rgb = big.crop((left_px, top_px, left_px + RIGHT_W, top_px + H))

    bd = ImageDraw.Draw(panel_rgb)
    bd.rounded_rectangle((1, 1, RIGHT_W - 2, H - 2), radius=radius, outline=(45, 212, 191), width=2)

    mask = _rounded_mask((RIGHT_W, H), radius)
    panel_rgba = panel_rgb.convert("RGBA")
    panel_rgba.putalpha(mask)

    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba.paste(panel_rgba, (LEFT_W, 0), panel_rgba)
    out = canvas_rgba.convert("RGB")

    fd = ImageDraw.Draw(out)
    for i in range(4):
        fd.line([(LEFT_W + i, 32), (LEFT_W + i, H - 32)], fill=(14 + i * 18, 116 + i * 8, 160 + i * 6), width=1)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_png, "PNG", optimize=True)


def write_plain_top_crop_thumbnail(src: Path, dst: Path, width: int = 1000, height: int = 750) -> None:
    """Simple 4:3 top-crop (legacy plain resize)."""
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

    ap = argparse.ArgumentParser(description="Build 1000×750 catalog thumbnail from dashboard PNG")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plain", action="store_true", help="Plain top-crop only (no branded rail)")
    ns = ap.parse_args()
    if ns.plain:
        write_plain_top_crop_thumbnail(ns.src, ns.out)
    else:
        render_catalog_thumbnail(ns.src, ns.out)
    print(f"Wrote {ns.out}")
