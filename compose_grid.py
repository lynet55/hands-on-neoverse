"""
Stage 3 of the keyframe-grid figure pipeline (no GPU needed).

Reads the per-frame PNGs + metadata.json produced by Stage 1 (NeoVerse) and
Stage 2 (HaMeR) and lays them out as a single comparison figure:

    rows    = the 4 keyframes (top -> bottom = time)
    columns = Ground Truth | Naive Gaussian | Gaussian Params | HaMeR skeleton

Column headers, per-row frame labels, per-cell mIoU (for the two NeoVerse model
columns) and a colour legend are drawn on. Mirrors the velocity-head grid layout
from the poster / the NeoVerse-paper deformation sequence panels.

Usage:
    python compose_grid.py --grid_dir outputs/grid_clip-001100
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLUMNS = [
    ("gt",       "Ground Truth"),
    ("naive",    "Naive Gaussian"),
    ("gsparam",  "Gaussian Params"),
    ("skeleton", "HaMeR skeleton"),
]
# (label, RGB) legend entries shared across the figure.
LEGEND = [
    ("right hand", (255, 60, 60)),
    ("left hand",  (60, 120, 255)),
    ("object",     (60, 220, 60)),
]


def load_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def text_centered(draw, cx, y, text, font, fill=(0, 0, 0)):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (r - l) / 2, y), text, font=font, fill=fill)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_dir", required=True)
    parser.add_argument("--output", default=None,
                        help="Output PNG. Defaults to <grid_dir>/grid_<clip>.png")
    parser.add_argument("--gap", type=int, default=6)
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    meta = json.loads((grid_dir / "metadata.json").read_text())
    frames = meta["frames"]
    W, H = meta["width"], meta["height"]
    per_frame = meta.get("per_frame", {})
    has_skeleton = (grid_dir / f"skeleton_{frames[0]}.png").exists()

    gap = args.gap
    left_margin = 70          # row labels
    top_header = 44           # column headers
    title_h = 40              # title bar
    legend_h = 40             # bottom legend
    cell_label_h = 22         # per-cell mIoU strip on model columns

    ncols = len(COLUMNS)
    nrows = len(frames)
    canvas_w = left_margin + ncols * W + (ncols - 1) * gap
    canvas_h = (title_h + top_header
                + nrows * (H + cell_label_h) + (nrows - 1) * gap
                + legend_h)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_title = load_font(22)
    f_head = load_font(18)
    f_row = load_font(15)
    f_small = load_font(13)

    # ---- title ----
    title = f"Segmentation over time - {meta['clip']} ({meta['stream']})"
    text_centered(draw, canvas_w / 2, 8, title, f_title)

    # ---- column headers ----
    for c, (_, label) in enumerate(COLUMNS):
        cx = left_margin + c * (W + gap) + W / 2
        text_centered(draw, cx, title_h + 12, label, f_head)

    # ---- grid cells ----
    y0 = title_h + top_header
    for r, fi in enumerate(frames):
        row_top = y0 + r * (H + cell_label_h + gap)
        # row label (rotated-free, left margin)
        draw.text((6, row_top + H / 2 - 10), f"frame", font=f_row, fill=(0, 0, 0))
        draw.text((6, row_top + H / 2 + 6), f"{fi}", font=f_row, fill=(0, 0, 0))

        pf = per_frame.get(str(fi), {})
        for c, (key, _) in enumerate(COLUMNS):
            cx = left_margin + c * (W + gap)
            img_path = grid_dir / f"{key}_{fi}.png"
            if img_path.exists():
                cell = Image.open(img_path).convert("RGB").resize((W, H))
            else:
                cell = Image.new("RGB", (W, H), (40, 40, 40))
                text_centered(draw, cx + W / 2, row_top + H / 2,
                              "missing", f_small, fill=(255, 255, 255))
            canvas.paste(cell, (cx, row_top))
            draw.rectangle([cx, row_top, cx + W - 1, row_top + H - 1],
                           outline=(180, 180, 180))

            # per-cell mIoU for the two NeoVerse model columns
            note = ""
            if key == "naive" and "naive_miou" in pf:
                note = f"mIoU = {pf['naive_miou']:.2f}"
            elif key == "gsparam" and "gsparam_miou" in pf:
                note = f"mIoU = {pf['gsparam_miou']:.2f}"
            elif key == "skeleton":
                if not has_skeleton:
                    note = "(no HaMeR output)"
                elif "hamer_miou" in pf:
                    note = f"mIoU = {pf['hamer_miou']:.2f}"
            if note:
                text_centered(draw, cx + W / 2, row_top + H + 3, note, f_small)

    # ---- legend ----
    ly = canvas_h - legend_h + 10
    lx = left_margin
    for label, color in LEGEND:
        draw.rectangle([lx, ly, lx + 16, ly + 16], fill=color)
        draw.text((lx + 22, ly), label, font=f_small, fill=(0, 0, 0))
        lx += 22 + 10 + draw.textlength(label, font=f_small) + 24
    draw.text((lx, ly), "HaMeR: hand mask + joints (darker red = right, darker blue = left)",
              font=f_small, fill=(0, 0, 0))

    out_path = Path(args.output) if args.output else grid_dir / f"grid_{meta['clip']}.png"
    canvas.save(out_path)
    print(f"Saved grid figure: {out_path}  ({canvas_w}x{canvas_h})")


if __name__ == "__main__":
    main()
