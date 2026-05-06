#!/usr/bin/env python3
"""
Visualize raycast score_map + reid_score_maps + reference crops for one frame.

Layout (one PNG per person):

  Row 0 │ RGB+bbox │ Raycast overlay │ ref_00 crop │ ref_01 crop │ … │ ref_07 crop │
  Row 1 │  (blank) │     (blank)     │ reid map 00 │ reid map 01 │ … │ reid map 07 │

All cells are CELL×CELL pixels. Reference crops are letterboxed (black bars).
Reid maps use jet colormap, same as raycast.

Usage:
    python debug/debug_reid_scoremap.py [--pos-dir <path>] [--frame 000]
                                        [--person-id person_000]
                                        [--reference-dir <path>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POS_DIR = REPO_ROOT / "outputs_parallel-v2-raycasted/841917/pos_000"
DEFAULT_REF_DIR = REPO_ROOT / "outputs_parallel-v2-raycasted/reference"

PERSON_TO_CHAR = {
    "person_000": "F_Business_02",
    "person_001": "male_adult_medical_01",
}
NUM_REFS = 8
CELL = 256   # all grid cells are CELL×CELL pixels
LABEL_H = 20  # height of text label above each cell


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pos-dir",      type=Path, default=DEFAULT_POS_DIR)
    p.add_argument("--frame",        type=str,  default="000")
    p.add_argument("--person-id",    choices=("person_000", "person_001"), default=None)
    p.add_argument("--reference-dir",type=Path, default=DEFAULT_REF_DIR)
    return p.parse_args()


# ── Drawing helpers ───────────────────────────────────────────────────────────

def jet_color(score: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(score, dtype=np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * s - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * s - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * s - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def score_overlay(rgb_arr: np.ndarray, score_map: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """Blend jet colormap onto RGB where score_map > 0. Returns uint8 H×W×3."""
    rgb = np.asarray(rgb_arr, dtype=np.float32) / 255.0
    mask = np.isfinite(score_map) & (score_map > 0.0)
    a = np.zeros((*score_map.shape, 1), dtype=np.float32)
    a[mask, 0] = alpha
    blended = rgb * (1.0 - a) + jet_color(score_map) * a
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Resize image to fit in size×size with black bars (letterbox)."""
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    x = (size - img.width)  // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_cell(arr: np.ndarray) -> Image.Image:
    """Convert H×W×3 uint8 array to a CELL×CELL PIL image."""
    img = Image.fromarray(arr, mode="RGB")
    if img.size != (CELL, CELL):
        img = img.resize((CELL, CELL), Image.LANCZOS)
    return img


def labeled(img: Image.Image, text: str,
            bg: tuple = (30, 30, 30), fg: tuple = (220, 220, 220)) -> Image.Image:
    """Add a small label bar above the image."""
    canvas = Image.new("RGB", (img.width, img.height + LABEL_H), bg)
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), text, fill=fg)
    canvas.paste(img, (0, LABEL_H))
    return canvas


def load_bbox(path: Path, w: int, h: int) -> tuple[int, int, int, int] | None:
    data = json.loads(path.read_text())
    xyxy = data.get("xyxy_norm")
    if not xyxy:
        return None
    x1 = max(0, min(w - 1, round(xyxy[0] * w)))
    y1 = max(0, min(h - 1, round(xyxy[1] * h)))
    x2 = max(0, min(w - 1, round(xyxy[2] * w)))
    y2 = max(0, min(h - 1, round(xyxy[3] * h)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


# ── Per-person panel ──────────────────────────────────────────────────────────

def build_panel(pos_dir: Path, frame: str, person_id: str,
                reference_dir: Path) -> Image.Image:
    """
    Build a 2-row panel:
      Row 0: [RGB+bbox] [Raycast] [ref_00]..[ref_07]
      Row 1: [blank   ] [blank  ] [reid_00]..[reid_07]
    """
    rgb_arr = np.asarray(Image.open(pos_dir / "rgb" / f"{frame}.png").convert("RGB"))
    H, W = rgb_arr.shape[:2]

    # ── Raycast score map overlay ────────────────────────────────────────────
    raycast_map = np.load(pos_dir / "score_map" / person_id / f"{frame}.npy")
    raycast_cell = make_cell(score_overlay(rgb_arr, raycast_map))

    # ── RGB + bbox ───────────────────────────────────────────────────────────
    rgb_img = Image.fromarray(rgb_arr)
    bbox_path = pos_dir / "person_bbox" / person_id / f"{frame}.json"
    if bbox_path.exists():
        bbox = load_bbox(bbox_path, W, H)
        if bbox:
            color = (255, 60, 60) if person_id == "person_000" else (60, 220, 80)
            draw = ImageDraw.Draw(rgb_img)
            draw.rectangle(bbox, outline=color, width=3)
    rgb_cell = make_cell(np.asarray(rgb_img))

    # ── Reference crops ──────────────────────────────────────────────────────
    char_name = PERSON_TO_CHAR[person_id]
    ref_dir = reference_dir / char_name
    ref_cells: list[Image.Image] = []
    for j in range(NUM_REFS):
        ref_path = ref_dir / f"ref_{j:02d}.png"
        if ref_path.exists():
            ref_cells.append(letterbox(Image.open(ref_path).convert("RGB"), CELL))
        else:
            ref_cells.append(Image.new("RGB", (CELL, CELL), (40, 40, 40)))

    # ── ReID score map overlays ───────────────────────────────────────────────
    reid_cells: list[Image.Image] = []
    reid_map_base = pos_dir / "reid_score_map" / person_id
    for j in range(NUM_REFS):
        reid_path = reid_map_base / f"ref_{j:02d}" / f"{frame}.npy"
        if reid_path.exists():
            reid_map = np.load(reid_path)
            cell = make_cell(score_overlay(rgb_arr, reid_map))
        else:
            cell = Image.new("RGB", (CELL, CELL), (20, 20, 20))
        reid_cells.append(cell)

    # ── Std map across 8 reid maps ───────────────────────────────────────────
    valid_reid_maps = [
        np.load(reid_map_base / f"ref_{j:02d}" / f"{frame}.npy")
        for j in range(NUM_REFS)
        if (reid_map_base / f"ref_{j:02d}" / f"{frame}.npy").exists()
    ]
    if len(valid_reid_maps) > 1:
        stack = np.stack(valid_reid_maps, axis=0)      # (8, H, W)
        std_map  = np.std(stack, axis=0)               # (H, W)
        mean_map = np.mean(stack, axis=0)              # (H, W)
        # Normalise std to [0,1] for visualisation (relative to mean range)
        std_norm = std_map / (std_map.max() + 1e-8)
        std_cell  = make_cell(score_overlay(np.zeros_like(rgb_arr), std_norm, alpha=0.9))
        mean_cell = make_cell(score_overlay(rgb_arr, mean_map))
        active = (stack > 0).any(axis=0)
        n_active = int(active.sum())
        mean_std = float(std_map[active].mean()) if n_active > 0 else 0.0
        std_label  = f"std across refs  (mean={mean_std:.3f})"
        mean_label = "mean reid map"
    else:
        std_cell   = Image.new("RGB", (CELL, CELL), (40, 40, 40))
        mean_cell  = Image.new("RGB", (CELL, CELL), (40, 40, 40))
        std_label  = "std (no data)"
        mean_label = "mean (no data)"

    # ── Assemble grid ────────────────────────────────────────────────────────
    # Row 0: RGB+bbox | Raycast  | ref_00..ref_07
    # Row 1: mean map | std map  | reid_00..reid_07
    COLS = 2 + NUM_REFS
    cell_h = CELL + LABEL_H

    row0_cells = (
        [labeled(rgb_cell,     "RGB + bbox")]
        + [labeled(raycast_cell, "Raycast score")]
        + [labeled(ref_cells[j], f"ref_{j:02d}") for j in range(NUM_REFS)]
    )
    row1_cells = (
        [labeled(mean_cell, mean_label)]
        + [labeled(std_cell,  std_label)]
        + [labeled(reid_cells[j], f"reid map {j:02d}") for j in range(NUM_REFS)]
    )

    panel_w = COLS * CELL
    panel_h = 2 * cell_h
    panel = Image.new("RGB", (panel_w, panel_h), (10, 10, 10))

    for col, img in enumerate(row0_cells):
        panel.paste(img, (col * CELL, 0))
    for col, img in enumerate(row1_cells):
        panel.paste(img, (col * CELL, cell_h))

    # Title bar
    title_h = 28
    full = Image.new("RGB", (panel_w, panel_h + title_h), (10, 10, 10))
    draw = ImageDraw.Draw(full)
    title = (f"{pos_dir.parent.name}/{pos_dir.name}  frame={frame}  {person_id} "
             f"({char_name})")
    draw.text((8, 6), title, fill=(255, 220, 80))
    full.paste(panel, (0, title_h))
    return full


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    person_ids = [args.person_id] if args.person_id else ["person_000", "person_001"]

    for person_id in person_ids:
        panel = build_panel(args.pos_dir, args.frame, person_id, args.reference_dir)
        out = args.pos_dir / "rgb" / f"{args.frame}_reid_debug_{person_id}.png"
        panel.save(out)
        print(f"Saved {out}  ({panel.width}×{panel.height})")


if __name__ == "__main__":
    main()
