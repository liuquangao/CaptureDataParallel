"""Overlay a saved score_map `.npy` on top of an RGB image."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay a score_map `.npy` heatmap on top of an RGB image."
    )
    parser.add_argument(
        "--rgb",
        type=Path,
        default=Path("/home/leo/FusionLab/CaptureData/outputs/839875/pos_000/rgb/000.png"),
        help="Path to the RGB image.",
    )
    parser.add_argument(
        "--score-map",
        type=Path,
        default=Path("/home/leo/FusionLab/CaptureData/outputs/839875/pos_000/score_map/000.npy"),
        help="Path to the score_map `.npy` file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/leo/FusionLab/CaptureData/outputs/839875/pos_000/score_map_overlay_000.png"),
        help="Path to save the overlay image.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Overlay alpha in [0, 1].",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="turbo",
        help="Matplotlib colormap name for the score heatmap.",
    )
    parser.add_argument(
        "--hide-zero",
        action="store_true",
        help="Hide zero-score pixels instead of drawing the full colormap.",
    )
    return parser.parse_args()


def load_inputs(rgb_path: Path, score_map_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image does not exist: {rgb_path}")
    if not score_map_path.exists():
        raise FileNotFoundError(f"score_map file does not exist: {score_map_path}")

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    score_map = np.asarray(np.load(score_map_path), dtype=np.float32)

    if score_map.ndim != 2:
        raise ValueError(f"Expected score_map to be 2D, got shape {score_map.shape}")
    if rgb.shape[:2] != score_map.shape:
        raise ValueError(
            f"RGB shape {rgb.shape[:2]} does not match score_map shape {score_map.shape}"
        )
    return rgb, score_map


def main() -> None:
    args = parse_args()
    rgb, score_map = load_inputs(args.rgb, args.score_map)

    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    vmin = 0.0
    vmax = float(np.max(score_map)) if np.any(np.isfinite(score_map)) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    score_to_draw = score_map.copy()
    if args.hide_zero:
        score_to_draw = np.ma.masked_where(score_map <= 0.0, score_map)

    height, width = score_map.shape
    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax.imshow(rgb)
    heat = ax.imshow(score_to_draw, cmap=args.cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("score", rotation=270, labelpad=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
