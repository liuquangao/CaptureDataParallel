"""Overlay room polygons from structure.json using semantic-map coordinates."""

import argparse
import json
from pathlib import Path

from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def room_world_to_plot(profile):
    pts = np.asarray(profile, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("room profile must be an Nx2 array of world coordinates")
    return pts


def format_world_xy(x, y):
    return f"({x:.2f}, {y:.2f})"


def rasterize_polygon_like_semantic_builder(profile, h, w, scale, x_min, y_min):
    xys = np.asarray(profile, dtype=float)
    poly = MplPath(xys, closed=True)
    min_x_pixel = int(np.floor((np.min(xys[:, 0]) - x_min) / scale))
    max_x_pixel = int(np.floor((np.max(xys[:, 0]) - x_min) / scale))
    min_y_pixel = int(np.floor((np.min(xys[:, 1]) - y_min) / scale))
    max_y_pixel = int(np.floor((np.max(xys[:, 1]) - y_min) / scale))
    min_x_pixel = np.clip(min_x_pixel, 0, w - 1)
    max_x_pixel = np.clip(max_x_pixel, 0, w - 1)
    min_y_pixel = np.clip(min_y_pixel, 0, h - 1)
    max_y_pixel = np.clip(max_y_pixel, 0, h - 1)

    mask = np.zeros((h, w), dtype=bool)
    for j in range(min_x_pixel, max_x_pixel + 1):
        for i in range(min_y_pixel, max_y_pixel + 1):
            i_flip = h - 1 - i
            j_flip = w - 1 - j
            cx = x_min + (j + 0.5) * scale
            cy = y_min + (i + 0.5) * scale
            if poly.contains_point((cx, cy), radius=1e-9):
                mask[i_flip, j_flip] = True
    return mask


def mask_to_outline_segments(mask, scale, x_min, y_min):
    segments = []
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            x0 = x_min + x * scale
            x1 = x_min + (x + 1) * scale
            y0 = y_min + y * scale
            y1 = y_min + (y + 1) * scale
            if y == 0 or not mask[y - 1, x]:
                segments.append(((x0, y0), (x1, y0)))
            if y == h - 1 or not mask[y + 1, x]:
                segments.append(((x0, y1), (x1, y1)))
            if x == 0 or not mask[y, x - 1]:
                segments.append(((x0, y0), (x0, y1)))
            if x == w - 1 or not mask[y, x + 1]:
                segments.append(((x1, y0), (x1, y1)))
    return segments


def world_to_semantic_display_xy(points, scale, x_min, y_min, h, w):
    pts = np.asarray(points, dtype=float)
    px = np.floor((pts[:, 0] - x_min) / scale).astype(int)
    py = np.floor((pts[:, 1] - y_min) / scale).astype(int)
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    px_flip = w - 1 - px
    py_flip = h - 1 - py
    x_plot = x_min + (px_flip + 0.5) * scale
    y_plot = y_min + (py_flip + 0.5) * scale
    return np.stack([x_plot, y_plot], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scene",
        default="/home/leo/FusionLab/DataSets/spatialverse/InteriorGS/0110_840000",
    )
    ap.add_argument("--out", default="/home/leo/FusionLab/CaptureData/outputs/room_overlay.png")
    args = ap.parse_args()

    scene = Path(args.scene)
    occ_img = np.array(Image.open(scene / "occupancy.png"))
    occ_meta = json.loads((scene / "occupancy.json").read_text())
    struct = json.loads((scene / "structure.json").read_text())

    scale = occ_meta["scale"]
    x_min, y_min = occ_meta["min"][:2]
    H, W = occ_img.shape[:2]
    extent = [float(x_min), float(x_min) + W * scale, float(y_min), float(y_min) + H * scale]
    occ_plot = np.fliplr(np.flipud(occ_img))

    fig, ax = plt.subplots(figsize=(W / 40, H / 40), dpi=200)
    ax.imshow(
        occ_plot,
        cmap="gray",
        interpolation="nearest",
        origin="lower",
        extent=extent,
    )

    colors = ["#ff3b30", "#0a84ff", "#30d158", "#ff9f0a", "#bf5af2"]
    for i, room in enumerate(struct.get("rooms", [])):
        profile = room["profile"]
        pts_world = room_world_to_plot(profile)
        mask = rasterize_polygon_like_semantic_builder(pts_world, H, W, scale, x_min, y_min)
        pts_plot = world_to_semantic_display_xy(pts_world, scale, x_min, y_min, H, W)
        color = colors[i % len(colors)]
        rgba = np.zeros((H, W, 4), dtype=float)
        rgba[mask] = [1.0, 59 / 255, 48 / 255, 0.14]
        ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
        for (p0, p1) in mask_to_outline_segments(mask, scale, x_min, y_min):
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=color,
                linewidth=0.8,
            )
        ax.scatter(pts_plot[:, 0], pts_plot[:, 1], color=color, s=10, zorder=3)
        for (x_plot, y_plot), (x_world, y_world) in zip(pts_plot, pts_world):
            ax.annotate(
                format_world_xy(x_world, y_world),
                (x_plot, y_plot),
                xytext=(4, 4),
                textcoords="offset points",
                color=color,
                fontsize=6,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.75),
            )
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            continue
        cx = x_min + (xs.mean() + 0.5) * scale
        cy = y_min + (ys.mean() + 0.5) * scale
        ax.text(
            cx,
            cy,
            str(i + 1),
            color=color,
            fontsize=10,
            ha="center",
            va="center",
            weight="bold",
        )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=7, framealpha=0.8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
