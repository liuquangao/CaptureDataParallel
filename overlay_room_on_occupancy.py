"""Overlay room polygons from structure.json onto occupancy.png."""

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def world_to_pixel(xy, scale, lower, upper):
    wx = np.asarray(xy)[:, 0]
    wy = np.asarray(xy)[:, 1]
    px = (upper[0] - wx) / scale  # reverse X
    py = (wy - lower[1]) / scale  # keep Y
    return np.stack([px, py], axis=1)


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
    lower = occ_meta["lower"]
    upper = occ_meta["upper"]
    H, W = occ_img.shape[:2]

    fig, ax = plt.subplots(figsize=(W / 40, H / 40), dpi=200)
    ax.imshow(occ_img, cmap="gray", interpolation="nearest")

    colors = ["#ff3b30", "#0a84ff", "#30d158", "#ff9f0a", "#bf5af2"]
    for i, room in enumerate(struct.get("rooms", [])):
        profile = room["profile"]
        pts = world_to_pixel(profile, scale, lower, upper)
        poly = mpatches.Polygon(
            pts,
            closed=True,
            fill=False,
            edgecolor=colors[i % len(colors)],
            linewidth=1.5,
            label=f"room {i + 1}",
        )
        ax.add_patch(poly)
        cx, cy = pts.mean(axis=0)
        ax.text(
            cx,
            cy,
            str(i + 1),
            color=colors[i % len(colors)],
            fontsize=10,
            ha="center",
            va="center",
            weight="bold",
        )

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=7, framealpha=0.8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
