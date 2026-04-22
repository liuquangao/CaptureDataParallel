#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np


REPO_ROOT = Path("/home/leo/FusionLab/CaptureData")
YAW_MAP_PATH = REPO_ROOT / "outputs/839873/pos_000/yaw_map/001.npy"
IMAGE_PATH = REPO_ROOT / "outputs/839873/pos_000/rgb/001.png"
OVERLAY_OUTPUT_PATH = REPO_ROOT / "outputs/839873/pos_000/rgb/001_debug_yaw_overlay.png"
MAP_OUTPUT_PATH = REPO_ROOT / "outputs/839873/pos_000/rgb/001_debug_yaw_map.png"
ARROW_LENGTH = 20
RING_RADIUS = 12


def load_yaw_map(yaw_map_path: Path) -> np.ndarray:
    yaw_map = np.load(yaw_map_path)
    if yaw_map.ndim != 3 or yaw_map.shape[0] != 2:
        raise ValueError(f"Expected yaw map with shape (2, H, W), got {yaw_map.shape}")
    return yaw_map.astype(np.float32)


def get_nonzero_points(yaw_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.any(yaw_map != 0, axis=0)
    ys, xs = np.where(mask)
    return xs, ys


def draw_with_pillow(image_path: Path, yaw_map: np.ndarray) -> None:
    from PIL import Image, ImageColor, ImageDraw

    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    if yaw_map.shape[1:] != (height, width):
        raise ValueError(f"Yaw map shape {yaw_map.shape[1:]} does not match image size {(height, width)}")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    map_vis = Image.new("RGB", image.size, (0, 0, 0))
    map_draw = ImageDraw.Draw(map_vis)

    xs, ys = get_nonzero_points(yaw_map)
    for x, y in zip(xs.tolist(), ys.tolist()):
        cos_delta = float(yaw_map[0, y, x])
        sin_delta = float(yaw_map[1, y, x])
        delta_yaw = float(np.arctan2(sin_delta, cos_delta))
        end_x, end_y = compute_arrow_endpoint(x, y, delta_yaw)
        color = angle_to_rgb(delta_yaw)

        draw_gauge(draw, x, y, end_x, end_y, (*color, 220), ImageColor.getrgb("white"))
        draw_gauge(map_draw, x, y, end_x, end_y, color, ImageColor.getrgb("white"))

    merged = Image.alpha_composite(image, overlay).convert("RGB")
    merged.save(OVERLAY_OUTPUT_PATH)
    map_vis.save(MAP_OUTPUT_PATH)


def compute_arrow_endpoint(x: int, y: int, delta_yaw: float) -> tuple[int, int]:
    # Compass: delta_yaw=0 → up (facing person), +δ → left turn, -δ → right turn.
    end_x = round(x - ARROW_LENGTH * np.sin(delta_yaw))
    end_y = round(y - ARROW_LENGTH * np.cos(delta_yaw))
    return end_x, end_y


def draw_gauge(draw, x: int, y: int, end_x: int, end_y: int, color, point_color) -> None:
    draw.ellipse((x - RING_RADIUS, y - RING_RADIUS, x + RING_RADIUS, y + RING_RADIUS), outline=color, width=2)
    draw.line((x, y, end_x, end_y), fill=color, width=3)
    draw_arrow_head(draw, x, y, end_x, end_y, color)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=point_color)


def draw_arrow_head(draw, start_x: int, start_y: int, end_x: int, end_y: int, color) -> None:
    dx = end_x - start_x
    dy = end_y - start_y
    norm = float(np.hypot(dx, dy))
    if norm == 0.0:
        return

    ux = dx / norm
    uy = dy / norm
    px = -uy
    py = ux
    head_length = 8
    head_width = 5

    left_x = end_x - head_length * ux + head_width * px
    left_y = end_y - head_length * uy + head_width * py
    right_x = end_x - head_length * ux - head_width * px
    right_y = end_y - head_length * uy - head_width * py

    draw.polygon(
        [(end_x, end_y), (round(left_x), round(left_y)), (round(right_x), round(right_y))],
        fill=color,
    )


def angle_to_rgb(angle_rad: float) -> tuple[int, int, int]:
    r = int(255 * (np.cos(angle_rad) * 0.5 + 0.5))
    g = int(255 * (np.sin(angle_rad) * 0.5 + 0.5))
    b = 180
    return r, g, b


def main() -> None:
    if not YAW_MAP_PATH.exists():
        raise FileNotFoundError(f"Missing yaw map: {YAW_MAP_PATH}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Missing image: {IMAGE_PATH}")

    yaw_map = load_yaw_map(YAW_MAP_PATH)
    draw_with_pillow(IMAGE_PATH, yaw_map)

    xs, ys = get_nonzero_points(yaw_map)
    print(f"Saved overlay image to: {OVERLAY_OUTPUT_PATH}")
    print(f"Saved yaw map visualization to: {MAP_OUTPUT_PATH}")
    print(f"Non-zero yaw vectors: {len(xs)}")


if __name__ == "__main__":
    main()
