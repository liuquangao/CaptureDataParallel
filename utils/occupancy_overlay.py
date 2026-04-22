"""把 score_field / 候选点叠加到占用图上保存为 PNG。

与 `utils.occupancy_map` 的关注点分离:这里只做可视化。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from utils.occupancy_map import OccupancyMap


def _star_points(
    center_x: float, center_y: float, outer_radius: float, inner_radius: float
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(10):
        angle = -math.pi / 2.0 + index * math.pi / 5.0
        radius = outer_radius if index % 2 == 0 else inner_radius
        points.append(
            (
                float(center_x + radius * math.cos(angle)),
                float(center_y + radius * math.sin(angle)),
            )
        )
    return points


def _jet_color(score: float) -> tuple[int, int, int]:
    s = float(np.clip(score, 0.0, 1.0))
    r = np.clip(1.5 - abs(4.0 * s - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - abs(4.0 * s - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - abs(4.0 * s - 1.0), 0.0, 1.0)
    return (int(r * 255.0), int(g * 255.0), int(b * 255.0))


def _render_occupancy_base(occupancy_map: OccupancyMap) -> Image.Image:
    base = np.asarray(occupancy_map.data, dtype=np.uint8)
    rgb = np.stack([base, base, base], axis=-1)
    rgb[occupancy_map.free_mask] = np.array([240, 240, 240], dtype=np.uint8)
    rgb[occupancy_map.occupied_mask] = np.array([40, 40, 40], dtype=np.uint8)
    rgb[occupancy_map.unknown_mask] = np.array([150, 150, 150], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def save_score_field_overlay(
    occupancy_map: OccupancyMap,
    score_field: list,
    out_path: str | Path,
    person_position_xy: tuple[float, float] | None = None,
    selected_candidates: list | None = None,
) -> Path:
    """score_field 每项需有 .x, .y, .score;selected_candidates 每项需有 .x, .y。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = _render_occupancy_base(occupancy_map)
    draw = ImageDraw.Draw(image)

    for item in score_field:
        row, col = occupancy_map.world_to_grid(float(item.x), float(item.y))
        if not (0 <= row < occupancy_map.height and 0 <= col < occupancy_map.width):
            continue
        fill = _jet_color(float(item.score))
        radius = 2
        draw.ellipse((col - radius, row - radius, col + radius, row + radius), fill=fill, outline=fill)

    if selected_candidates is not None:
        for candidate in selected_candidates:
            row, col = occupancy_map.world_to_grid(float(candidate.x), float(candidate.y))
            if not (0 <= row < occupancy_map.height and 0 <= col < occupancy_map.width):
                continue
            draw.polygon(
                _star_points(float(col), float(row), outer_radius=4.0, inner_radius=1.8),
                fill=(64, 200, 64),
                outline=(255, 255, 255),
            )

    if person_position_xy is not None:
        person_x, person_y = person_position_xy
        row, col = occupancy_map.world_to_grid(float(person_x), float(person_y))
        if 0 <= row < occupancy_map.height and 0 <= col < occupancy_map.width:
            r = 3
            draw.ellipse((col - r, row - r, col + r, row + r), fill=(64, 96, 255), outline=(255, 255, 255))

    image.save(out_path)
    return out_path
