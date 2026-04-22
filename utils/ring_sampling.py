"""围绕人物的环形相机候选点采样 + 基于占用图的可见性短路判定。

- `iter_ring_camera_samples`:在 [min_radius, max_radius] 环形区域内按栅格步长
  枚举 free + 与障碍距离 >= clearance 的候选相机点。
- `_has_full_width_occupancy_visibility`:用占用图判断"人的全身宽度左右两端
  到相机的连线是否完全无遮挡"——若是,可以直接给分 1.0,不再渲 seg。
- `select_capture_candidates`:从 score_field 中按分数带挑选最终渲染候选。
- `ScoreFieldPoint` / `GroundCandidate` 数据类。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2
import random


@dataclass
class ScoreFieldPoint:
    x: float
    y: float
    z: float
    camera_z: float
    yaw_rad: float
    score: float
    distance_m: float
    visible_person_pixels: int = 0
    total_person_pixels: int = 0
    scoring_mode: str = "segmentation_visibility"


@dataclass
class GroundCandidate:
    x: float
    y: float
    z: float
    hit_path: str | None
    yaw_rad: float | None = None


def _iter_grid_line_cells(row0: int, col0: int, row1: int, col1: int):
    d_row = abs(int(row1) - int(row0))
    d_col = abs(int(col1) - int(col0))
    step_row = 1 if int(row0) < int(row1) else -1
    step_col = 1 if int(col0) < int(col1) else -1
    err = d_col - d_row
    row, col = int(row0), int(col0)

    while True:
        yield row, col
        if row == int(row1) and col == int(col1):
            break
        err2 = 2 * err
        if err2 > -d_row:
            err -= d_row
            col += step_col
        if err2 < d_col:
            err += d_col
            row += step_row


def _is_free_line_in_occupancy(
    occupancy_map,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
) -> bool:
    start_row, start_col = occupancy_map.world_to_grid(float(start_xy[0]), float(start_xy[1]))
    end_row, end_col = occupancy_map.world_to_grid(float(end_xy[0]), float(end_xy[1]))
    if not (0 <= start_row < occupancy_map.height and 0 <= start_col < occupancy_map.width):
        return False
    if not (0 <= end_row < occupancy_map.height and 0 <= end_col < occupancy_map.width):
        return False

    for row, col in _iter_grid_line_cells(start_row, start_col, end_row, end_col):
        if not (0 <= row < occupancy_map.height and 0 <= col < occupancy_map.width):
            return False
        if not bool(occupancy_map.free_mask[row, col]):
            return False
    return True


def _has_full_width_occupancy_visibility(
    occupancy_map,
    person_position_xy: tuple[float, float],
    camera_position_xy: tuple[float, float],
    body_width_m: float,
) -> bool:
    person_x, person_y = float(person_position_xy[0]), float(person_position_xy[1])
    camera_x, camera_y = float(camera_position_xy[0]), float(camera_position_xy[1])
    dx = camera_x - person_x
    dy = camera_y - person_y
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 1e-6:
        return False

    half_width = 0.5 * float(body_width_m)
    perp_x = -dy / norm
    perp_y = dx / norm
    left_person_xy = (person_x + perp_x * half_width, person_y + perp_y * half_width)
    right_person_xy = (person_x - perp_x * half_width, person_y - perp_y * half_width)

    return (
        _is_free_line_in_occupancy(occupancy_map, left_person_xy, (camera_x, camera_y))
        and _is_free_line_in_occupancy(occupancy_map, right_person_xy, (camera_x, camera_y))
    )


def _iter_annulus_xy(
    person_x: float,
    person_y: float,
    min_radius_m: float,
    max_radius_m: float,
    grid_step_m: float,
):
    import numpy as np

    half = float(max_radius_m) + float(grid_step_m)
    xs = np.arange(float(person_x) - half, float(person_x) + half + 1e-6, float(grid_step_m))
    ys = np.arange(float(person_y) - half, float(person_y) + half + 1e-6, float(grid_step_m))
    for x in xs:
        for y in ys:
            dx = float(x) - float(person_x)
            dy = float(y) - float(person_y)
            distance_m = (dx * dx + dy * dy) ** 0.5
            if float(min_radius_m) <= distance_m <= float(max_radius_m):
                yield float(x), float(y), float(distance_m)


def iter_ring_camera_samples(
    occupancy_map,
    person_position_xy: tuple[float, float],
    camera_height_m: float,
    min_radius_m: float,
    max_radius_m: float,
    grid_step_m: float,
    min_obstacle_distance_m: float = 0.0,
):
    person_x, person_y = person_position_xy
    visited_cells: set[tuple[int, int]] = set()

    for x, y, distance_m in _iter_annulus_xy(person_x, person_y, min_radius_m, max_radius_m, grid_step_m):
        row, col = occupancy_map.world_to_grid(x, y)
        if not (0 <= row < occupancy_map.height and 0 <= col < occupancy_map.width):
            continue
        if not bool(occupancy_map.free_mask[row, col]):
            continue
        if occupancy_map.room_free_mask is not None and not bool(occupancy_map.room_free_mask[row, col]):
            continue
        if not occupancy_map._is_cell_clear_of_obstacles(int(row), int(col), float(min_obstacle_distance_m)):
            continue
        if (row, col) in visited_cells:
            continue
        visited_cells.add((row, col))

        yaw_rad = atan2(float(person_y) - float(y), float(person_x) - float(x))
        yield {
            "x": float(x),
            "y": float(y),
            "z": 0.0,
            "camera_z": float(camera_height_m),
            "yaw_rad": float(yaw_rad),
            "distance_m": float(distance_m),
        }


def select_capture_candidates(
    score_field: list[ScoreFieldPoint],
    score_min: float,
    score_max: float,
    seed: int = 0,
    max_candidates: int | None = None,
    fallback_to_nearest: bool = True,
) -> list[ScoreFieldPoint]:
    eligible = [item for item in score_field if float(score_min) <= item.score <= float(score_max)]
    used_fallback = False
    if not eligible and fallback_to_nearest:
        used_fallback = True
        positive = [item for item in score_field if float(item.score) > 0.0]
        pool = positive if positive else list(score_field)
        target_score = 0.5 * (float(score_min) + float(score_max))
        eligible = sorted(
            pool,
            key=lambda item: (
                abs(float(item.score) - target_score),
                abs(float(item.score) - float(score_max)),
                abs(float(item.score) - float(score_min)),
            ),
        )
    if not used_fallback:
        rng = random.Random(seed)
        rng.shuffle(eligible)
    if max_candidates is not None:
        eligible = eligible[: int(max_candidates)]
    return list(eligible)


# 保持和旧 collector.score_field.save_score_field 接口一致,方便未来替换。
def save_score_field(path, score_field: list[ScoreFieldPoint]) -> None:
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in score_field]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
