from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import atan2, cos, sin
from pathlib import Path
import random

from collector.sampling import GroundCandidate


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


def _iter_annulus_xy(person_x: float, person_y: float, min_radius_m: float, max_radius_m: float, grid_step_m: float):
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


def generate_segmentation_score_field(
    occupancy_map,
    person_position_xy: tuple[float, float],
    camera_height_m: float,
    min_radius_m: float,
    max_radius_m: float,
    grid_step_m: float,
    visibility_batch_scorer,
    occupancy_full_visibility_width_m: float = 0.25,
    min_camera_obstacle_distance_m: float = 0.0,
) -> list[ScoreFieldPoint]:
    samples = list(
        iter_ring_camera_samples(
            occupancy_map=occupancy_map,
            person_position_xy=person_position_xy,
            camera_height_m=camera_height_m,
            min_radius_m=min_radius_m,
            max_radius_m=max_radius_m,
            grid_step_m=grid_step_m,
            min_obstacle_distance_m=min_camera_obstacle_distance_m,
        )
    )
    if not samples:
        return []

    field: list[ScoreFieldPoint] = []
    segmentation_samples: list[dict] = []
    segmentation_indices: list[int] = []
    for index, sample in enumerate(samples):
        if _has_full_width_occupancy_visibility(
            occupancy_map=occupancy_map,
            person_position_xy=person_position_xy,
            camera_position_xy=(float(sample["x"]), float(sample["y"])),
            body_width_m=float(occupancy_full_visibility_width_m),
        ):
            field.append(
                ScoreFieldPoint(
                    x=sample["x"],
                    y=sample["y"],
                    z=sample["z"],
                    camera_z=sample["camera_z"],
                    yaw_rad=sample["yaw_rad"],
                    score=1.0,
                    distance_m=sample["distance_m"],
                    visible_person_pixels=1,
                    total_person_pixels=1,
                    scoring_mode="occupancy_full_visibility",
                )
            )
        else:
            field.append(None)
            segmentation_samples.append(sample)
            segmentation_indices.append(index)

    visibility_results = visibility_batch_scorer(segmentation_samples) if segmentation_samples else []
    for field_index, sample, result in zip(segmentation_indices, segmentation_samples, visibility_results):
        visibility_ratio, visible_person_pixels, total_person_pixels = result
        field[field_index] = (
            ScoreFieldPoint(
                x=sample["x"],
                y=sample["y"],
                z=sample["z"],
                camera_z=sample["camera_z"],
                yaw_rad=sample["yaw_rad"],
                score=float(visibility_ratio),
                distance_m=sample["distance_m"],
                visible_person_pixels=int(visible_person_pixels),
                total_person_pixels=int(total_person_pixels),
                scoring_mode="segmentation_visibility",
            )
        )
    return [item for item in field if item is not None]


def save_score_field(path: Path, score_field: list[ScoreFieldPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in score_field]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def select_capture_candidates(
    score_field: list[ScoreFieldPoint],
    score_min: float,
    score_max: float,
    seed: int = 0,
    max_candidates: int | None = None,
    fallback_to_nearest: bool = True,
) -> list[GroundCandidate]:
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

    return [
        GroundCandidate(
            x=item.x,
            y=item.y,
            z=item.z,
            hit_path=None,
            yaw_rad=item.yaw_rad,
        )
        for item in eligible
    ]
