"""InteriorGS 占用图加载与坐标系转换。

只负责 map 本体(像素数据、free/occupied/unknown mask、世界-栅格转换、
基于 room polygon 的采样约束)。绘图 overlay 请见 utils/occupancy_overlay.py。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class OccupancyMap:
    image_path: Path
    yaml_path: Path
    resolution: float
    origin_xy: tuple[float, float]
    width: int
    height: int
    data: np.ndarray
    free_mask: np.ndarray
    occupied_mask: np.ndarray
    unknown_mask: np.ndarray
    room_free_mask: np.ndarray | None = None
    flip_x: bool = True
    flip_y: bool = True
    negate_xy: bool = True

    @property
    def max_x(self) -> float:
        return float(self.origin_xy[0] + self.width * self.resolution)

    @property
    def max_y(self) -> float:
        return float(self.origin_xy[1] + self.height * self.resolution)

    def _map_to_isaac_xy(self, x: float, y: float) -> tuple[float, float]:
        if self.flip_x:
            x = (self.origin_xy[0] + self.max_x) - x
        if self.flip_y:
            y = (self.origin_xy[1] + self.max_y) - y
        if self.negate_xy:
            x = -x
            y = -y
        return float(x), float(y)

    def _isaac_to_map_xy(self, x: float, y: float) -> tuple[float, float]:
        if self.negate_xy:
            x = -x
            y = -y
        if self.flip_x:
            x = (self.origin_xy[0] + self.max_x) - x
        if self.flip_y:
            y = (self.origin_xy[1] + self.max_y) - y
        return float(x), float(y)

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        x_map = self.origin_xy[0] + (col + 0.5) * self.resolution
        y_map = self.origin_xy[1] + (self.height - row - 0.5) * self.resolution
        return self._map_to_isaac_xy(float(x_map), float(y_map))

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        x_map, y_map = self._isaac_to_map_xy(float(x), float(y))
        col = int((x_map - self.origin_xy[0]) / self.resolution)
        row_from_bottom = int((y_map - self.origin_xy[1]) / self.resolution)
        row = self.height - row_from_bottom - 1
        return row, col

    def _disk_offsets(self, clearance_m: float) -> list[tuple[int, int]]:
        radius_cells = int(math.ceil(float(clearance_m) / max(float(self.resolution), 1e-6)))
        offsets: list[tuple[int, int]] = []
        for d_row in range(-radius_cells, radius_cells + 1):
            for d_col in range(-radius_cells, radius_cells + 1):
                distance_m = ((float(d_row) * self.resolution) ** 2 + (float(d_col) * self.resolution) ** 2) ** 0.5
                if distance_m <= float(clearance_m) + 1e-6:
                    offsets.append((d_row, d_col))
        return offsets

    def _is_cell_clear_of_obstacles(self, row: int, col: int, clearance_m: float) -> bool:
        if clearance_m <= 0.0:
            return bool(self.free_mask[row, col])
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        if not bool(self.free_mask[row, col]):
            return False
        for d_row, d_col in self._disk_offsets(clearance_m):
            n_row = int(row) + int(d_row)
            n_col = int(col) + int(d_col)
            if not (0 <= n_row < self.height and 0 <= n_col < self.width):
                return False
            if not bool(self.free_mask[n_row, n_col]):
                return False
        return True

    def sample_free_world_point(self, rng: random.Random | None = None) -> tuple[float, float]:
        rng = rng or random
        free_cells = np.argwhere(self.free_mask)
        if free_cells.size == 0:
            raise RuntimeError("Occupancy map contains no free cells")
        indices = list(range(len(free_cells)))
        rng.shuffle(indices)
        for idx in indices:
            row, col = free_cells[idx]
            return self.grid_to_world(int(row), int(col))
        raise RuntimeError("Occupancy map contains no free cells")

    def sample_room_free_world_point(self, rng: random.Random | None = None) -> tuple[float, float]:
        rng = rng or random
        if self.room_free_mask is None:
            return self.sample_free_world_point(rng)

        room_free_cells = np.argwhere(self.room_free_mask)
        if room_free_cells.size == 0:
            raise RuntimeError("Occupancy map contains no free cells inside room polygons")

        indices = list(range(len(room_free_cells)))
        rng.shuffle(indices)
        for idx in indices:
            row, col = room_free_cells[idx]
            return self.grid_to_world(int(row), int(col))
        raise RuntimeError("Occupancy map contains no free cells inside room polygons")

    def sample_room_free_world_point_with_constraints(
        self,
        rng: random.Random | None = None,
        min_obstacle_distance_m: float = 0.0,
        existing_world_points_xy: list[tuple[float, float]] | None = None,
        min_point_distance_m: float = 0.0,
    ) -> tuple[float, float]:
        rng = rng or random
        base_mask = self.room_free_mask if self.room_free_mask is not None else self.free_mask
        candidate_cells = np.argwhere(base_mask)
        if candidate_cells.size == 0:
            raise RuntimeError("Occupancy map contains no eligible room-free cells")

        indices = list(range(len(candidate_cells)))
        rng.shuffle(indices)
        existing_world_points_xy = existing_world_points_xy or []
        min_point_distance_m = float(min_point_distance_m)

        for idx in indices:
            row, col = candidate_cells[idx]
            row_i, col_i = int(row), int(col)
            if not self._is_cell_clear_of_obstacles(row_i, col_i, float(min_obstacle_distance_m)):
                continue
            world_xy = self.grid_to_world(row_i, col_i)
            if existing_world_points_xy:
                too_close = False
                for other_x, other_y in existing_world_points_xy:
                    dx = float(world_xy[0]) - float(other_x)
                    dy = float(world_xy[1]) - float(other_y)
                    if (dx * dx + dy * dy) ** 0.5 < min_point_distance_m:
                        too_close = True
                        break
                if too_close:
                    continue
            return world_xy

        raise RuntimeError(
            "Occupancy map could not find a room-free point satisfying obstacle clearance and position spacing"
        )


@dataclass
class OccupancySummary:
    yaml_path: str
    image_path: str
    resolution: float
    origin_xy: tuple[float, float]
    width: int
    height: int
    free_cells: int
    occupied_cells: int
    unknown_cells: int
    sampled_world_points: list[tuple[float, float]]


def _extract_scene_id_from_stage_url(stage_url: str | Path) -> str:
    scene_id = Path(stage_url).stem
    if not scene_id:
        raise ValueError(f"Could not extract scene_id from stage_url: {stage_url}")
    return scene_id


def _resolve_interiorgs_scene_dir(scene_cfg: dict) -> Path:
    explicit = scene_cfg.get("interiorgs_scene_dir")
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"InteriorGS scene directory does not exist: {path}")
        return path

    stage_url = scene_cfg.get("stage_url")
    if not stage_url:
        raise ValueError("Missing required config field: stage_url")
    scene_id = _extract_scene_id_from_stage_url(stage_url)

    root = Path(scene_cfg.get("interiorgs_root", "/home/leo/FusionLab/DataSets/spatialverse/InteriorGS"))
    if not root.exists():
        raise FileNotFoundError(f"InteriorGS root does not exist: {root}")

    matches = sorted(p for p in root.glob(f"*_{scene_id}") if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"Could not find InteriorGS scene dir matching '*_{scene_id}' under {root}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple InteriorGS scene dirs for scene_id={scene_id}: {[str(p) for p in matches]}"
        )
    return matches[0]


def _world_to_occupancy_pixel(
    xy,
    scale: float,
    lower: tuple[float, float],
    upper: tuple[float, float],
) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected Nx2 room profile, got shape {points.shape}")

    px = (float(upper[0]) - points[:, 0]) / float(scale)
    py = (points[:, 1] - float(lower[1])) / float(scale)
    return np.stack([px, py], axis=1)


def _build_interiorgs_room_free_mask(
    structure_path: Path,
    occupancy_meta: dict,
    occupancy_shape: tuple[int, int],
    free_mask: np.ndarray,
) -> np.ndarray:
    with structure_path.open("r", encoding="utf-8") as f:
        structure = json.load(f)

    height, width = int(occupancy_shape[0]), int(occupancy_shape[1])
    scale = float(occupancy_meta["scale"])
    lower_meta = occupancy_meta.get("lower", occupancy_meta.get("min"))
    if lower_meta is None:
        raise KeyError("occupancy.json missing both 'lower' and 'min'")
    lower = (float(lower_meta[0]), float(lower_meta[1]))

    upper_meta = occupancy_meta.get("upper")
    if upper_meta is None:
        upper = (lower[0] + width * scale, lower[1] + height * scale)
    else:
        upper = (float(upper_meta[0]), float(upper_meta[1]))

    room_mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(room_mask_img)

    for room in structure.get("rooms", []):
        profile = room.get("profile", []) if isinstance(room, dict) else []
        if len(profile) < 3:
            continue
        pts = _world_to_occupancy_pixel(profile, scale=scale, lower=lower, upper=upper)
        polygon = [(float(x), float(y)) for x, y in pts[:, :2]]
        draw.polygon(polygon, fill=255)

    room_mask = np.array(room_mask_img, dtype=np.uint8) > 0
    return np.logical_and(room_mask, free_mask)


def load_interiorgs_occupancy_map(scene_cfg: dict) -> OccupancyMap:
    scene_dir = _resolve_interiorgs_scene_dir(scene_cfg)
    json_path = scene_dir / "occupancy.json"
    image_path = scene_dir / "occupancy.png"
    if not json_path.exists():
        raise FileNotFoundError(f"InteriorGS occupancy.json does not exist: {json_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"InteriorGS occupancy.png does not exist: {image_path}")

    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    data = np.array(Image.open(image_path).convert("L"))
    if data.ndim != 2:
        raise ValueError(f"Expected grayscale occupancy image, got shape {data.shape}")

    free_mask = data == 255
    occupied_mask = data == 0
    unknown_mask = data == 127

    occupancy_map = OccupancyMap(
        image_path=image_path,
        yaml_path=json_path,
        resolution=float(meta["scale"]),
        origin_xy=(float(meta["min"][0]), float(meta["min"][1])),
        width=int(data.shape[1]),
        height=int(data.shape[0]),
        data=data,
        free_mask=free_mask,
        occupied_mask=occupied_mask,
        unknown_mask=unknown_mask,
        flip_x=bool(scene_cfg.get("occupancy_flip_x", True)),
        flip_y=bool(scene_cfg.get("occupancy_flip_y", True)),
        negate_xy=bool(scene_cfg.get("occupancy_negate_xy", True)),
    )
    structure_path = scene_dir / "structure.json"
    if structure_path.exists():
        occupancy_map.room_free_mask = _build_interiorgs_room_free_mask(
            structure_path=structure_path,
            occupancy_meta=meta,
            occupancy_shape=data.shape,
            free_mask=free_mask,
        )
    return occupancy_map


def summarize_occupancy_map(occupancy_map: OccupancyMap, num_samples: int = 5) -> OccupancySummary:
    rng = random.Random(0)
    samples = [occupancy_map.sample_free_world_point(rng) for _ in range(num_samples)]
    return OccupancySummary(
        yaml_path=str(occupancy_map.yaml_path),
        image_path=str(occupancy_map.image_path),
        resolution=occupancy_map.resolution,
        origin_xy=occupancy_map.origin_xy,
        width=occupancy_map.width,
        height=occupancy_map.height,
        free_cells=int(occupancy_map.free_mask.sum()),
        occupied_cells=int(occupancy_map.occupied_mask.sum()),
        unknown_cells=int(occupancy_map.unknown_mask.sum()),
        sampled_world_points=samples,
    )
