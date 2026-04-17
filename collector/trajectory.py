from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trajectory2D:
    path: Path
    xy: np.ndarray
    timestamps: np.ndarray


def load_training_trajectory(path: str | Path) -> Trajectory2D:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training trajectory file does not exist: {path}")

    timestamps: list[float] = []
    xy_points: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            timestamps.append(float(parts[0]))
            xy_points.append((float(parts[1]), float(parts[2])))

    if not xy_points:
        raise RuntimeError(f"No valid trajectory points found in: {path}")

    xy = np.asarray(xy_points, dtype=np.float32)
    timestamps_np = np.asarray(timestamps, dtype=np.float64)
    return Trajectory2D(path=path, xy=xy, timestamps=timestamps_np)


def compute_nearest_trajectory_info(
    trajectory: Trajectory2D,
    x: float,
    y: float,
) -> tuple[int, float, tuple[float, float]]:
    diffs = trajectory.xy - np.array([x, y], dtype=np.float32)
    dists_sq = np.sum(diffs * diffs, axis=1)
    nearest_idx = int(np.argmin(dists_sq))
    nearest_dist = float(np.sqrt(dists_sq[nearest_idx]))

    prev_idx = max(nearest_idx - 1, 0)
    next_idx = min(nearest_idx + 1, len(trajectory.xy) - 1)
    tangent = trajectory.xy[next_idx] - trajectory.xy[prev_idx]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-6:
        tangent_xy = (1.0, 0.0)
    else:
        tangent_xy = (float(tangent[0] / tangent_norm), float(tangent[1] / tangent_norm))
    return nearest_idx, nearest_dist, tangent_xy
