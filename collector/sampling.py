from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

from collector.debug_viz import raycast_ground


@dataclass
class GroundCandidate:
    x: float
    y: float
    z: float
    hit_path: str | None
    yaw_rad: float | None = None


def _far_enough(
    accepted: list[GroundCandidate],
    x: float,
    y: float,
    min_spacing: float,
) -> bool:
    min_spacing_sq = min_spacing * min_spacing
    for item in accepted:
        dx = item.x - x
        dy = item.y - y
        if dx * dx + dy * dy < min_spacing_sq:
            return False
    return True


def sample_valid_ground_candidates(
    occupancy_map,
    num_candidates: int,
    min_spacing: float,
    max_trials: int,
    seed: int = 0,
) -> list[GroundCandidate]:
    rng = random.Random(seed)
    accepted: list[GroundCandidate] = []

    for _ in range(max_trials):
        if len(accepted) >= num_candidates:
            break
        x, y = occupancy_map.sample_free_world_point(rng)
        if not _far_enough(accepted, x, y, min_spacing):
            continue
        hit, hit_z, hit_path = raycast_ground(x, y)
        if not hit or hit_z is None:
            continue
        accepted.append(
            GroundCandidate(
                x=float(x),
                y=float(y),
                z=float(hit_z),
                hit_path=str(hit_path) if hit_path is not None else None,
                yaw_rad=float(rng.uniform(-math.pi, math.pi)),
            )
        )
    return accepted


def save_ground_candidates(path: Path, candidates: list[GroundCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in candidates]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
