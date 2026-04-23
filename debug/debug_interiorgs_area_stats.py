#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize InteriorGS free-area distribution.")
    parser.add_argument(
        "--interiorgs-root",
        type=Path,
        default=Path("/home/leo/FusionLab/DataSets/spatialverse/InteriorGS"),
        help="Path to the InteriorGS scene root.",
    )
    parser.add_argument(
        "--max-free-ratio",
        type=float,
        default=0.75,
        help="Skip scenes whose free ratio exceeds this threshold.",
    )
    return parser.parse_args()


def compute_scene_areas(scene_dir: Path) -> dict:
    occupancy_json_path = scene_dir / "occupancy.json"
    occupancy_png_path = scene_dir / "occupancy.png"
    structure_json_path = scene_dir / "structure.json"

    with occupancy_json_path.open("r", encoding="utf-8") as f:
        occupancy_meta = json.load(f)

    occupancy_data = np.array(Image.open(occupancy_png_path).convert("L"))
    free_mask = occupancy_data == 255
    scale = float(occupancy_meta["scale"])
    total_cells = int(occupancy_data.shape[0] * occupancy_data.shape[1])
    free_cells = int(free_mask.sum())
    free_ratio = float(free_cells) / float(total_cells) if total_cells > 0 else 0.0
    free_area_m2 = float(free_cells) * scale ** 2
    room_free_area_m2 = free_area_m2

    if structure_json_path.exists():
        structure = json.loads(structure_json_path.read_text(encoding="utf-8"))
        height, width = int(occupancy_data.shape[0]), int(occupancy_data.shape[1])
        lower_meta = occupancy_meta.get("lower", occupancy_meta.get("min"))
        lower = (float(lower_meta[0]), float(lower_meta[1]))
        upper_meta = occupancy_meta.get("upper")
        if upper_meta is None:
            upper = (lower[0] + width * scale, lower[1] + height * scale)
        else:
            upper = (float(upper_meta[0]), float(upper_meta[1]))

        room_mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(room_mask_image)
        for room in structure.get("rooms", []):
            profile = room.get("profile", []) if isinstance(room, dict) else []
            if len(profile) < 3:
                continue
            profile_arr = np.asarray(profile, dtype=np.float64)
            pixel_x = (float(upper[0]) - profile_arr[:, 0]) / float(scale)
            pixel_y = (profile_arr[:, 1] - float(lower[1])) / float(scale)
            polygon = [(float(x), float(y)) for x, y in np.stack([pixel_x, pixel_y], axis=1)[:, :2]]
            draw.polygon(polygon, fill=255)

        room_mask = np.array(room_mask_image, dtype=np.uint8) > 0
        room_free_area_m2 = float(np.logical_and(room_mask, free_mask).sum()) * scale ** 2

    return {
        "scene": scene_dir.name,
        "free_ratio": free_ratio,
        "free_area_m2": free_area_m2,
        "room_free_area_m2": room_free_area_m2,
    }


def summarize(values: list[float]) -> dict | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def print_histogram(values: list[float], bins: list[float]) -> None:
    for idx in range(len(bins) - 1):
        lo = bins[idx]
        hi = bins[idx + 1]
        if idx == len(bins) - 2:
            count = sum(1 for value in values if lo <= value <= hi)
            bracket = "]"
        else:
            count = sum(1 for value in values if lo <= value < hi)
            bracket = ")"
        print(f"  [{lo}, {hi}{bracket}: {count}")


def main() -> None:
    args = parse_args()
    rows = []
    for scene_dir in sorted(path for path in args.interiorgs_root.iterdir() if path.is_dir()):
        if not (scene_dir / "occupancy.json").exists() or not (scene_dir / "occupancy.png").exists():
            continue
        rows.append(compute_scene_areas(scene_dir))

    kept_rows = [row for row in rows if row["free_ratio"] <= float(args.max_free_ratio)]
    free_area_values = [float(row["free_area_m2"]) for row in kept_rows]
    room_free_area_values = [float(row["room_free_area_m2"]) for row in kept_rows]

    print(f"InteriorGS scenes: {len(rows)}")
    print(f"Kept after max_free_ratio<={args.max_free_ratio}: {len(kept_rows)}")
    print(f"Filtered out: {len(rows) - len(kept_rows)}")
    print("")
    print("free_area_m2 summary")
    print(json.dumps(summarize(free_area_values), indent=2))
    print("")
    print("room_free_area_m2 summary")
    print(json.dumps(summarize(room_free_area_values), indent=2))
    print("")
    print("room_free_area_m2 histogram")
    print_histogram(room_free_area_values, [0, 20, 40, 60, 80, 100, 150, 200, 300, 500])
    print("")
    print("Suggested area -> num_positions mapping")
    print("  room_free_area_m2 < 25  -> 1")
    print("  room_free_area_m2 < 50  -> 2")
    print("  room_free_area_m2 < 80  -> 3")
    print("  room_free_area_m2 < 120 -> 4")
    print("  room_free_area_m2 < 180 -> 5")
    print("  otherwise               -> 6")


if __name__ == "__main__":
    main()
