#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path("/home/leo/FusionLab/AHO/CaptureDataParallel")
DEFAULT_POS_DIR = REPO_ROOT / "outputs_parallel-v2-raycasted/839874/pos_000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw one person's bbox and score_map over an RGB frame.")
    parser.add_argument("--pos-dir", type=Path, default=DEFAULT_POS_DIR)
    parser.add_argument("--frame", type=str, default="000")
    parser.add_argument("--person-id", choices=("person_000", "person_001"), default=None)
    return parser.parse_args()


def denormalize_bbox(bbox_norm: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_norm
    return (
        max(0, min(width - 1, round(float(x1) * width))),
        max(0, min(height - 1, round(float(y1) * height))),
        max(0, min(width - 1, round(float(x2) * width))),
        max(0, min(height - 1, round(float(y2) * height))),
    )


def load_bbox(json_path: Path, width: int, height: int) -> tuple[int, int, int, int] | None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    bbox = data.get("xyxy_norm")
    if bbox is None:
        return None
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Expected xyxy_norm with 4 values in {json_path}")
    return denormalize_bbox([float(v) for v in bbox], width, height)


def jet_color(score: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(score, dtype=np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * s - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * s - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * s - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def draw_overlay(pos_dir: Path, frame: str, person_id: str) -> Path:
    rgb_path = pos_dir / "rgb" / f"{frame}.png"
    score_map_path = pos_dir / "score_map" / person_id / f"{frame}.npy"
    bbox_path = pos_dir / "person_bbox" / person_id / f"{frame}.json"
    output_path = pos_dir / "rgb" / f"{frame}_score_map_bbox_{person_id}.png"

    image = Image.open(rgb_path).convert("RGB")
    width, height = image.size
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    score_map = np.load(score_map_path)
    if score_map.shape != (height, width):
        raise ValueError(f"Score map shape {score_map.shape} does not match image shape {(height, width)}")

    mask = np.isfinite(score_map) & (score_map > 0.0)
    alpha = np.zeros((height, width, 1), dtype=np.float32)
    alpha[mask, 0] = 0.65
    blended = rgb * (1.0 - alpha) + jet_color(score_map) * alpha
    output = Image.fromarray(np.clip(blended * 255.0, 0, 255).astype(np.uint8), mode="RGB")

    bbox = load_bbox(bbox_path, width, height)
    if bbox is not None:
        color = (255, 40, 40) if person_id == "person_000" else (40, 220, 80)
        draw = ImageDraw.Draw(output)
        draw.rectangle(bbox, outline=color, width=4)
        draw.text((bbox[0] + 4, max(0, bbox[1] - 18)), person_id, fill=color)

    output.save(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    person_ids = [args.person_id] if args.person_id else ["person_000", "person_001"]
    for person_id in person_ids:
        output_path = draw_overlay(args.pos_dir, args.frame, person_id)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
