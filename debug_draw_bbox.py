#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path("/home/leo/FusionLab/CaptureData")
JSON_PATH = REPO_ROOT / "outputs/839873/pos_000/person_bbox/001.json"
IMAGE_PATH = REPO_ROOT / "outputs/839873/pos_000/rgb/001.png"
OUTPUT_PATH = REPO_ROOT / "outputs/839873/pos_000/rgb/001_debug_bbox.png"


def load_bbox(json_path: Path) -> list[float]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    bbox = data.get("xyxy_norm")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Expected 'xyxy_norm' with 4 values in {json_path}")
    return [float(v) for v in bbox]


def draw_with_pillow(image_path: Path, bbox_norm: list[float], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = denormalize_bbox(bbox_norm, width, height)

    draw = ImageDraw.Draw(image)
    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=4)
    image.save(output_path)


def draw_with_cv2(image_path: Path, bbox_norm: list[float], output_path: Path) -> None:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    height, width = image.shape[:2]
    x1, y1, x2, y2 = denormalize_bbox(bbox_norm, width, height)

    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=4)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise ValueError(f"Failed to write output image: {output_path}")


def denormalize_bbox(bbox_norm: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_norm
    x1_px = max(0, min(width - 1, round(x1 * width)))
    y1_px = max(0, min(height - 1, round(y1 * height)))
    x2_px = max(0, min(width - 1, round(x2 * width)))
    y2_px = max(0, min(height - 1, round(y2 * height)))
    return x1_px, y1_px, x2_px, y2_px


def main() -> None:
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Missing JSON file: {JSON_PATH}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Missing image file: {IMAGE_PATH}")

    bbox_norm = load_bbox(JSON_PATH)

    try:
        draw_with_pillow(IMAGE_PATH, bbox_norm, OUTPUT_PATH)
        backend = "Pillow"
    except ModuleNotFoundError:
        draw_with_cv2(IMAGE_PATH, bbox_norm, OUTPUT_PATH)
        backend = "OpenCV"

    print(f"Saved debug image to: {OUTPUT_PATH}")
    print(f"Backend: {backend}")


if __name__ == "__main__":
    main()
