"""单视图 RGB/Depth/Instance-seg 的后处理 + 落盘。

所有函数都是纯 numpy/PIL,和 omni.* 完全解耦,可以在任何地方调用。

- `_compute_ground_mask`:把 depth 反投影到世界,保留 |z-floor|<=tol 且在
  occupancy.free_mask 内的像素。
- `_compute_score_map`:把 score_field 的点投影到像素,如果落在 ground_mask 上
  且深度一致,就把该点的 score 写到对应像素;同时输出 yaw_map 的 cos/sin 分量。
- `_instance_bbox_xyxy`:从 instance_id_segmentation 帧里按 prim path 前缀找到
  对应像素,返回 (u_min, v_min, u_max, v_max) 或 None。
- `_save_*`:把上述结果写盘为 PNG / NPY / JSON。
"""
from __future__ import annotations

import json
from math import atan2, cos, sin
from pathlib import Path

import numpy as np
from PIL import Image


def _yaw_to_world_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    half = float(yaw_rad) * 0.5
    return (float(cos(half)), 0.0, 0.0, float(sin(half)))


def _save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="RGB").save(path)


def _save_depth(depth_m: np.ndarray, png_path: Path, npy_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    depth_arr = np.asarray(depth_m, dtype=np.float32)
    depth_mm = np.clip(depth_arr * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(png_path)
    np.save(npy_path, depth_arr)


def _save_ground_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    Image.fromarray(mask_arr * np.uint8(255), mode="L").save(path)


def _save_score_map(score_map: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(score_map, dtype=np.float32))


def _save_valid_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(mask, dtype=bool))


def _save_yaw_map(yaw_map: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(yaw_map, dtype=np.float32))


def _save_person_bbox_norm(
    path: Path,
    bbox_xyxy: tuple[int, int, int, int] | None,
    image_width: int,
    image_height: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bbox_xyxy is None or image_width <= 0 or image_height <= 0:
        payload = {"xyxy_norm": None}
    else:
        u_min, v_min, u_max, v_max = bbox_xyxy
        payload = {
            "xyxy_norm": [
                float(u_min) / float(image_width),
                float(v_min) / float(image_height),
                float(u_max) / float(image_width),
                float(v_max) / float(image_height),
            ]
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compute_ground_mask(
    depth_m: np.ndarray,
    camera_position: tuple[float, float, float],
    camera_orientation_wxyz: tuple[float, float, float, float],
    resolution: tuple[int, int],
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    occupancy_map=None,
    floor_z: float = 0.0,
    ground_tolerance_m: float = 0.05,
) -> np.ndarray:
    width, height = int(resolution[0]), int(resolution[1])
    depth_arr = np.asarray(depth_m, dtype=np.float32)
    if depth_arr.shape[0] != height or depth_arr.shape[1] != width:
        raise ValueError(
            f"Depth shape {depth_arr.shape} does not match configured resolution {(width, height)}"
        )

    fx = width * float(focal_length) / max(float(horizontal_aperture), 1e-6)
    fy = height * float(focal_length) / max(float(vertical_aperture), 1e-6)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    u_coords = np.arange(width, dtype=np.float32)
    v_coords = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u_coords, v_coords)

    valid_depth = np.isfinite(depth_arr) & (depth_arr > 0.0)
    forward = np.where(valid_depth, depth_arr, 0.0).astype(np.float32, copy=False)
    lateral = -(uu - cx) * forward / max(fx, 1e-6)
    vertical = -(vv - cy) * forward / max(fy, 1e-6)

    yaw_rad = 2.0 * np.arctan2(float(camera_orientation_wxyz[3]), float(camera_orientation_wxyz[0]))
    cos_yaw = float(cos(float(yaw_rad)))
    sin_yaw = float(sin(float(yaw_rad)))

    world_x = np.float32(camera_position[0]) + forward * cos_yaw - lateral * sin_yaw
    world_y = np.float32(camera_position[1]) + forward * sin_yaw + lateral * cos_yaw
    world_z = np.float32(camera_position[2]) + vertical
    ground_mask = valid_depth & (np.abs(world_z - float(floor_z)) <= float(ground_tolerance_m))

    if occupancy_map is not None:
        occupancy_mask = np.zeros((height, width), dtype=bool)
        valid_rows, valid_cols = np.where(ground_mask)
        for row, col in zip(valid_rows.tolist(), valid_cols.tolist()):
            wx = float(world_x[row, col])
            wy = float(world_y[row, col])
            if not np.isfinite(wx) or not np.isfinite(wy):
                continue
            occ_row, occ_col = occupancy_map.world_to_grid(wx, wy)
            if not (0 <= occ_row < occupancy_map.height and 0 <= occ_col < occupancy_map.width):
                continue
            if bool(occupancy_map.free_mask[occ_row, occ_col]):
                occupancy_mask[row, col] = True
        ground_mask = ground_mask & occupancy_mask

    ground_output = np.zeros((height, width), dtype=np.uint8)
    ground_output[ground_mask] = 1
    return ground_output


def _project_world_point_to_pixel(
    world_xyz: tuple[float, float, float],
    camera_position: tuple[float, float, float],
    camera_orientation_wxyz: tuple[float, float, float, float],
    resolution: tuple[int, int],
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
) -> tuple[float, float, float] | None:
    yaw_rad = 2.0 * np.arctan2(float(camera_orientation_wxyz[3]), float(camera_orientation_wxyz[0]))
    dx = float(world_xyz[0]) - float(camera_position[0])
    dy = float(world_xyz[1]) - float(camera_position[1])
    dz = float(world_xyz[2]) - float(camera_position[2])

    cos_yaw = cos(float(yaw_rad))
    sin_yaw = sin(float(yaw_rad))
    forward = dx * cos_yaw + dy * sin_yaw
    lateral = -dx * sin_yaw + dy * cos_yaw
    vertical = dz
    if forward <= 1e-6:
        return None

    width, height = int(resolution[0]), int(resolution[1])
    fx = width * float(focal_length) / max(float(horizontal_aperture), 1e-6)
    fy = height * float(focal_length) / max(float(vertical_aperture), 1e-6)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    u = cx - fx * lateral / forward
    v = cy - fy * vertical / forward
    if not (0.0 <= u < float(width) and 0.0 <= v < float(height)):
        return None
    return float(u), float(v), float(forward)


def _compute_score_map(
    score_field,
    depth_m: np.ndarray,
    ground_mask: np.ndarray,
    camera_position: tuple[float, float, float],
    camera_orientation_wxyz: tuple[float, float, float, float],
    resolution: tuple[int, int],
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    depth_tolerance_m: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = int(resolution[0]), int(resolution[1])
    score_map = np.zeros((height, width), dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)
    yaw_map = np.zeros((2, height, width), dtype=np.float32)
    if score_field is None:
        return score_map, valid_mask, yaw_map

    depth_arr = np.asarray(depth_m, dtype=np.float32)
    if depth_arr.shape != (height, width):
        raise ValueError(f"Depth shape {depth_arr.shape} does not match score_map shape {(height, width)}")

    cam_yaw = 2.0 * np.arctan2(float(camera_orientation_wxyz[3]), float(camera_orientation_wxyz[0]))

    for item in score_field:
        projection = _project_world_point_to_pixel(
            world_xyz=(float(item.x), float(item.y), float(item.z)),
            camera_position=camera_position,
            camera_orientation_wxyz=camera_orientation_wxyz,
            resolution=resolution,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
        )
        if projection is None:
            continue

        u, v, point_depth = projection
        col = int(round(u))
        row = int(round(v))
        if not (0 <= row < height and 0 <= col < width):
            continue
        if int(ground_mask[row, col]) != 1:
            continue

        pixel_depth = float(depth_arr[row, col])
        if not np.isfinite(pixel_depth) or pixel_depth <= 0.0:
            continue
        if abs(pixel_depth - float(point_depth)) > float(depth_tolerance_m):
            continue

        delta_yaw = np.arctan2(
            np.sin(float(item.yaw_rad) - float(cam_yaw)),
            np.cos(float(item.yaw_rad) - float(cam_yaw)),
        )
        if (not valid_mask[row, col]) or (float(item.score) > float(score_map[row, col])):
            valid_mask[row, col] = True
            score_map[row, col] = float(item.score)
            yaw_map[0, row, col] = float(np.cos(delta_yaw))
            yaw_map[1, row, col] = float(np.sin(delta_yaw))

    return score_map, valid_mask, yaw_map


def _extract_instance_ids_by_path_prefix(segmentation_frame: dict, prim_path_prefix: str) -> set[int]:
    info = segmentation_frame.get("info", {}) if isinstance(segmentation_frame, dict) else {}
    id_to_labels = info.get("idToLabels", {}) if isinstance(info, dict) else {}
    prefix = str(prim_path_prefix).rstrip("/")
    if not prefix:
        return set()
    matching_ids: set[int] = set()
    for raw_id, label_info in id_to_labels.items():
        text = str(label_info)
        if text == prefix or text.startswith(prefix + "/"):
            matching_ids.add(int(raw_id))
    return matching_ids


def _instance_mask(segmentation_frame: dict, prim_path_prefix: str) -> np.ndarray | None:
    if not isinstance(segmentation_frame, dict) or "data" not in segmentation_frame:
        return None
    data = np.asarray(segmentation_frame["data"])
    if data.ndim == 3 and data.shape[-1] == 1:
        data = data[..., 0]
    label_ids = _extract_instance_ids_by_path_prefix(segmentation_frame, prim_path_prefix)
    if not label_ids:
        return np.zeros(data.shape, dtype=bool)
    return np.isin(data.astype(np.int64, copy=False), np.asarray(sorted(label_ids), dtype=np.int64))


def _instance_bbox_xyxy(segmentation_frame: dict, prim_path_prefix: str) -> tuple[int, int, int, int] | None:
    mask = _instance_mask(segmentation_frame, prim_path_prefix)
    if mask is None or not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    v_min, v_max = int(rows[0]), int(rows[-1])
    u_min, u_max = int(cols[0]), int(cols[-1])
    return (u_min, v_min, u_max, v_max)


def _semantic_class_mask(segmentation_frame: dict, class_name: str) -> np.ndarray | None:
    """从 semantic_segmentation 帧中按 class 名取像素 mask。

    ``idToLabels`` 结构是 {"0": {"class": "person"}, ...}。
    """
    if not isinstance(segmentation_frame, dict) or "data" not in segmentation_frame:
        return None
    data = np.asarray(segmentation_frame["data"])
    if data.ndim == 3 and data.shape[-1] == 1:
        data = data[..., 0]

    info = segmentation_frame.get("info", {})
    id_to_labels = info.get("idToLabels", {}) if isinstance(info, dict) else {}
    matching_ids = [
        int(raw_id)
        for raw_id, label_info in id_to_labels.items()
        if isinstance(label_info, dict) and label_info.get("class") == str(class_name)
    ]
    if not matching_ids:
        return np.zeros(data.shape, dtype=bool)
    return np.isin(data.astype(np.int64, copy=False), np.asarray(sorted(matching_ids), dtype=np.int64))


def _semantic_bbox_xyxy(segmentation_frame: dict, class_name: str) -> tuple[int, int, int, int] | None:
    mask = _semantic_class_mask(segmentation_frame, class_name)
    if mask is None or not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    v_min, v_max = int(rows[0]), int(rows[-1])
    u_min, u_max = int(cols[0]), int(cols[-1])
    return (u_min, v_min, u_max, v_max)


def _camera_yaw_from_person(
    camera_xy: tuple[float, float],
    person_xy: tuple[float, float],
) -> float:
    return float(atan2(float(person_xy[1]) - float(camera_xy[1]), float(person_xy[0]) - float(camera_xy[0])))
