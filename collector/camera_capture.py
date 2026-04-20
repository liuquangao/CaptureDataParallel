from __future__ import annotations

import json
from dataclasses import dataclass
from math import atan2, cos, sin
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image


@dataclass
class CameraCaptureConfig:
    prim_path: str
    resolution: tuple[int, int]
    camera_height: float
    visibility_settle_updates: int
    visibility_toggle_settle_updates: int
    focus_distance: float
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float
    near_clipping_distance: float
    depth_unit: str
    look_direction_mode: str
    target_position_xy: tuple[float, float] | None = None
    debug_logging: bool = False
    enable_rgb: bool = True
    enable_depth: bool = True
    enable_instance_segmentation: bool = True
    yaw_jitter_margin: float = 0.0


@dataclass
class CameraCaptureRecord:
    index: int
    camera_position: tuple[float, float, float]
    camera_orientation_wxyz: tuple[float, float, float, float]
    rgb_path: str
    depth_png_path: str
    depth_npy_path: str
    ground_mask_path: str
    score_map_path: str
    valid_mask_path: str
    yaw_map_path: str
    visibility_ratio: float
    visible_person_pixels: int
    total_person_pixels: int
    person_bbox_path: str
    depth_unit: str


def create_collector_camera(cfg: CameraCaptureConfig):
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom

    camera = Camera(
        prim_path=cfg.prim_path,
        name="collector_camera",
        resolution=cfg.resolution,
        position=np.array([0.0, 0.0, cfg.camera_height], dtype=np.float32),
        orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    camera.initialize()
    if cfg.enable_rgb:
        camera.add_rgb_to_frame()
    if cfg.enable_depth:
        camera.add_distance_to_image_plane_to_frame()
    if cfg.enable_instance_segmentation:
        camera.add_instance_id_segmentation_to_frame()
    camera.set_focal_length(cfg.focal_length)
    camera.set_horizontal_aperture(cfg.horizontal_aperture)
    camera.set_vertical_aperture(cfg.vertical_aperture)
    camera.set_focus_distance(cfg.focus_distance)

    camera_prim = camera.prim
    if camera_prim and camera_prim.IsValid():
        usd_camera = UsdGeom.Camera(camera_prim)
        usd_camera.GetClippingRangeAttr().Set((float(cfg.near_clipping_distance), 1000000.0))
    return camera


def _yaw_to_world_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    half = yaw_rad * 0.5
    return (float(cos(half)), 0.0, 0.0, float(sin(half)))


def _select_camera_orientation(candidate, cfg: CameraCaptureConfig) -> tuple[float, float, float, float]:
    if cfg.look_direction_mode == "look_at_target" and cfg.target_position_xy is not None:
        target_x, target_y = cfg.target_position_xy
        dx = float(target_x) - float(candidate.x)
        dy = float(target_y) - float(candidate.y)
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            yaw = atan2(dy, dx)
            return _yaw_to_world_quaternion(yaw)
    if getattr(candidate, "yaw_rad", None) is not None:
        return _yaw_to_world_quaternion(float(candidate.yaw_rad))
    return (1.0, 0.0, 0.0, 0.0)


def _save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_uint8 = np.asarray(rgb, dtype=np.uint8)
    Image.fromarray(rgb_uint8, mode="RGB").save(path)


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


def _save_valid_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(mask, dtype=bool))


def _save_yaw_map(yaw_map: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(yaw_map, dtype=np.float32))


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


def _apply_yaw_jitter(
    camera_position: tuple[float, float, float],
    base_orientation: tuple[float, float, float, float],
    cfg: CameraCaptureConfig,
    rng,
) -> tuple[float, float, float, float]:
    margin = float(cfg.yaw_jitter_margin)
    if margin <= 0.0:
        return base_orientation
    if cfg.target_position_xy is None:
        return base_orientation

    width = int(cfg.resolution[0])
    fx = width * float(cfg.focal_length) / max(float(cfg.horizontal_aperture), 1e-6)
    if fx <= 1e-6:
        return base_orientation

    usable_margin = min(max(margin, 0.0), 0.49)
    delta_max = float(np.arctan((0.5 - usable_margin) * float(width) / float(fx)))
    if delta_max <= 0.0:
        return base_orientation

    base_yaw = 2.0 * np.arctan2(float(base_orientation[3]), float(base_orientation[0]))
    jitter_rad = float(rng.uniform(-delta_max, delta_max))
    return _yaw_to_world_quaternion(float(base_yaw + jitter_rad))


def _set_prim_visibility(stage, prim_path: str, visible: bool, recursive: bool = False) -> bool:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False

    def _apply(imageable_prim) -> None:
        imageable = UsdGeom.Imageable(imageable_prim)
        if not imageable:
            return
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    _apply(prim)
    if recursive:
        for child in Usd.PrimRange(prim):
            if child == prim:
                continue
            _apply(child)
    return True


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


def _count_instance_pixels(segmentation_frame: dict, prim_path_prefix: str) -> int:
    mask = _instance_mask(segmentation_frame, prim_path_prefix)
    if mask is None:
        return 0
    return int(mask.sum())


def _instance_bbox_xyxy(segmentation_frame: dict, prim_path_prefix: str) -> tuple[int, int, int, int] | None:
    mask = _instance_mask(segmentation_frame, prim_path_prefix)
    if mask is None or not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    v_min, v_max = int(rows[0]), int(rows[-1])
    u_min, u_max = int(cols[0]), int(cols[-1])
    return (u_min, v_min, u_max, v_max)


def _set_camera_pose(camera, camera_position, camera_orientation) -> None:
    camera.set_world_pose(
        position=np.array(camera_position, dtype=np.float32),
        orientation=np.array(camera_orientation, dtype=np.float32),
        camera_axes="world",
    )


def _collect_instance_pixel_counts_for_poses(
    simulation_app,
    timeline,
    camera,
    poses: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]],
    visibility_settle_updates: int,
    person_prim_path: str,
) -> list[int]:
    counts: list[int] = []
    for camera_position, camera_orientation in poses:
        _set_camera_pose(camera, camera_position, camera_orientation)
        _render_warmup(simulation_app, timeline, visibility_settle_updates)
        segmentation_frame = camera.get_current_frame().get("instance_id_segmentation")
        counts.append(_count_instance_pixels(segmentation_frame, person_prim_path))
    return counts


def estimate_candidate_visibility_ratios_batch(
    simulation_app,
    timeline,
    camera,
    poses: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]],
    visibility_settle_updates: int,
    visibility_toggle_settle_updates: int,
    person_prim_path: str,
    scene_collision_exists: bool,
    geometry_root: str | None,
) -> list[tuple[float, int, int]]:
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    if scene_collision_exists:
        _set_prim_visibility(stage, "/World/scene_collision", visible=False, recursive=True)
    elif geometry_root:
        _set_prim_visibility(stage, geometry_root, visible=False, recursive=True)
    _render_warmup(simulation_app, timeline, visibility_toggle_settle_updates)
    total_person_pixels_list = _collect_instance_pixel_counts_for_poses(
        simulation_app=simulation_app,
        timeline=timeline,
        camera=camera,
        poses=poses,
        visibility_settle_updates=visibility_settle_updates,
        person_prim_path=person_prim_path,
    )

    if scene_collision_exists:
        _set_prim_visibility(stage, "/World/scene_collision", visible=True, recursive=True)
    elif geometry_root:
        _set_prim_visibility(stage, geometry_root, visible=True, recursive=True)
    _render_warmup(simulation_app, timeline, visibility_toggle_settle_updates)
    visible_person_pixels_list = _collect_instance_pixel_counts_for_poses(
        simulation_app=simulation_app,
        timeline=timeline,
        camera=camera,
        poses=poses,
        visibility_settle_updates=visibility_settle_updates,
        person_prim_path=person_prim_path,
    )

    if scene_collision_exists:
        _set_prim_visibility(stage, "/World/scene_collision", visible=False, recursive=True)
    elif geometry_root:
        _set_prim_visibility(stage, geometry_root, visible=False, recursive=True)
    _render_warmup(simulation_app, timeline, visibility_toggle_settle_updates)

    results: list[tuple[float, int, int]] = []
    for visible_person_pixels, total_person_pixels in zip(visible_person_pixels_list, total_person_pixels_list):
        visibility_ratio = float(visible_person_pixels) / float(total_person_pixels) if total_person_pixels > 0 else 0.0
        results.append((float(visibility_ratio), int(visible_person_pixels), int(total_person_pixels)))
    return results


def estimate_candidate_visibility_ratio(
    simulation_app,
    timeline,
    camera,
    camera_position: tuple[float, float, float],
    camera_orientation: tuple[float, float, float, float],
    visibility_settle_updates: int,
    visibility_toggle_settle_updates: int,
    person_prim_path: str,
    scene_collision_exists: bool,
    geometry_root: str | None,
) -> tuple[float, int, int]:
    return estimate_candidate_visibility_ratios_batch(
        simulation_app=simulation_app,
        timeline=timeline,
        camera=camera,
        poses=[(camera_position, camera_orientation)],
        visibility_settle_updates=visibility_settle_updates,
        visibility_toggle_settle_updates=visibility_toggle_settle_updates,
        person_prim_path=person_prim_path,
        scene_collision_exists=scene_collision_exists,
        geometry_root=geometry_root,
    )[0]


def _render_warmup(simulation_app, timeline, updates: int) -> None:
    timeline.play()
    for _ in range(updates):
        simulation_app.update()
    timeline.pause()


def _find_all_mesh_paths(stage) -> list[str]:
    mesh_paths: list[str] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            mesh_paths.append(str(prim.GetPath()))
    return mesh_paths


def _detect_visibility_layers(stage) -> dict:
    volume_root_exists = bool(stage.GetPrimAtPath("/World/volume") and stage.GetPrimAtPath("/World/volume").IsValid())
    scene_collision_exists = bool(
        stage.GetPrimAtPath("/World/scene_collision") and stage.GetPrimAtPath("/World/scene_collision").IsValid()
    )

    geometry_root = None
    for candidate in ("/World/volume/mesh", "/World/mesh", "/World/GroundPlane"):
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid():
            geometry_root = candidate
            break

    return {
        "volume_root_exists": volume_root_exists,
        "scene_collision_exists": scene_collision_exists,
        "geometry_root": geometry_root,
    }


def capture_candidate_views(
    simulation_app,
    timeline,
    camera,
    candidates,
    scene_dir: Path,
    cfg: CameraCaptureConfig,
    score_field=None,
    person_prim_path: str | None = None,
    occupancy_map=None,
) -> list[CameraCaptureRecord]:
    records: list[CameraCaptureRecord] = []
    import omni.usd
    import random

    stage = omni.usd.get_context().get_stage()
    layers = _detect_visibility_layers(stage)
    if cfg.debug_logging and layers["volume_root_exists"]:
        print("[Capture] NuRec volume root found at /World/volume.", flush=True)
    if cfg.debug_logging:
        all_mesh_paths = _find_all_mesh_paths(stage)
    else:
        all_mesh_paths = []
    if cfg.debug_logging and all_mesh_paths:
        print("[Capture] Mesh prims found in loaded stage:", flush=True)
        for path in all_mesh_paths:
            print(f"  - {path}", flush=True)
    if cfg.debug_logging and layers["geometry_root"]:
        print(f"[Capture] Geometry visibility root: {layers['geometry_root']}", flush=True)
    poses: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]] = []
    jitter_rng = random.Random(0)
    for candidate in candidates:
        camera_orientation = _select_camera_orientation(candidate, cfg)
        camera_position = (candidate.x, candidate.y, cfg.camera_height)
        camera_orientation = _apply_yaw_jitter(camera_position, camera_orientation, cfg, jitter_rng)
        poses.append((camera_position, camera_orientation))

    if layers["scene_collision_exists"]:
        if cfg.debug_logging:
            print("[Capture] scene_collision visibility toggling enabled.", flush=True)
    elif layers["geometry_root"]:
        if cfg.debug_logging:
            print(f"[Capture] RGB will hide {layers['geometry_root']}; depth will show {layers['geometry_root']}.", flush=True)
        if cfg.debug_logging and layers["volume_root_exists"]:
            print("[Capture] NuRec volume remains visible during RGB; depth capture can be compared against mesh-backed geometry.", flush=True)
    else:
        if cfg.debug_logging:
            print("[Capture] No known geometry visibility root found; using same visibility for RGB and depth.", flush=True)

    person_path = str(person_prim_path) if person_prim_path is not None else ""
    visibility_results = estimate_candidate_visibility_ratios_batch(
        simulation_app=simulation_app,
        timeline=timeline,
        camera=camera,
        poses=poses,
        visibility_settle_updates=cfg.visibility_settle_updates,
        visibility_toggle_settle_updates=cfg.visibility_toggle_settle_updates,
        person_prim_path=person_path,
        scene_collision_exists=bool(layers["scene_collision_exists"]),
        geometry_root=layers["geometry_root"],
    )

    # Batch RGB first, then batch depth, so capture ordering stays deterministic.
    rgb_frames: list[np.ndarray] = []
    if layers["scene_collision_exists"]:
        _set_prim_visibility(stage, "/World/scene_collision", visible=False, recursive=True)
    elif layers["geometry_root"]:
        _set_prim_visibility(stage, layers["geometry_root"], visible=False, recursive=True)
    _render_warmup(simulation_app, timeline, cfg.visibility_toggle_settle_updates)
    for camera_position, camera_orientation in poses:
        _set_camera_pose(camera, camera_position, camera_orientation)
        _render_warmup(simulation_app, timeline, cfg.visibility_settle_updates)
        rgb = camera.get_rgb()
        if rgb is None:
            raise RuntimeError("RGB capture returned None during batch capture")
        rgb_frames.append(np.asarray(rgb))

    depth_frames: list[np.ndarray] = []
    person_bboxes: list[tuple[int, int, int, int] | None] = []
    if layers["scene_collision_exists"]:
        _set_prim_visibility(stage, "/World/scene_collision", visible=True, recursive=True)
    elif layers["geometry_root"]:
        _set_prim_visibility(stage, layers["geometry_root"], visible=True, recursive=True)
    _render_warmup(simulation_app, timeline, cfg.visibility_toggle_settle_updates)
    for camera_position, camera_orientation in poses:
        _set_camera_pose(camera, camera_position, camera_orientation)
        _render_warmup(simulation_app, timeline, cfg.visibility_settle_updates)
        depth = camera.get_depth()
        if depth is None:
            raise RuntimeError("Depth capture returned None during batch capture")
        depth_frames.append(np.asarray(depth))
        segmentation_frame = camera.get_current_frame().get("instance_id_segmentation")
        person_bboxes.append(_instance_bbox_xyxy(segmentation_frame, person_path))

    if layers["scene_collision_exists"]:
        _set_prim_visibility(stage, "/World/scene_collision", visible=False, recursive=True)
    elif layers["geometry_root"]:
        _set_prim_visibility(stage, layers["geometry_root"], visible=False, recursive=True)
    _render_warmup(simulation_app, timeline, cfg.visibility_toggle_settle_updates)

    for idx, (candidate, pose, visibility_result, rgb, depth, person_bbox) in enumerate(
        zip(candidates, poses, visibility_results, rgb_frames, depth_frames, person_bboxes)
    ):
        camera_position, camera_orientation = pose
        visibility_ratio, visible_person_pixels, total_person_pixels = visibility_result

        file_stem = f"{idx:03d}"
        rgb_path = scene_dir / "rgb" / f"{file_stem}.png"
        depth_png_path = scene_dir / "depth" / f"{file_stem}.png"
        depth_npy_path = scene_dir / "depth" / f"{file_stem}.npy"
        ground_mask_path = scene_dir / "ground_mask" / f"{file_stem}.png"
        score_map_path = scene_dir / "score_map" / f"{file_stem}.npy"
        valid_mask_path = scene_dir / "valid_mask" / f"{file_stem}.npy"
        yaw_map_path = scene_dir / "yaw_map" / f"{file_stem}.npy"
        person_bbox_path = scene_dir / "person_bbox" / f"{file_stem}.json"
        _save_rgb(rgb, rgb_path)
        _save_depth(depth, depth_png_path, depth_npy_path)
        ground_mask = _compute_ground_mask(
            depth_m=depth,
            camera_position=camera_position,
            camera_orientation_wxyz=camera_orientation,
            resolution=cfg.resolution,
            focal_length=cfg.focal_length,
            horizontal_aperture=cfg.horizontal_aperture,
            vertical_aperture=cfg.vertical_aperture,
            occupancy_map=occupancy_map,
        )
        _save_ground_mask(
            ground_mask,
            ground_mask_path,
        )
        score_map, valid_mask, yaw_map = _compute_score_map(
            score_field=score_field,
            depth_m=depth,
            ground_mask=ground_mask,
            camera_position=camera_position,
            camera_orientation_wxyz=camera_orientation,
            resolution=cfg.resolution,
            focal_length=cfg.focal_length,
            horizontal_aperture=cfg.horizontal_aperture,
            vertical_aperture=cfg.vertical_aperture,
        )
        _save_score_map(
            score_map,
            score_map_path,
        )
        _save_valid_mask(valid_mask, valid_mask_path)
        _save_yaw_map(yaw_map, yaw_map_path)
        _save_person_bbox_norm(
            person_bbox_path,
            person_bbox,
            image_width=int(cfg.resolution[0]),
            image_height=int(cfg.resolution[1]),
        )

        record = CameraCaptureRecord(
            index=idx,
            camera_position=tuple(float(v) for v in camera_position),
            camera_orientation_wxyz=camera_orientation,
            rgb_path=str(rgb_path),
            depth_png_path=str(depth_png_path),
            depth_npy_path=str(depth_npy_path),
            ground_mask_path=str(ground_mask_path),
            score_map_path=str(score_map_path),
            valid_mask_path=str(valid_mask_path),
            yaw_map_path=str(yaw_map_path),
            visibility_ratio=float(visibility_ratio),
            visible_person_pixels=int(visible_person_pixels),
            total_person_pixels=int(total_person_pixels),
            person_bbox_path=str(person_bbox_path),
            depth_unit=cfg.depth_unit,
        )
        records.append(record)

    return records
