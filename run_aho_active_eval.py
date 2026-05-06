# Copyright (c) 2026, Fusion Intelligence Labs, University of Exeter. All rights reserved.
#
# ActiveHumanObservation active inference evaluation (two-person):
#   place person_000 + person_001 -> shared before view -> AHO score map per target
#   -> pixel backprojection -> after view per target.

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


from utils.config_io import load_config
from utils.occupancy_map import load_interiorgs_occupancy_map
from utils.scene_selection import check_scene_filter, resolve_scene_configs


_REP_DEFAULT_HORIZONTAL_APERTURE = 20.955
_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AHO active inference evaluator (two-person)")
    parser.add_argument("--config", type=str, default="configs/aho_active_eval.yaml")
    return parser.parse_args()


def _resolve_num_positions_for_scene(
    config: dict,
    occupancy_map,
    default_num_positions: int,
) -> tuple[int, str]:
    position_sampling_cfg = config.get("position_sampling", {}) or {}
    area_rules = position_sampling_cfg.get("area_to_num_positions")
    if not area_rules:
        return int(default_num_positions), f"fixed num_positions={int(default_num_positions)}"

    metric = str(position_sampling_cfg.get("metric", "room_free_area_m2"))
    if metric == "room_free_area_m2":
        usable_mask = occupancy_map.room_free_mask if occupancy_map.room_free_mask is not None else occupancy_map.free_mask
    elif metric == "free_area_m2":
        usable_mask = occupancy_map.free_mask
    else:
        raise ValueError(f"Unsupported position_sampling.metric: {metric}")

    usable_area_m2 = float(int(usable_mask.sum())) * float(occupancy_map.resolution) ** 2
    for rule in area_rules:
        if usable_area_m2 < float(rule["max_area_m2"]):
            return int(rule["num_positions"]), f"{metric}={usable_area_m2:.3f} < {rule['max_area_m2']:.3f}"

    fallback = int(position_sampling_cfg.get("default_num_positions", default_num_positions))
    return fallback, f"{metric}={usable_area_m2:.3f} >= all thresholds"


def _score_from_counts(visible: int, total: int) -> float:
    return float(visible) / float(total) if int(total) > 0 else 0.0


def _save_bbox_json(path: Path, bbox_xyxy, image_width: int, image_height: int) -> list[float] | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bbox_xyxy is None:
        path.write_text(json.dumps({"xyxy_norm": None}, indent=2), encoding="utf-8")
        return None
    u_min, v_min, u_max, v_max = bbox_xyxy
    bbox_norm = [
        float(u_min) / float(image_width),
        float(v_min) / float(image_height),
        float(u_max) / float(image_width),
        float(v_max) / float(image_height),
    ]
    path.write_text(json.dumps({"xyxy_norm": bbox_norm}, indent=2), encoding="utf-8")
    return bbox_norm


def _resize_score_map(score_map: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(score_map, dtype=np.float32)
    image = Image.fromarray(arr, mode="F")
    image = image.resize((int(width), int(height)), _BILINEAR)
    return np.asarray(image, dtype=np.float32)


def _select_best_valid_pixel(score_map: np.ndarray, depth_m: np.ndarray) -> tuple[int, int, float] | None:
    height, width = depth_m.shape
    score_resized = _resize_score_map(score_map, width, height)
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m < 15.0)
    if not valid.any():
        return None
    masked = np.where(valid, score_resized, -np.inf)
    flat_idx = int(np.argmax(masked))
    row, col = np.unravel_index(flat_idx, masked.shape)
    if not np.isfinite(masked[row, col]):
        return None
    return int(col), int(row), float(masked[row, col])


def _backproject_pixel_to_world(
    pixel_xy: tuple[int, int],
    depth_m: np.ndarray,
    camera_position: tuple[float, float, float],
    camera_orientation_wxyz: tuple[float, float, float, float],
    resolution: tuple[int, int],
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
) -> tuple[float, float, float] | None:
    width, height = int(resolution[0]), int(resolution[1])
    col, row = int(pixel_xy[0]), int(pixel_xy[1])
    if not (0 <= col < width and 0 <= row < height):
        return None
    depth = float(depth_m[row, col])
    if not np.isfinite(depth) or depth <= 0.0:
        return None
    fx = width * float(focal_length) / max(float(horizontal_aperture), 1e-6)
    fy = height * float(focal_length) / max(float(vertical_aperture), 1e-6)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    forward = depth
    lateral = -(float(col) - cx) * forward / max(fx, 1e-6)
    vertical = -(float(row) - cy) * forward / max(fy, 1e-6)
    yaw_rad = 2.0 * math.atan2(float(camera_orientation_wxyz[3]), float(camera_orientation_wxyz[0]))
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    world_x = float(camera_position[0]) + forward * cos_yaw - lateral * sin_yaw
    world_y = float(camera_position[1]) + forward * sin_yaw + lateral * cos_yaw
    world_z = float(camera_position[2]) + vertical
    return float(world_x), float(world_y), float(world_z)


def _save_aho_overlay(
    rgb: np.ndarray,
    score_map: np.ndarray,
    selected_pixel_xy: tuple[int, int] | None,
    path: Path,
) -> None:
    import matplotlib.cm as cm
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    height, width = arr.shape[:2]
    score_resized = _resize_score_map(score_map, width, height)
    score_min = float(np.nanmin(score_resized))
    score_max = float(np.nanmax(score_resized))
    denom = max(score_max - score_min, 1e-6)
    score_norm = (score_resized - score_min) / denom
    heat = (cm.jet(score_norm)[..., :3] * 255.0).astype(np.uint8)
    overlay = (0.5 * heat + 0.5 * arr[..., :3]).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(overlay, mode="RGB")
    if selected_pixel_xy is not None:
        draw = ImageDraw.Draw(image)
        x, y = int(selected_pixel_xy[0]), int(selected_pixel_xy[1])
        r_outer, r_inner = 18, 7
        points = []
        for i in range(10):
            angle = -math.pi / 2.0 + i * math.pi / 5.0
            radius = r_outer if i % 2 == 0 else r_inner
            points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
        draw.polygon(points, fill=(0, 255, 0), outline=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _is_free_xy(occupancy_map, x: float, y: float) -> bool:
    row, col = occupancy_map.world_to_grid(float(x), float(y))
    if not (0 <= row < occupancy_map.height and 0 <= col < occupancy_map.width):
        return False
    return bool(occupancy_map.free_mask[row, col])


def _world_point_in_fov(
    world_xyz: tuple[float, float, float],
    camera_xy: tuple[float, float],
    camera_z: float,
    yaw_rad: float,
    resolution: tuple[int, int],
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    margin_px: float = 4.0,
) -> bool:
    dx = float(world_xyz[0]) - float(camera_xy[0])
    dy = float(world_xyz[1]) - float(camera_xy[1])
    dz = float(world_xyz[2]) - float(camera_z)
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    forward = dx * cos_yaw + dy * sin_yaw
    lateral = -dx * sin_yaw + dy * cos_yaw
    if forward <= 1e-6:
        return False
    width, height = int(resolution[0]), int(resolution[1])
    fx = width * float(focal_length) / max(float(horizontal_aperture), 1e-6)
    fy = height * float(focal_length) / max(float(vertical_aperture), 1e-6)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    u = cx - fx * lateral / forward
    v = cy - fy * dz / forward
    return (
        float(margin_px) <= float(u) < float(width) - float(margin_px)
        and float(margin_px) <= float(v) < float(height) - float(margin_px)
    )


def _filter_existing_scene_outputs(scene_configs: list[dict], output_roots: list[str]) -> tuple[list[dict], list[str]]:
    existing_scene_ids: set[str] = set()
    for root_text in output_roots:
        root = Path(root_text).expanduser()
        if not root.is_dir():
            continue
        existing_scene_ids.update(path.name for path in root.iterdir() if path.is_dir())
    pending = []
    skipped = []
    for scene_cfg in scene_configs:
        scene_id = str(scene_cfg.get("name", Path(scene_cfg["stage_url"]).stem))
        if scene_id in existing_scene_ids:
            skipped.append(scene_id)
            continue
        pending.append(scene_cfg)
    return pending, skipped


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(launch_config=config["launch_config"])

    import carb
    import carb.settings
    import omni.replicator.core as rep
    from isaacsim.core.experimental.utils.semantics import add_labels, remove_all_labels
    from isaacsim.core.utils.stage import get_current_stage, open_stage

    from utils.aho_inference import AHOInferenceRunner
    from utils.capture_outputs import (
        _camera_yaw_from_person,
        _save_depth,
        _save_rgb,
        _semantic_bbox_xyxy,
        _yaw_to_world_quaternion,
    )
    from utils.occupancy_overlay import save_score_field_overlay
    from utils.person_placement import place_person, place_person_near_anchor
    from utils.replicator_tools import (
        create_camera_pool,
        iter_batches,
        set_batch_poses_with_orientation,
        set_prim_visibility,
        set_render_products_updates_enabled,
        teardown_camera_pool,
    )
    from utils.ring_sampling import (
        ScoreFieldPoint,
        _has_full_width_occupancy_visibility,
        iter_shared_pair_camera_samples,
        select_capture_candidates,
    )
    from utils.score import read_semantic_counts

    carb.settings.get_settings().set("/log/level", "error")
    carb.settings.get_settings().set("/log/outputStreamLevel", "Error")
    carb_settings = carb.settings.get_settings()

    def _set_low_quality():
        carb_settings.set("/rtx/rendermode", "RayTracedLighting")

    def _set_high_quality():
        carb_settings.set("/rtx/rendermode", "PathTracing")

    rep.set_global_seed(int(config.get("sampling", {}).get("seed", 42)))
    rep.orchestrator.set_capture_on_play(False)
    _set_low_quality()

    score_field_cfg = config["score_field"]
    sampling_cfg = config.get("sampling", {})
    camera_cfg = config["camera"]
    person_cfg = config["person"]
    resolution = tuple(camera_cfg["resolution"])
    focal_length = float(camera_cfg["focal_length"])
    horizontal_aperture = float(_REP_DEFAULT_HORIZONTAL_APERTURE)
    vertical_aperture = float(_REP_DEFAULT_HORIZONTAL_APERTURE) * float(resolution[1]) / float(resolution[0])
    camera_z = float(camera_cfg.get("camera_height", 0.4))

    num_cameras = int(config["num_cameras"])
    rt_subframes = int(config.get("rt_subframes", 2))
    base_seed = int(sampling_cfg.get("seed", 42))
    min_position_distance = float(sampling_cfg.get("min_position_distance_m", 3.0))
    character_min_obstacle = float(sampling_cfg.get("character_min_obstacle_distance_m", 0.2))
    max_attempts = max(1, int(sampling_cfg.get("max_attempts_per_position", 8)))
    default_num_positions = int(config.get("num_positions", 1))
    body_width_m = float(score_field_cfg["occupancy_full_visibility_width_m"])
    max_cap = int(score_field_cfg.get("max_capture_candidates", 16))
    capture_score_min = float(score_field_cfg["capture_score_min"])
    capture_score_max = float(score_field_cfg["capture_score_max"])
    output_root = Path(config["backend_params"]["output_dir"])

    pair_cfg = config.get("pair_sampling", {}) or {}
    pair_min_distance = float(pair_cfg.get("min_pair_distance_m", 1.5))
    pair_max_distance = float(pair_cfg.get("max_pair_distance_m", 3.5))
    second_person_min_obstacle = float(pair_cfg.get("second_person_min_obstacle_distance_m", character_min_obstacle))
    pair_require_connectivity = bool(pair_cfg.get("require_pair_connectivity", True))
    max_pair_layout_attempts = max(1, int(pair_cfg.get("max_pair_layout_attempts", 6)))
    min_camera_distance_to_any_person = float(
        score_field_cfg.get("min_camera_distance_to_any_person_m", score_field_cfg["min_radius_m"])
    )

    primary_person_url = str(person_cfg["url"])
    secondary_person_url = str(person_cfg.get("secondary_url", primary_person_url))

    aho_runner = AHOInferenceRunner(config["aho_inference"])

    eval_cfg = config.get("eval", {}) or {}
    max_scenes = max(1, int(eval_cfg.get("max_scenes", 999999)))

    scene_configs = resolve_scene_configs(config)
    exclude_roots = config.get("dataset", {}).get("selection", {}).get("exclude_existing_output_roots", []) or []
    scene_configs, skipped_existing = _filter_existing_scene_outputs(scene_configs, exclude_roots)
    if skipped_existing:
        print(f"[AHO-EVAL] skipped {len(skipped_existing)} scene(s) already present in exclude_existing_output_roots")
    print(f"[AHO-EVAL] {len(scene_configs)} scene(s) to evaluate (max_scenes={max_scenes})")

    scenes_done = 0
    scenes_skipped_filter = 0
    scenes_skipped_no_candidates = 0
    scenes_failed = 0

    for scene_cfg in scene_configs:
        if scenes_done >= max_scenes:
            break

        scene_id = str(scene_cfg.get("name", Path(scene_cfg["stage_url"]).stem))
        stage_url = str(scene_cfg["stage_url"])
        print(f"\n[AHO-EVAL] ======== scene {scene_id} ========")

        try:
            occupancy_map = load_interiorgs_occupancy_map(scene_cfg)
        except Exception as exc:
            print(f"[AHO-EVAL] {scene_id}: occupancy load failed ({exc}); skipping.")
            scenes_failed += 1
            continue

        skip, reason = check_scene_filter(occupancy_map, config)
        if skip:
            print(f"[AHO-EVAL] {scene_id}: skipped by scene_filter ({reason})")
            scenes_skipped_filter += 1
            continue

        num_positions_scene, num_positions_reason = _resolve_num_positions_for_scene(
            config=config,
            occupancy_map=occupancy_map,
            default_num_positions=default_num_positions,
        )
        print(f"[AHO-EVAL] {scene_id}: num_positions={num_positions_scene} ({num_positions_reason})")

        if not open_stage(stage_url):
            print(f"[AHO-EVAL] {scene_id}: open_stage failed; skipping.")
            scenes_failed += 1
            continue
        _set_low_quality()

        stage = get_current_stage()
        if config.get("clear_previous_semantics", True):
            for prim in stage.Traverse():
                remove_all_labels(prim, include_descendants=True)
        stage.DefinePrim("/SDG", "Scope")

        set_prim_visibility(stage, "/World/gauss", visible=True)
        set_prim_visibility(stage, "/World/scene_collision", visible=False)
        add_labels("/World/scene_collision", labels="scene", taxonomy="class")

        driver_cams, render_products, annotators_by_name = create_camera_pool(
            num_cameras=num_cameras,
            resolution=resolution,
            focal_length=focal_length,
            focus_distance=float(camera_cfg.get("focus_distance", 400.0)),
            vertical_aperture=vertical_aperture,
            near_clipping_distance=float(camera_cfg.get("near_clipping_distance", 0.01)),
            annotator_names=("semantic_segmentation", "rgb", "distance_to_image_plane"),
        )
        set_render_products_updates_enabled(render_products, False)

        seg_annotators = annotators_by_name["semantic_segmentation"]
        rgb_annotators = annotators_by_name["rgb"]
        depth_annotators = annotators_by_name["distance_to_image_plane"]
        warmup_updates = int(config.get("scene", {}).get("warmup_updates", 4))

        def _begin_render_pass(scene_mesh_visible: bool):
            set_render_products_updates_enabled(render_products, False)
            set_prim_visibility(stage, "/World/scene_collision", visible=scene_mesh_visible)
            set_render_products_updates_enabled(render_products, True)
            for _ in range(warmup_updates):
                simulation_app.update()

        def _end_render_pass():
            set_render_products_updates_enabled(render_products, False)

        pair_initialized = False
        pair_labels_initialized = False
        existing_world_points_xy: list[tuple[float, float]] = []

        try:
            for pos_idx in range(num_positions_scene):
                pos_tag = f"pos_{pos_idx:03d}"
                pos_seed = base_seed + pos_idx * 10007
                print(f"\n[AHO-EVAL] ===== {scene_id} / {pos_tag} =====")
                _set_low_quality()

                # --- 放置两个人 ---
                person_xyz = None
                context_person_xyz = None
                for attempt in range(max_attempts):
                    attempt_seed = pos_seed + attempt * 17
                    try:
                        primary_result = place_person(
                            stage=stage,
                            occupancy_map=occupancy_map,
                            prim_path="/SDG/Persons/person_000",
                            seed=attempt_seed,
                            character_usd_path=primary_person_url,
                            arm_drop_degrees=float(config.get("scene", {}).get("character_arm_drop_degrees", 75.0)),
                            min_obstacle_distance_m=character_min_obstacle,
                            existing_world_points_xy=existing_world_points_xy if existing_world_points_xy else None,
                            min_point_distance_m=min_position_distance,
                            reuse_existing_prim=pair_initialized,
                        )
                        person_xyz = primary_result["position"]

                        secondary_result = None
                        for pair_attempt in range(max_pair_layout_attempts):
                            secondary_seed = attempt_seed + pair_attempt * 101 + 1
                            try:
                                secondary_result = place_person_near_anchor(
                                    stage=stage,
                                    occupancy_map=occupancy_map,
                                    anchor_position_xy=(float(person_xyz[0]), float(person_xyz[1])),
                                    prim_path="/SDG/Persons/person_001",
                                    seed=secondary_seed,
                                    character_usd_path=secondary_person_url,
                                    arm_drop_degrees=float(config.get("scene", {}).get("character_arm_drop_degrees", 75.0)),
                                    min_distance_m=pair_min_distance,
                                    max_distance_m=pair_max_distance,
                                    min_obstacle_distance_m=second_person_min_obstacle,
                                    require_pair_connectivity=pair_require_connectivity,
                                    reuse_existing_prim=pair_initialized,
                                )
                                break
                            except Exception as pair_exc:
                                print(f"[AHO-EVAL] secondary placement attempt {pair_attempt} failed: {pair_exc}")
                        if secondary_result is None:
                            raise RuntimeError("failed to place secondary person near primary")
                        context_person_xyz = secondary_result["position"]
                        pair_initialized = True
                        break
                    except Exception as exc:
                        person_xyz = None
                        context_person_xyz = None
                        print(f"[AHO-EVAL] place_person attempt {attempt} failed: {exc}")

                if person_xyz is None:
                    print(f"[AHO-EVAL] {pos_tag}: failed to place persons; skipping pos.")
                    continue

                if not pair_labels_initialized:
                    add_labels("/SDG/Persons/person_000", labels="person_000", taxonomy="class")
                    add_labels("/SDG/Persons/person_001", labels="person_001", taxonomy="class")
                    pair_labels_initialized = True

                for _ in range(warmup_updates):
                    simulation_app.update()

                print(f"[AHO-EVAL] {pos_tag} person_000: {person_xyz}")
                print(f"[AHO-EVAL] {pos_tag} person_001: {context_person_xyz}")

                targets = [
                    {"instance_id": "person_000", "semantic_label": "person_000", "position": person_xyz},
                    {"instance_id": "person_001", "semantic_label": "person_001", "position": context_person_xyz},
                ]
                pair_center_xy = (
                    0.5 * (float(person_xyz[0]) + float(context_person_xyz[0])),
                    0.5 * (float(person_xyz[1]) + float(context_person_xyz[1])),
                )

                # --- 共享候选位置（对两人都可见的 ring） ---
                ring_samples = list(
                    iter_shared_pair_camera_samples(
                        occupancy_map=occupancy_map,
                        pair_center_xy=pair_center_xy,
                        target_positions_xy=[
                            (float(person_xyz[0]), float(person_xyz[1])),
                            (float(context_person_xyz[0]), float(context_person_xyz[1])),
                        ],
                        camera_height_m=camera_z,
                        min_radius_m=float(score_field_cfg["min_radius_m"]),
                        max_radius_m=float(score_field_cfg["max_radius_m"]),
                        grid_step_m=float(score_field_cfg["grid_step_m"]),
                        min_obstacle_distance_m=float(score_field_cfg["camera_min_obstacle_distance_m"]),
                        min_distance_to_any_person_m=min_camera_distance_to_any_person,
                    )
                )
                if not ring_samples:
                    print(f"[AHO-EVAL] {pos_tag}: no shared ring samples; skipping pos.")
                    continue

                # 相机朝向 pair_center 用于 certain/uncertain 判断和 scoring
                # certain: 对每个 target 都有完整宽度可见性
                # 分别对每个 target 判断 certain/uncertain，uncertain = 任一 target 不确定
                certain_indices: list[int] = []
                uncertain_indices: list[int] = []
                for i, pose in enumerate(ring_samples):
                    both_certain = all(
                        _has_full_width_occupancy_visibility(
                            occupancy_map=occupancy_map,
                            person_position_xy=(float(t["position"][0]), float(t["position"][1])),
                            camera_position_xy=(pose["x"], pose["y"]),
                            body_width_m=body_width_m,
                        )
                        for t in targets
                    )
                    if both_certain:
                        certain_indices.append(i)
                    else:
                        uncertain_indices.append(i)

                # 对 uncertain poses 做 segmentation scoring（每个 target 分别）
                # look_at_xyz 用 pair_center
                uncertain_poses = [ring_samples[i] for i in uncertain_indices]

                # 为每个 uncertain pose 计算朝向 pair_center 的 yaw
                look_at_xyz = (float(pair_center_xy[0]), float(pair_center_xy[1]), camera_z + 1.0)
                uncertain_poses_oriented = []
                for pose in uncertain_poses:
                    yaw = _camera_yaw_from_person((pose["x"], pose["y"]), pair_center_xy)
                    uncertain_poses_oriented.append({**pose, "yaw_rad": float(yaw)})

                # target_label -> list of counts per uncertain pose
                uncertain_total_by_target: dict[str, list[int]] = {t["semantic_label"]: [] for t in targets}
                uncertain_visible_by_target: dict[str, list[int]] = {t["semantic_label"]: [] for t in targets}

                if uncertain_poses_oriented:
                    print(f"[AHO-EVAL] {pos_tag}: scoring {len(uncertain_poses_oriented)} uncertain candidates")
                    _begin_render_pass(scene_mesh_visible=False)
                    for batch in iter_batches(uncertain_poses_oriented, num_cameras):
                        set_batch_poses_with_orientation(driver_cams, batch)
                        rep.orchestrator.step(rt_subframes=rt_subframes)
                        for t in targets:
                            uncertain_total_by_target[t["semantic_label"]].extend(
                                read_semantic_counts(seg_annotators, t["semantic_label"])[: len(batch)]
                            )
                    _end_render_pass()

                    _begin_render_pass(scene_mesh_visible=True)
                    for batch in iter_batches(uncertain_poses_oriented, num_cameras):
                        set_batch_poses_with_orientation(driver_cams, batch)
                        rep.orchestrator.step(rt_subframes=rt_subframes)
                        for t in targets:
                            uncertain_visible_by_target[t["semantic_label"]].extend(
                                read_semantic_counts(seg_annotators, t["semantic_label"])[: len(batch)]
                            )
                    _end_render_pass()

                # 为每个 target 构建 score_field，然后取交集筛选候选
                # score 取两 target 分数的最小值（保守策略）
                score_field: list[ScoreFieldPoint] = []

                for i in certain_indices:
                    pose = ring_samples[i]
                    yaw = _camera_yaw_from_person((pose["x"], pose["y"]), pair_center_xy)
                    score_field.append(ScoreFieldPoint(
                        x=pose["x"], y=pose["y"], z=pose["z"],
                        camera_z=pose["camera_z"], yaw_rad=float(yaw),
                        score=1.0, distance_m=pose["distance_m"],
                        visible_person_pixels=1, total_person_pixels=1,
                        scoring_mode="occupancy_full_visibility",
                    ))

                for idx_u, i in enumerate(uncertain_indices):
                    pose = ring_samples[i]
                    yaw = _camera_yaw_from_person((pose["x"], pose["y"]), pair_center_xy)
                    scores_per_target = []
                    for t in targets:
                        lbl = t["semantic_label"]
                        visible = int(uncertain_visible_by_target[lbl][idx_u])
                        total = int(uncertain_total_by_target[lbl][idx_u])
                        scores_per_target.append(_score_from_counts(visible, total))
                    min_score = min(scores_per_target)
                    score_field.append(ScoreFieldPoint(
                        x=pose["x"], y=pose["y"], z=pose["z"],
                        camera_z=pose["camera_z"], yaw_rad=float(yaw),
                        score=float(min_score), distance_m=pose["distance_m"],
                        scoring_mode="segmentation_visibility",
                    ))

                # 候选筛选：score 在范围内 + 两个人都在 FOV 内
                eligible = []
                for sf in score_field:
                    if not (capture_score_min <= sf.score <= capture_score_max):
                        continue
                    both_in_fov = all(
                        _world_point_in_fov(
                            world_xyz=(float(t["position"][0]), float(t["position"][1]), float(t["position"][2]) + 1.0),
                            camera_xy=(sf.x, sf.y),
                            camera_z=camera_z,
                            yaw_rad=sf.yaw_rad,
                            resolution=resolution,
                            focal_length=focal_length,
                            horizontal_aperture=horizontal_aperture,
                            vertical_aperture=vertical_aperture,
                        )
                        for t in targets
                    )
                    if both_in_fov:
                        eligible.append(sf)

                rng = np.random.default_rng(pos_seed)
                rng.shuffle(eligible)
                selected = eligible[:max_cap]

                if not selected:
                    print(
                        f"[AHO-EVAL] {pos_tag}: no candidates in score range "
                        f"[{capture_score_min:.2f}, {capture_score_max:.2f}] with both in FOV; skipping pos."
                    )
                    scenes_skipped_no_candidates += 1
                    continue

                # --- Before views ---
                scene_dir = output_root / scene_id / pos_tag
                scene_dir.mkdir(parents=True, exist_ok=True)

                before_poses: list[dict] = []
                before_meta: list[dict] = []
                for sf in selected:
                    quat = _yaw_to_world_quaternion(sf.yaw_rad)
                    before_poses.append({"x": float(sf.x), "y": float(sf.y), "camera_z": camera_z, "yaw_rad": sf.yaw_rad})
                    before_meta.append({
                        "camera_position": [float(sf.x), float(sf.y), camera_z],
                        "camera_orientation_wxyz": [float(v) for v in quat],
                        "source_score": float(sf.score),
                    })

                _set_high_quality()
                print(f"[AHO-EVAL] {pos_tag}: rendering {len(before_poses)} before views")
                _begin_render_pass(scene_mesh_visible=False)
                before_rgb_frames = [None] * len(before_poses)
                before_seg_hidden = [None] * len(before_poses)
                before_total_by_target = {t["semantic_label"]: [] for t in targets}
                view_idx = 0
                for batch in iter_batches(before_poses, num_cameras):
                    set_batch_poses_with_orientation(driver_cams, batch)
                    rep.orchestrator.step(rt_subframes=rt_subframes)
                    for t in targets:
                        before_total_by_target[t["semantic_label"]].extend(
                            read_semantic_counts(seg_annotators, t["semantic_label"])[: len(batch)]
                        )
                    for i in range(len(batch)):
                        before_rgb_frames[view_idx + i] = np.asarray(rgb_annotators[i].get_data())
                        before_seg_hidden[view_idx + i] = seg_annotators[i].get_data()
                    view_idx += len(batch)
                _end_render_pass()

                _begin_render_pass(scene_mesh_visible=True)
                before_depth_frames = [None] * len(before_poses)
                before_visible_by_target = {t["semantic_label"]: [] for t in targets}
                view_idx = 0
                for batch in iter_batches(before_poses, num_cameras):
                    set_batch_poses_with_orientation(driver_cams, batch)
                    rep.orchestrator.step(rt_subframes=rt_subframes)
                    for t in targets:
                        before_visible_by_target[t["semantic_label"]].extend(
                            read_semantic_counts(seg_annotators, t["semantic_label"])[: len(batch)]
                        )
                    for i in range(len(batch)):
                        before_depth_frames[view_idx + i] = np.asarray(depth_annotators[i].get_data(), dtype=np.float32)
                    view_idx += len(batch)
                _end_render_pass()

                # --- AHO inference per view per target ---
                # after_requests: list of (view_idx, target_id, after_pose)
                after_requests: list[tuple[int, str, dict]] = []
                records = []

                for idx, meta in enumerate(before_meta):
                    view_dir = scene_dir / f"view_{idx:03d}"
                    before_dir = view_dir / "before"
                    before_dir.mkdir(parents=True, exist_ok=True)

                    before_rgb = before_rgb_frames[idx]
                    before_depth = before_depth_frames[idx]
                    before_seg = before_seg_hidden[idx]

                    from utils.capture_outputs import _save_rgb, _save_depth
                    _save_rgb(before_rgb, before_dir / "rgb.png")
                    _save_depth(before_depth, before_dir / "depth.png", before_dir / "depth.npy")

                    record = {
                        "view_idx": idx,
                        "camera_position": meta["camera_position"],
                        "camera_orientation_wxyz": meta["camera_orientation_wxyz"],
                        "source_score": meta["source_score"],
                        "targets": {},
                    }

                    for t in targets:
                        lbl = t["semantic_label"]
                        before_score = _score_from_counts(
                            before_visible_by_target[lbl][idx],
                            before_total_by_target[lbl][idx],
                        )
                        bbox = _semantic_bbox_xyxy(before_seg, lbl)
                        bbox_norm = _save_bbox_json(
                            before_dir / f"bbox_{lbl}.json",
                            bbox,
                            image_width=int(resolution[0]),
                            image_height=int(resolution[1]),
                        )

                        aho_prediction = None
                        after_pose = None
                        if bbox_norm is not None:
                            aho_pred = aho_runner.predict(
                                rgb=before_rgb,
                                depth_m=before_depth,
                                bbox_norm=bbox_norm,
                            )
                            score_map = np.asarray(aho_pred["score_map"], dtype=np.float32)
                            np.save(view_dir / f"aho_score_map_{lbl}.npy", score_map)
                            pixel_choice = _select_best_valid_pixel(score_map, before_depth)
                            backprojected_world = None
                            selected_pixel = None
                            if pixel_choice is not None:
                                col, row, selected_score = pixel_choice
                                selected_pixel = [int(col), int(row)]
                                backprojected_world = _backproject_pixel_to_world(
                                    pixel_xy=(col, row),
                                    depth_m=before_depth,
                                    camera_position=tuple(meta["camera_position"]),
                                    camera_orientation_wxyz=tuple(meta["camera_orientation_wxyz"]),
                                    resolution=resolution,
                                    focal_length=focal_length,
                                    horizontal_aperture=horizontal_aperture,
                                    vertical_aperture=vertical_aperture,
                                )
                                if backprojected_world is not None:
                                    after_x, after_y, _ = backprojected_world
                                    after_yaw = _camera_yaw_from_person(
                                        (after_x, after_y),
                                        (float(t["position"][0]), float(t["position"][1])),
                                    )
                                    after_pose = {"x": float(after_x), "y": float(after_y), "camera_z": camera_z, "yaw_rad": float(after_yaw)}
                                    after_requests.append((idx, lbl, after_pose))

                            _save_aho_overlay(
                                before_rgb, score_map,
                                tuple(selected_pixel) if selected_pixel else None,
                                view_dir / f"aho_overlay_{lbl}.png",
                            )
                            aho_prediction = {
                                "quality": float(aho_pred["quality"]),
                                "max_score": float(aho_pred["max_score"]),
                                "selected_pixel": selected_pixel,
                                "backprojected_world": (
                                    [float(v) for v in backprojected_world] if backprojected_world else None
                                ),
                                "backprojected_xy_free": (
                                    bool(_is_free_xy(occupancy_map, backprojected_world[0], backprojected_world[1]))
                                    if backprojected_world else False
                                ),
                            }

                        record["targets"][lbl] = {
                            "before_score": float(before_score),
                            "bbox_norm": bbox_norm,
                            "aho_prediction": aho_prediction,
                            "after_pose": after_pose,
                            "after_score": None,
                        }

                    (view_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                    records.append(record)

                # --- After views (render all after poses at once) ---
                if after_requests:
                    print(f"[AHO-EVAL] {pos_tag}: rendering {len(after_requests)} after views")
                    after_poses_flat = [req[2] for req in after_requests]

                    _begin_render_pass(scene_mesh_visible=False)
                    after_seg_hidden_flat = [None] * len(after_poses_flat)
                    after_total_flat = {t["semantic_label"]: [] for t in targets}
                    view_idx = 0
                    for batch in iter_batches(after_poses_flat, num_cameras):
                        set_batch_poses_with_orientation(driver_cams, batch)
                        rep.orchestrator.step(rt_subframes=rt_subframes)
                        for t in targets:
                            after_total_flat[t["semantic_label"]].extend(
                                read_semantic_counts(seg_annotators, t["semantic_label"])[: len(batch)]
                            )
                        for i in range(len(batch)):
                            after_seg_hidden_flat[view_idx + i] = seg_annotators[i].get_data()
                        view_idx += len(batch)
                    _end_render_pass()

                    _begin_render_pass(scene_mesh_visible=True)
                    after_rgb_flat = [None] * len(after_poses_flat)
                    after_depth_flat = [None] * len(after_poses_flat)
                    after_visible_flat = {t["semantic_label"]: [] for t in targets}
                    view_idx = 0
                    for batch in iter_batches(after_poses_flat, num_cameras):
                        set_batch_poses_with_orientation(driver_cams, batch)
                        rep.orchestrator.step(rt_subframes=rt_subframes)
                        for t in targets:
                            after_visible_flat[t["semantic_label"]].extend(
                                read_semantic_counts(seg_annotators, t["semantic_label"])[: len(batch)]
                            )
                        for i in range(len(batch)):
                            after_rgb_flat[view_idx + i] = np.asarray(rgb_annotators[i].get_data())
                            after_depth_flat[view_idx + i] = np.asarray(depth_annotators[i].get_data(), dtype=np.float32)
                        view_idx += len(batch)
                    _end_render_pass()

                    for after_idx, (view_i, lbl, _) in enumerate(after_requests):
                        after_score = _score_from_counts(
                            after_visible_flat[lbl][after_idx],
                            after_total_flat[lbl][after_idx],
                        )
                        after_dir = scene_dir / f"view_{view_i:03d}" / f"after_{lbl}"
                        after_dir.mkdir(parents=True, exist_ok=True)
                        _save_rgb(after_rgb_flat[after_idx], after_dir / "rgb.png")
                        _save_depth(after_depth_flat[after_idx], after_dir / "depth.png", after_dir / "depth.npy")
                        records[view_i]["targets"][lbl]["after_score"] = float(after_score)

                    # 更新 record.json
                    for record in records:
                        view_dir = scene_dir / f"view_{record['view_idx']:03d}"
                        (view_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

                # --- Summary ---
                target_stats = {}
                for t in targets:
                    lbl = t["semantic_label"]
                    before_scores = [r["targets"][lbl]["before_score"] for r in records]
                    after_scores = [r["targets"][lbl]["after_score"] for r in records if r["targets"][lbl]["after_score"] is not None]
                    paired_deltas = [
                        r["targets"][lbl]["after_score"] - r["targets"][lbl]["before_score"]
                        for r in records if r["targets"][lbl]["after_score"] is not None
                    ]
                    target_stats[lbl] = {
                        "mean_before": float(np.mean(before_scores)) if before_scores else 0.0,
                        "mean_after": float(np.mean(after_scores)) if after_scores else 0.0,
                        "mean_delta": float(np.mean(paired_deltas)) if paired_deltas else 0.0,
                        "success_rate": float(np.mean([d > 0.0 for d in paired_deltas])) if paired_deltas else 0.0,
                        "num_after": len(after_scores),
                    }

                all_deltas = []
                for t in targets:
                    lbl = t["semantic_label"]
                    all_deltas.extend([
                        r["targets"][lbl]["after_score"] - r["targets"][lbl]["before_score"]
                        for r in records if r["targets"][lbl]["after_score"] is not None
                    ])

                summary = {
                    "scene_id": scene_id,
                    "pos_idx": pos_idx,
                    "person_000_position": [float(v) for v in person_xyz],
                    "person_001_position": [float(v) for v in context_person_xyz],
                    "num_before_views": len(records),
                    "target_stats": target_stats,
                    "overall_mean_delta": float(np.mean(all_deltas)) if all_deltas else 0.0,
                    "overall_success_rate": float(np.mean([d > 0.0 for d in all_deltas])) if all_deltas else 0.0,
                    "records": records,
                }
                (scene_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

                print(
                    f"[AHO-EVAL] {pos_tag} summary: "
                    f"overall delta={summary['overall_mean_delta']:.4f}, "
                    f"success={summary['overall_success_rate']:.3f}"
                )
                for lbl, st in target_stats.items():
                    print(
                        f"  {lbl}: before={st['mean_before']:.4f} after={st['mean_after']:.4f} "
                        f"delta={st['mean_delta']:.4f} success={st['success_rate']:.3f}"
                    )

                existing_world_points_xy.append((float(person_xyz[0]), float(person_xyz[1])))
                scenes_done += 1

        finally:
            print(f"[AHO-EVAL] tearing down camera pool for {scene_id}")
            try:
                teardown_camera_pool(
                    stage=stage,
                    annotators_by_name=annotators_by_name,
                    render_products=render_products,
                    scope_path="/SDG/Cameras",
                )
            except Exception as exc:
                print(f"[AHO-EVAL] teardown warned: {exc}")

    print(
        f"\n[AHO-EVAL] finished: done={scenes_done}, "
        f"skipped_filter={scenes_skipped_filter}, "
        f"skipped_no_candidates={scenes_skipped_no_candidates}, "
        f"failed={scenes_failed}"
    )

    if config.get("close_app_after_run", True):
        simulation_app.close()
    else:
        print("[AHO-EVAL] GUI remains open; close the window to exit.")
        while simulation_app.is_running():
            simulation_app.update()
        simulation_app.close()


if __name__ == "__main__":
    main()
