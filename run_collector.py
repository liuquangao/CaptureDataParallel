# Copyright (c) 2026, Fusion Intelligence Labs, University of Exeter. All rights reserved.
#
# 并行采集入口。流程与 debug_replicator_cameras.py 保持一致,所有可复用函数都从
# utils/ 引入,不再依赖 collector/ 下的模块。
#
# 输出 <backend_params.output_dir>/<scene_id>/pos_{idx:03d}/:
#   score_field_overlay.png         (Stage 1, Stage 3 叠加 selected 后覆写)
#   rgb/{idx:03d}.png               (Stage 3)
#   depth/{idx:03d}.png             (uint16 mm)
#   depth/{idx:03d}.npy             (float32 meters)
#   ground_mask/{idx:03d}.png
#   score_map/{idx:03d}.npy
#   valid_mask/{idx:03d}.npy
#   yaw_map/{idx:03d}.npy
#   person_bbox/{idx:03d}.json
#   score_field_views/{target_id}/...  (optional target-facing candidate RGBD)
#   metadata.json
#   scores.json

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path


_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


from utils.config_io import load_config


# rep.functional.create.camera 默认 horizontal_aperture,和 create_camera_pool
# 内部保持一致,用于 compute_ground_mask / compute_score_map 的投影公式。
_REP_DEFAULT_HORIZONTAL_APERTURE = 20.955


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim parallel data collector")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
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
        max_area_m2 = float(rule["max_area_m2"])
        num_positions = int(rule["num_positions"])
        if usable_area_m2 < max_area_m2:
            return num_positions, f"{metric}={usable_area_m2:.3f} < {max_area_m2:.3f}"

    fallback_num_positions = int(position_sampling_cfg.get("default_num_positions", default_num_positions))
    return fallback_num_positions, f"{metric}={usable_area_m2:.3f} >= all thresholds"


def _filter_existing_scene_outputs(scene_configs: list[dict], output_root: Path) -> tuple[list[dict], list[str]]:
    pending: list[dict] = []
    skipped: list[str] = []
    for scene_cfg in scene_configs:
        scene_id = str(scene_cfg.get("name", Path(scene_cfg["stage_url"]).stem))
        if (output_root / scene_id).is_dir():
            skipped.append(scene_id)
            continue
        pending.append(scene_cfg)
    return pending, skipped


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # --- SimulationApp 必须最早启动;启动后才能 import omni.* ---
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(launch_config=config["launch_config"])

    import carb
    import carb.settings

    carb.settings.get_settings().set("/log/level", "error")
    # 压掉 carb 插件框架/hydratexture 的 Warning(走独立 output stream,不受 /log/level 约束)
    carb.settings.get_settings().set("/log/outputStreamLevel", "Error")

    import numpy as np
    import omni.timeline
    import omni.replicator.core as rep
    from isaacsim.core.experimental.utils.semantics import add_labels, remove_all_labels
    from isaacsim.core.utils.stage import get_current_stage, open_stage

    from utils.capture_outputs import (
        _camera_yaw_from_person,
        _compute_ground_mask,
        _compute_score_map,
        _save_depth,
        _save_ground_mask,
        _save_person_bbox_norm,
        _save_rgb,
        _save_score_map,
        _save_valid_mask,
        _save_yaw_map,
        _semantic_bbox_xyxy,
        _yaw_to_world_quaternion,
    )
    from utils.occupancy_map import load_interiorgs_occupancy_map
    from utils.occupancy_overlay import save_score_field_overlay
    from utils.person_placement import ensure_scene_query_support, place_person, place_person_near_anchor
    from utils.raycast_score import get_skeleton_joint_world_positions, score_target_joint_visibility
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
        iter_shared_pair_camera_samples,
    )
    from utils.scene_selection import check_scene_filter, resolve_scene_configs

    rep.set_global_seed(int(config.get("sampling", {}).get("seed", 42)))
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
    timeline = omni.timeline.get_timeline_interface()

    # ------------------------------------------------------------------
    # 0. 通用参数(跨场景共享)
    # ------------------------------------------------------------------
    score_field_cfg = config["score_field"]
    score_field_views_cfg = config.get("score_field_views", {}) or {}
    sampling_cfg = config.get("sampling", {})
    camera_cfg = config["camera"]
    person_cfg = config["person"]
    resolution = tuple(camera_cfg["resolution"])
    focal_length = float(camera_cfg["focal_length"])
    # RTX 默认 horizontal fit,保证方形像素
    horizontal_aperture = float(_REP_DEFAULT_HORIZONTAL_APERTURE)
    vertical_aperture = float(_REP_DEFAULT_HORIZONTAL_APERTURE) * float(resolution[1]) / float(resolution[0])

    camera_z = float(camera_cfg.get("camera_height", 0.4))

    num_cameras = int(config["num_cameras"])
    rt_subframes = int(config.get("rt_subframes", 2))

    default_num_positions = int(config.get("num_positions", 1))
    base_seed = int(sampling_cfg.get("seed", 42))
    min_position_distance = float(sampling_cfg.get("min_position_distance_m", 3.0))
    character_min_obstacle = float(sampling_cfg.get("character_min_obstacle_distance_m", 0.2))
    max_attempts = max(1, int(sampling_cfg.get("max_attempts_per_position", 2)))
    min_camera_distance_to_any_person = float(
        score_field_cfg.get("min_camera_distance_to_any_person_m", score_field_cfg["min_radius_m"])
    )
    max_cap = int(score_field_cfg.get("max_capture_candidates", 16))
    capture_score_min = float(score_field_cfg["capture_score_min"])
    capture_score_max = float(score_field_cfg["capture_score_max"])
    save_score_field_views = bool(score_field_views_cfg.get("enabled", False))
    score_field_views_save_rgb = bool(score_field_views_cfg.get("save_rgb", True))
    score_field_views_save_depth = bool(score_field_views_cfg.get("save_depth", True))
    score_field_views_save_bbox = bool(score_field_views_cfg.get("save_bbox", True))
    score_field_views_save_metadata = bool(score_field_views_cfg.get("save_metadata", True))
    output_root = Path(config["backend_params"]["output_dir"])
    primary_person_url = str(person_cfg["url"])
    secondary_person_url = str(person_cfg.get("secondary_url", primary_person_url))
    pair_cfg = config.get("pair_sampling", {})
    pair_min_distance = float(pair_cfg.get("min_pair_distance_m", 1.5))
    pair_max_distance = float(pair_cfg.get("max_pair_distance_m", 3.5))
    second_person_min_obstacle = float(
        pair_cfg.get("second_person_min_obstacle_distance_m", character_min_obstacle)
    )
    pair_require_connectivity = bool(pair_cfg.get("require_pair_connectivity", True))
    max_pair_layout_attempts = max(1, int(pair_cfg.get("max_pair_layout_attempts", 6)))

    # ------------------------------------------------------------------
    # 1. 解析要采的场景列表
    # ------------------------------------------------------------------
    scene_configs = resolve_scene_configs(config)
    skip_existing_scene_dirs = bool(config.get("backend_params", {}).get("skip_existing_scene_dirs", False))
    if skip_existing_scene_dirs:
        scene_configs, skipped_existing_scenes = _filter_existing_scene_outputs(scene_configs, output_root)
        print(
            f"[SDG] Skipped {len(skipped_existing_scenes)} existing scene output dir(s) under {output_root}"
        )
    print(f"[SDG] Resolved {len(scene_configs)} scene(s) to collect")

    filtered_scenes: list[tuple[str, str]] = []
    failed_scenes: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # 2. 场景循环
    # ------------------------------------------------------------------
    for scene_idx, scene_cfg in enumerate(scene_configs):
        scene_id = str(scene_cfg.get("name", Path(scene_cfg["stage_url"]).stem))
        stage_url = str(scene_cfg["stage_url"])
        print(f"\n[SDG] ======== scene {scene_idx + 1}/{len(scene_configs)}: {scene_id} ========")

        # --- 2.1 先 load 占用图,按 scene_filter 决定是否跳过(未 open_stage,成本低) ---
        try:
            occupancy_map = load_interiorgs_occupancy_map(scene_cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"[SDG] {scene_id}: occupancy load failed ({exc}); skipping scene.")
            failed_scenes.append((scene_id, f"occupancy load: {exc}"))
            continue
        skip, reason = check_scene_filter(occupancy_map, config)
        if skip:
            print(f"[SDG] {scene_id}: skipped by scene_filter ({reason})")
            filtered_scenes.append((scene_id, reason or ""))
            continue

        num_positions_scene, num_positions_reason = _resolve_num_positions_for_scene(
            config=config,
            occupancy_map=occupancy_map,
            default_num_positions=default_num_positions,
        )
        print(f"[SDG] {scene_id}: num_positions={num_positions_scene} ({num_positions_reason})")

        # --- 2.2 打开 stage + 场景级设置 ---
        print(f"[SDG] {scene_id}: loading stage {stage_url}")
        if not open_stage(stage_url):
            print(f"[SDG] {scene_id}: open_stage failed; skipping scene.")
            failed_scenes.append((scene_id, "open_stage failed"))
            continue
        stage = get_current_stage()
        if config.get("clear_previous_semantics", True):
            for prim in stage.Traverse():
                remove_all_labels(prim, include_descendants=True)
        stage.DefinePrim("/SDG", "Scope")

        set_prim_visibility(stage, "/World/gauss", visible=True)
        set_prim_visibility(stage, "/World/scene_collision", visible=True)
        add_labels("/World/scene_collision", labels="scene", taxonomy="class")
        ensure_scene_query_support(stage)
        from omni.physx import get_physx_scene_query_interface

        scene_query = get_physx_scene_query_interface()
        warmup_updates = int(config.get("scene", {}).get("warmup_updates", 4))
        if not timeline.is_playing():
            timeline.play()
        for _ in range(warmup_updates):
            simulation_app.update()

        # --- 2.3 相机池(每场景独立建 + 拆,避免 hydra handle 指向旧 stage) ---
        driver_cams, render_products, annotators_by_name = create_camera_pool(
            num_cameras=num_cameras,
            resolution=resolution,
            focal_length=focal_length,
            focus_distance=float(camera_cfg.get("focus_distance", 400.0)),
            vertical_aperture=vertical_aperture,
            near_clipping_distance=float(camera_cfg.get("near_clipping_distance", 0.01)),
            annotator_names=(
                "semantic_segmentation",
                "rgb",
                "distance_to_image_plane",
            ),
        )
        # 默认关 hydra 更新,只在每个 Pass 渲染时开(避免 visibility 切换触发 plugin released warning)
        set_render_products_updates_enabled(render_products, False)

        seg_annotators = annotators_by_name["semantic_segmentation"]
        rgb_annotators = annotators_by_name["rgb"]
        depth_annotators = annotators_by_name["distance_to_image_plane"]

        def _begin_render_pass(scene_mesh_visible: bool):
            """关 hydra → 切 /World/scene_collision → 开 hydra → warmup 数帧。渲染完必须配对 _end_render_pass。"""
            set_render_products_updates_enabled(render_products, False)
            set_prim_visibility(stage, "/World/scene_collision", visible=scene_mesh_visible)
            set_render_products_updates_enabled(render_products, True)
            for _ in range(warmup_updates):
                simulation_app.update()

        def _end_render_pass():
            set_render_products_updates_enabled(render_products, False)

        def _save_score_field_target_views(
            *,
            pos_dir: Path,
            pos_tag: str,
            target: dict,
            target_poses: list[dict],
            score_field: list[ScoreFieldPoint],
        ) -> None:
            if not save_score_field_views:
                return
            if not target_poses:
                return

            target_id = str(target["instance_id"])
            target_label = str(target["semantic_label"])
            target_dir = pos_dir / "score_field_views" / target_id
            point_by_candidate_id = {
                int(item.candidate_id): item
                for item in score_field
                if item.candidate_id is not None
            }
            manifest_records: list[dict] = []
            records_by_candidate_id: dict[int, dict] = {}

            for pose in target_poses:
                candidate_id = int(pose["candidate_id"])
                stem = f"{candidate_id:06d}"
                point = point_by_candidate_id.get(candidate_id)
                quat = _yaw_to_world_quaternion(float(pose["yaw_rad"]))
                record = {
                    "candidate_id": candidate_id,
                    "candidate_key": [round(float(pose["x"]), 6), round(float(pose["y"]), 6)],
                    "target_id": target_id,
                    "target_semantic_label": target_label,
                    "camera_position": [float(pose["x"]), float(pose["y"]), float(pose["camera_z"])],
                    "camera_orientation_wxyz": [float(v) for v in quat],
                    "x": float(pose["x"]),
                    "y": float(pose["y"]),
                    "z": float(pose["z"]),
                    "camera_z": float(pose["camera_z"]),
                    "yaw_rad": float(pose["yaw_rad"]),
                    "distance_m": float(pose["distance_m"]),
                    "raycast_score": float(point.score) if point is not None else None,
                    "scoring_mode": point.scoring_mode if point is not None else "missing",
                    "rgb_path": str(target_dir / "rgb" / f"{stem}.png") if score_field_views_save_rgb else None,
                    "depth_path": str(target_dir / "depth" / f"{stem}.png") if score_field_views_save_depth else None,
                    "depth_npy_path": str(target_dir / "depth" / f"{stem}.npy") if score_field_views_save_depth else None,
                    "bbox_path": str(target_dir / "bbox" / f"{stem}.json") if score_field_views_save_bbox else None,
                    "bbox_valid": None,
                }
                manifest_records.append(record)
                records_by_candidate_id[candidate_id] = record

            print(f"[SDG] {pos_tag} / {target_id} saving {len(target_poses)} score-field target-facing view(s)...")

            if score_field_views_save_rgb:
                _begin_render_pass(scene_mesh_visible=False)
                for batch_idx, batch in enumerate(iter_batches(target_poses, num_cameras)):
                    set_batch_poses_with_orientation(driver_cams, batch)
                    rep.orchestrator.step(rt_subframes=rt_subframes)
                    for i, pose in enumerate(batch):
                        candidate_id = int(pose["candidate_id"])
                        stem = f"{candidate_id:06d}"
                        _save_rgb(
                            np.asarray(rgb_annotators[i].get_data()),
                            target_dir / "rgb" / f"{stem}.png",
                        )
                    print(f"[SDG]   {target_id} score-field RGB batch {batch_idx}: {len(batch)} poses")
                _end_render_pass()

            if score_field_views_save_depth or score_field_views_save_bbox:
                _begin_render_pass(scene_mesh_visible=True)
                for batch_idx, batch in enumerate(iter_batches(target_poses, num_cameras)):
                    set_batch_poses_with_orientation(driver_cams, batch)
                    rep.orchestrator.step(rt_subframes=rt_subframes)
                    for i, pose in enumerate(batch):
                        candidate_id = int(pose["candidate_id"])
                        stem = f"{candidate_id:06d}"
                        record = records_by_candidate_id[candidate_id]
                        if score_field_views_save_depth:
                            _save_depth(
                                np.asarray(depth_annotators[i].get_data(), dtype=np.float32),
                                target_dir / "depth" / f"{stem}.png",
                                target_dir / "depth" / f"{stem}.npy",
                            )
                        if score_field_views_save_bbox:
                            seg_frame = seg_annotators[i].get_data()
                            bbox_xyxy = _semantic_bbox_xyxy(seg_frame, target_label)
                            record["bbox_valid"] = bbox_xyxy is not None
                            _save_person_bbox_norm(
                                target_dir / "bbox" / f"{stem}.json",
                                bbox_xyxy,
                                image_width=int(resolution[0]),
                                image_height=int(resolution[1]),
                            )
                    print(f"[SDG]   {target_id} score-field depth/bbox batch {batch_idx}: {len(batch)} poses")
                _end_render_pass()

            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "manifest.json").write_text(
                json.dumps(manifest_records, indent=2), encoding="utf-8"
            )
            if score_field_views_save_metadata:
                metadata_dir = target_dir / "metadata"
                metadata_dir.mkdir(parents=True, exist_ok=True)
                for record in manifest_records:
                    stem = f"{int(record['candidate_id']):06d}"
                    (metadata_dir / f"{stem}.json").write_text(
                        json.dumps(record, indent=2), encoding="utf-8"
                    )
            set_prim_visibility(stage, "/World/scene_collision", visible=True)

        existing_world_points_xy: list[tuple[float, float]] = []
        person_prim_path = "/SDG/Persons/person_000"
        pair_initialized = False
        context_person_prim_path = "/SDG/Persons/person_001"
        pair_labels_initialized = False

        # ------------------------------------------------------------------
        # 3. 多 pos 循环(在当前 scene 内)
        # ------------------------------------------------------------------
        for pos_idx in range(num_positions_scene):
            pos_tag = f"pos_{pos_idx:03d}"
            print(f"\n[SDG] ===== {scene_id} / {pos_tag} =====")
            pos_dir = output_root / scene_id / pos_tag
            pos_dir.mkdir(parents=True, exist_ok=True)

            # --- 3.1 放置/移动人物(按 max_attempts 重试,seed 每次换) ---
            person_xyz = None
            context_person_xyz = None
            for attempt in range(max_attempts):
                attempt_seed = base_seed + pos_idx * 10007 + attempt * 17
                try:
                    primary_result = place_person(
                        stage=stage,
                        occupancy_map=occupancy_map,
                        prim_path="/SDG/Persons/person_000",
                        seed=attempt_seed,
                        character_usd_path=primary_person_url,
                        arm_drop_degrees=float(
                            config.get("scene", {}).get("character_arm_drop_degrees", 75.0)
                        ),
                        min_obstacle_distance_m=character_min_obstacle,
                        existing_world_points_xy=existing_world_points_xy if existing_world_points_xy else None,
                        min_point_distance_m=min_position_distance,
                        reuse_existing_prim=pair_initialized,
                    )
                    person_prim_path = primary_result["prim_path"]
                    person_xyz = primary_result["position"]
                    secondary_result = None
                    for pair_attempt in range(max_pair_layout_attempts):
                        secondary_seed = attempt_seed + pair_attempt * 101 + 1
                        try:
                            secondary_result = place_person_near_anchor(
                                stage=stage,
                                occupancy_map=occupancy_map,
                                anchor_position_xy=(float(person_xyz[0]), float(person_xyz[1])),
                                prim_path=context_person_prim_path,
                                seed=secondary_seed,
                                character_usd_path=secondary_person_url,
                                arm_drop_degrees=float(
                                    config.get("scene", {}).get("character_arm_drop_degrees", 75.0)
                                ),
                                min_distance_m=pair_min_distance,
                                max_distance_m=pair_max_distance,
                                min_obstacle_distance_m=second_person_min_obstacle,
                                require_pair_connectivity=pair_require_connectivity,
                                reuse_existing_prim=pair_initialized,
                            )
                            break
                        except Exception as pair_exc:  # noqa: BLE001
                            print(
                                f"[SDG]   secondary placement attempt {pair_attempt} "
                                f"(seed={secondary_seed}) failed: {pair_exc}"
                            )
                    if secondary_result is None:
                        raise RuntimeError("failed to place secondary person near primary person")
                    context_person_xyz = secondary_result["position"]
                    pair_initialized = True
                    break
                except Exception as exc:  # noqa: BLE001
                    person_xyz = None
                    context_person_xyz = None
                    print(f"[SDG]   place_person attempt {attempt} (seed={attempt_seed}) failed: {exc}")

            if person_xyz is None:
                print(f"[SDG] {pos_tag}: failed after {max_attempts} attempts; skipping pos.")
                continue
            print(f"[SDG] {pos_tag} person position: {person_xyz}")
            if context_person_xyz is not None:
                print(f"[SDG] {pos_tag} context person position: {context_person_xyz}")
            if not pair_labels_initialized:
                add_labels("/SDG/Persons/person_000", labels="person_000", taxonomy="class")
                add_labels(context_person_prim_path, labels="person_001", taxonomy="class")
                pair_labels_initialized = True
            for _ in range(warmup_updates):
                simulation_app.update()

            targets = [
                {
                    "instance_id": "person_000",
                    "semantic_label": "person_000",
                    "position": person_xyz,
                    "prim_path": person_prim_path,
                }
            ]
            targets.append(
                {
                    "instance_id": "person_001",
                    "semantic_label": "person_001",
                    "position": context_person_xyz,
                    "prim_path": context_person_prim_path,
                }
            )

            if context_person_xyz is not None:
                pair_center_xy = (
                    0.5 * (float(person_xyz[0]) + float(context_person_xyz[0])),
                    0.5 * (float(person_xyz[1]) + float(context_person_xyz[1])),
                )
                pair_distance = (
                    (float(person_xyz[0]) - float(context_person_xyz[0])) ** 2
                    + (float(person_xyz[1]) - float(context_person_xyz[1])) ** 2
                ) ** 0.5
                shared_ring_samples = list(
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
                shared_ring_samples = [
                    {**pose, "candidate_id": int(candidate_id)}
                    for candidate_id, pose in enumerate(shared_ring_samples)
                ]
                print(
                    f"[SDG] {pos_tag} shared pair ring produced {len(shared_ring_samples)} candidate poses "
                    f"(pair_distance={pair_distance:.3f}m)"
                )
                if not shared_ring_samples:
                    print(f"[SDG] {pos_tag}: no shared pair ring samples; skipping pos.")
                    existing_world_points_xy.append((float(person_xyz[0]), float(person_xyz[1])))
                    continue

                def _xy_key(x: float, y: float) -> tuple[float, float]:
                    return (round(float(x), 6), round(float(y), 6))

                target_results: dict[str, dict] = {}
                set_prim_visibility(stage, "/World/scene_collision", visible=True)
                for _ in range(warmup_updates):
                    simulation_app.update()
                for target in targets:
                    target_id = str(target["instance_id"])
                    target_person_xyz = target["position"]
                    target_joints = get_skeleton_joint_world_positions(stage, str(target["prim_path"]))

                    target_poses = []
                    for pose in shared_ring_samples:
                        yaw_rad = _camera_yaw_from_person(
                            (float(pose["x"]), float(pose["y"])),
                            (float(target_person_xyz[0]), float(target_person_xyz[1])),
                        )
                        target_poses.append({**pose, "yaw_rad": float(yaw_rad)})

                    score_field: list[ScoreFieldPoint] = []
                    print(f"[SDG] {pos_tag} / {target_id} raycast scoring {len(target_poses)} poses...")
                    score_start = time.perf_counter()
                    for pose in target_poses:
                        ray_score = score_target_joint_visibility(
                            query=scene_query,
                            camera_xyz=(float(pose["x"]), float(pose["y"]), float(pose["camera_z"])),
                            camera_yaw_rad=float(pose["yaw_rad"]),
                            resolution=resolution,
                            focal_length=focal_length,
                            horizontal_aperture=horizontal_aperture,
                            vertical_aperture=vertical_aperture,
                            target_prim_path=str(target["prim_path"]),
                            joints=target_joints,
                        )
                        score_field.append(
                            ScoreFieldPoint(
                                x=pose["x"], y=pose["y"], z=pose["z"],
                                camera_z=pose["camera_z"], yaw_rad=pose["yaw_rad"],
                                score=float(ray_score.score), distance_m=pose["distance_m"],
                                scoring_mode="raycast_skeleton_joint_visibility",
                                candidate_id=int(pose["candidate_id"]),
                            )
                        )
                    score_elapsed = time.perf_counter() - score_start
                    print(f"[SDG] {pos_tag} / {target_id} raycast scoring elapsed: {score_elapsed:.3f}s")

                    target_results[target_id] = {
                        "target": target,
                        "score_field": score_field,
                        "score_by_xy": {_xy_key(item.x, item.y): item for item in score_field},
                    }
                    overlay_path = save_score_field_overlay(
                        occupancy_map=occupancy_map,
                        score_field=score_field,
                        out_path=pos_dir / "score_field_overlay" / f"{target_id}.png",
                        person_position_xy=(float(target_person_xyz[0]), float(target_person_xyz[1])),
                        selected_candidates=None,
                    )
                    print(f"[SDG] {pos_tag} / {target_id} wrote shared-domain overlay to {overlay_path}")
                    _save_score_field_target_views(
                        pos_dir=pos_dir,
                        pos_tag=pos_tag,
                        target=target,
                        target_poses=target_poses,
                        score_field=score_field,
                    )

                shared_rng = np.random.default_rng(base_seed + pos_idx * 7919)

                def _world_point_in_capture_view(
                    world_xyz: tuple[float, float, float],
                    camera_xy: tuple[float, float],
                    yaw_rad: float,
                    margin_px: float = 4.0,
                ) -> bool:
                    dx = float(world_xyz[0]) - float(camera_xy[0])
                    dy = float(world_xyz[1]) - float(camera_xy[1])
                    dz = float(world_xyz[2]) - camera_z
                    cos_yaw = math.cos(float(yaw_rad))
                    sin_yaw = math.sin(float(yaw_rad))
                    forward = dx * cos_yaw + dy * sin_yaw
                    lateral = -dx * sin_yaw + dy * cos_yaw
                    vertical = dz
                    if forward <= 1e-6:
                        return False

                    width, height = int(resolution[0]), int(resolution[1])
                    fx = width * float(focal_length) / max(float(horizontal_aperture), 1e-6)
                    fy = height * float(focal_length) / max(float(vertical_aperture), 1e-6)
                    cx = (width - 1) * 0.5
                    cy = (height - 1) * 0.5
                    u = cx - fx * lateral / forward
                    v = cy - fy * vertical / forward
                    return (
                        float(margin_px) <= float(u) < float(width) - float(margin_px)
                        and float(margin_px) <= float(v) < float(height) - float(margin_px)
                    )

                eligible_shared_samples = []
                for pose in shared_ring_samples:
                    key = _xy_key(pose["x"], pose["y"])
                    score_items = [
                        target_results[str(target["instance_id"])]["score_by_xy"].get(key)
                        for target in targets
                    ]
                    if any(
                        item is None
                        or not (capture_score_min <= float(item.score) <= capture_score_max)
                        for item in score_items
                    ):
                        continue

                    base_yaw = _camera_yaw_from_person((pose["x"], pose["y"]), pair_center_xy)
                    both_centers_in_view = True
                    for target in targets:
                        target_xyz = target["position"]
                        if not _world_point_in_capture_view(
                            world_xyz=(
                                float(target_xyz[0]),
                                float(target_xyz[1]),
                                float(target_xyz[2]) + 1.0,
                            ),
                            camera_xy=(float(pose["x"]), float(pose["y"])),
                            yaw_rad=base_yaw,
                        ):
                            both_centers_in_view = False
                            break
                    if not both_centers_in_view:
                        continue

                    eligible_shared_samples.append(pose)

                shared_rng.shuffle(eligible_shared_samples)
                selected_shared_samples = eligible_shared_samples[:max_cap]
                print(
                    f"[SDG] {pos_tag} shared capture eligible {len(eligible_shared_samples)} / "
                    f"{len(shared_ring_samples)} candidates with all target scores in "
                    f"[{capture_score_min:.3f}, {capture_score_max:.3f}] "
                    "and person centers in frame."
                )
                if not selected_shared_samples:
                    print(f"[SDG] {pos_tag}: no shared capture samples after score/FOV filtering; skipping pos.")
                    existing_world_points_xy.append((float(person_xyz[0]), float(person_xyz[1])))
                    continue

                capture_batch_input: list[dict] = []
                view_meta: list[dict] = []
                for pose in selected_shared_samples:
                    base_yaw = _camera_yaw_from_person((pose["x"], pose["y"]), pair_center_xy)
                    yaw = float(base_yaw)
                    quat = _yaw_to_world_quaternion(yaw)
                    capture_batch_input.append(
                        {
                            "x": float(pose["x"]),
                            "y": float(pose["y"]),
                            "camera_z": camera_z,
                            "yaw_rad": float(yaw),
                            "candidate_id": int(pose["candidate_id"]),
                        }
                    )
                    view_meta.append(
                        {
                            "candidate_id": int(pose["candidate_id"]),
                            "candidate_key": _xy_key(pose["x"], pose["y"]),
                            "camera_position": (float(pose["x"]), float(pose["y"]), camera_z),
                            "camera_orientation_wxyz": quat,
                            "base_yaw_rad": float(base_yaw),
                            "yaw_jitter_rad": 0.0,
                        }
                    )

                print(f"[SDG] {pos_tag} shared Capture Pass A: {len(capture_batch_input)} views...")
                _begin_render_pass(scene_mesh_visible=False)
                rgb_frames: list = [None] * len(capture_batch_input)
                passA_start = time.perf_counter()
                view_idx = 0
                for batch_idx, batch in enumerate(iter_batches(capture_batch_input, num_cameras)):
                    set_batch_poses_with_orientation(driver_cams, batch)
                    rep.orchestrator.step(rt_subframes=rt_subframes)
                    for i in range(len(batch)):
                        rgb_frames[view_idx + i] = np.asarray(rgb_annotators[i].get_data())
                    view_idx += len(batch)
                    print(f"[SDG]   shared Pass A batch {batch_idx}: {len(batch)} poses")
                passA_elapsed = time.perf_counter() - passA_start
                _end_render_pass()
                print(f"[SDG] {pos_tag} shared Pass A elapsed: {passA_elapsed:.3f}s")

                print(f"[SDG] {pos_tag} shared Capture Pass B: {len(capture_batch_input)} views...")
                _begin_render_pass(scene_mesh_visible=True)
                depth_frames: list = [None] * len(capture_batch_input)
                seg_frames: list = [None] * len(capture_batch_input)
                passB_start = time.perf_counter()
                view_idx = 0
                for batch_idx, batch in enumerate(iter_batches(capture_batch_input, num_cameras)):
                    set_batch_poses_with_orientation(driver_cams, batch)
                    rep.orchestrator.step(rt_subframes=rt_subframes)
                    for i in range(len(batch)):
                        depth_frames[view_idx + i] = np.asarray(
                            depth_annotators[i].get_data(), dtype=np.float32
                        )
                        seg_frames[view_idx + i] = seg_annotators[i].get_data()
                    view_idx += len(batch)
                    print(f"[SDG]   shared Pass B batch {batch_idx}: {len(batch)} poses")
                passB_elapsed = time.perf_counter() - passB_start
                _end_render_pass()
                print(f"[SDG] {pos_tag} shared Pass B elapsed: {passB_elapsed:.3f}s")

                target_observations: dict[str, list[dict]] = {str(t["instance_id"]): [] for t in targets}
                target_scores: dict[str, dict[str, float]] = {str(t["instance_id"]): {} for t in targets}
                target_selected_points: dict[str, list[ScoreFieldPoint]] = {str(t["instance_id"]): [] for t in targets}

                output_idx = 0
                for idx, meta in enumerate(view_meta):
                    seg_frame = seg_frames[idx]
                    target_bboxes = {
                        str(target["instance_id"]): _semantic_bbox_xyxy(seg_frame, str(target["semantic_label"]))
                        for target in targets
                    }
                    if any(bbox is None for bbox in target_bboxes.values()):
                        print(f"[SDG] {pos_tag} shared view {idx:03d}: missing target bbox; skipping view.")
                        continue

                    cam_pos = meta["camera_position"]
                    orient = meta["camera_orientation_wxyz"]
                    rgb = rgb_frames[idx]
                    depth = depth_frames[idx]
                    stem = f"{output_idx:03d}"

                    rgb_path = pos_dir / "rgb" / f"{stem}.png"
                    depth_png_path = pos_dir / "depth" / f"{stem}.png"
                    depth_npy_path = pos_dir / "depth" / f"{stem}.npy"
                    ground_mask_path = pos_dir / "ground_mask" / f"{stem}.png"
                    _save_rgb(rgb, rgb_path)
                    _save_depth(depth, depth_png_path, depth_npy_path)

                    ground_mask = _compute_ground_mask(
                        depth_m=depth,
                        camera_position=cam_pos,
                        camera_orientation_wxyz=orient,
                        resolution=resolution,
                        focal_length=focal_length,
                        horizontal_aperture=horizontal_aperture,
                        vertical_aperture=vertical_aperture,
                        occupancy_map=occupancy_map,
                    )
                    _save_ground_mask(ground_mask, ground_mask_path)

                    for target in targets:
                        target_id = str(target["instance_id"])
                        result = target_results[target_id]
                        score_field = result["score_field"]
                        score_item = result["score_by_xy"].get(meta["candidate_key"])
                        sampling_score = float(score_item.score) if score_item is not None else 0.0
                        if score_item is not None:
                            target_selected_points[target_id].append(score_item)

                        score_map_path = pos_dir / "score_map" / target_id / f"{stem}.npy"
                        valid_mask_path = pos_dir / "valid_mask" / target_id / f"{stem}.npy"
                        yaw_map_path = pos_dir / "yaw_map" / target_id / f"{stem}.npy"
                        person_bbox_path = pos_dir / "person_bbox" / target_id / f"{stem}.json"

                        score_map_arr, valid_mask_arr, yaw_map_arr = _compute_score_map(
                            score_field=score_field,
                            depth_m=depth,
                            ground_mask=ground_mask,
                            camera_position=cam_pos,
                            camera_orientation_wxyz=orient,
                            resolution=resolution,
                            focal_length=focal_length,
                            horizontal_aperture=horizontal_aperture,
                            vertical_aperture=vertical_aperture,
                        )
                        _save_score_map(score_map_arr, score_map_path)
                        _save_valid_mask(valid_mask_arr, valid_mask_path)
                        _save_yaw_map(yaw_map_arr, yaw_map_path)

                        _save_person_bbox_norm(
                            person_bbox_path,
                            target_bboxes[target_id],
                            image_width=int(resolution[0]),
                            image_height=int(resolution[1]),
                        )

                        observation = {
                            "idx": output_idx,
                            "rgb_path": str(rgb_path),
                            "depth_path": str(depth_png_path),
                            "depth_npy_path": str(depth_npy_path),
                            "ground_mask_path": str(ground_mask_path),
                            "score_map_path": str(score_map_path),
                            "valid_mask_path": str(valid_mask_path),
                            "yaw_map_path": str(yaw_map_path),
                            "person_bbox_path": str(person_bbox_path),
                            "candidate_id": int(meta["candidate_id"]),
                            "candidate_key": [float(v) for v in meta["candidate_key"]],
                            "camera_position": [float(v) for v in cam_pos],
                            "camera_orientation_wxyz": [float(v) for v in orient],
                            "base_yaw_rad": float(meta["base_yaw_rad"]),
                            "yaw_jitter_rad": float(meta["yaw_jitter_rad"]),
                            "sampling_score": sampling_score,
                            "reid_score": sampling_score,
                            "scoring_mode": score_item.scoring_mode if score_item is not None else "missing",
                            "shared_capture": True,
                        }
                        target_observations[target_id].append(observation)
                        target_scores[target_id][f"{stem}.png"] = sampling_score
                    output_idx += 1

                for target in targets:
                    target_id = str(target["instance_id"])
                    target_semantic_label = str(target["semantic_label"])
                    target_person_xyz = target["position"]
                    score_field = target_results[target_id]["score_field"]

                    save_score_field_overlay(
                        occupancy_map=occupancy_map,
                        score_field=score_field,
                        out_path=pos_dir / "score_field_overlay" / f"{target_id}.png",
                        person_position_xy=(float(target_person_xyz[0]), float(target_person_xyz[1])),
                        selected_candidates=target_selected_points[target_id] or None,
                    )

                    metadata = {
                        "scene_id": scene_id,
                        "stage_url": stage_url,
                        "pos_idx": pos_idx,
                        "target_instance_id": target_id,
                        "target_semantic_label": target_semantic_label,
                        "target_person_position": [float(v) for v in target_person_xyz],
                        "context_person_position": (
                            [float(v) for v in context_person_xyz] if context_person_xyz is not None else None
                        ),
                        "context_person_prim_path": context_person_prim_path,
                        "pair_center_xy": [float(v) for v in pair_center_xy],
                        "pair_distance_m": float(pair_distance),
                        "shared_score_field_domain": True,
                        "shared_capture": True,
                        "img_size": [int(resolution[1]), int(resolution[0])],
                        "score_field_size": len(score_field),
                        "camera": {
                            "camera_height": camera_z,
                            "focal_length": focal_length,
                            "horizontal_aperture": horizontal_aperture,
                            "vertical_aperture": vertical_aperture,
                            "resolution": [int(resolution[0]), int(resolution[1])],
                        },
                        "passA_elapsed_sec": float(passA_elapsed),
                        "passB_elapsed_sec": float(passB_elapsed),
                        "observations": target_observations[target_id],
                    }
                    metadata_path = pos_dir / "metadata" / f"{target_id}.json"
                    scores_path = pos_dir / "scores" / f"{target_id}.json"
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    scores_path.parent.mkdir(parents=True, exist_ok=True)
                    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    scores_path.write_text(
                        json.dumps(target_scores[target_id], indent=2), encoding="utf-8"
                    )
                    print(f"[SDG] {pos_tag} / {target_id} wrote shared metadata.json + scores.json")

                existing_world_points_xy.append((float(person_xyz[0]), float(person_xyz[1])))
                continue

            raise RuntimeError("secondary person missing after successful pair placement")

        # --- 2.4 拆相机池(下一场景会 open_stage 并重建) ---
        print(f"[SDG] {scene_id}: tearing down camera pool")
        try:
            teardown_camera_pool(
                stage=stage,
                annotators_by_name=annotators_by_name,
                render_products=render_products,
                scope_path="/SDG/Cameras",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[SDG] {scene_id}: teardown warned: {exc}")

    # ------------------------------------------------------------------
    # 4. 运行摘要 + 关闭/保持 GUI
    # ------------------------------------------------------------------
    if filtered_scenes:
        print(f"\n[SDG] {len(filtered_scenes)} scene(s) skipped by scene_filter:")
        for sid, reason in filtered_scenes:
            print(f"  - {sid}: {reason}")
    if failed_scenes:
        print(f"\n[SDG] {len(failed_scenes)} scene(s) failed:")
        for sid, err in failed_scenes:
            print(f"  - {sid}: {err}")

    if config.get("close_app_after_run", True):
        simulation_app.close()
    else:
        print("[SDG] GUI 保持打开,关闭窗口退出。")
        while simulation_app.is_running():
            simulation_app.update()
        simulation_app.close()


if __name__ == "__main__":
    main()
