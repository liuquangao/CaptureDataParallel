# Copyright (c) 2026, Fusion Intelligence Labs, University of Exeter. All rights reserved.
#
# 并行采集入口。流程与 debug_replicator_cameras.py 保持一致,所有可复用函数都从
# utils/ 引入,不再依赖 collector/ 下的模块。
#
# 输出 <backend_params.output_dir>/<scene_id>/pos_{idx:03d}/:
#   score_field.json                (Stage 1)
#   score_field_overlay.png         (Stage 1, Stage 3 叠加 selected 后覆写)
#   rgb/{idx:03d}.png               (Stage 3)
#   depth/{idx:03d}.png             (uint16 mm)
#   depth/{idx:03d}.npy             (float32 meters)
#   ground_mask/{idx:03d}.png
#   score_map/{idx:03d}.npy
#   valid_mask/{idx:03d}.npy
#   yaw_map/{idx:03d}.npy
#   person_bbox/{idx:03d}.json
#   metadata.json
#   scores.json

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import asdict
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
    from utils.person_placement import place_person
    from utils.replicator_tools import (
        create_camera_pool,
        iter_batches,
        set_batch_poses,
        set_batch_poses_with_orientation,
        set_prim_visibility,
        set_render_products_updates_enabled,
        teardown_camera_pool,
    )
    from utils.ring_sampling import (
        ScoreFieldPoint,
        _has_full_width_occupancy_visibility,
        iter_ring_camera_samples,
        select_capture_candidates,
    )
    from utils.scene_selection import check_scene_filter, resolve_scene_configs
    from utils.score import read_person_counts

    carb_settings = carb.settings.get_settings()

    def _set_low_quality():
        carb_settings.set("/rtx/rendermode", "RayTracedLighting")
        carb_settings.set("rtx/post/dlss/execMode", 0)

    def _set_high_quality():
        carb_settings.set("/rtx/rendermode", "PathTracing")
        carb_settings.set("rtx/post/dlss/execMode", 2)

    rep.set_global_seed(int(config.get("sampling", {}).get("seed", 42)))
    rep.orchestrator.set_capture_on_play(False)
    _set_low_quality()

    # ------------------------------------------------------------------
    # 0. 通用参数(跨场景共享)
    # ------------------------------------------------------------------
    score_field_cfg = config["score_field"]
    sampling_cfg = config.get("sampling", {})
    camera_cfg = config["camera"]
    resolution = tuple(camera_cfg["resolution"])
    focal_length = float(camera_cfg["focal_length"])
    vertical_aperture = float(camera_cfg["vertical_aperture"])
    horizontal_aperture = float(_REP_DEFAULT_HORIZONTAL_APERTURE)
    camera_z = float(camera_cfg.get("camera_height", 0.4))

    num_cameras = int(config["num_cameras"])
    rt_subframes = int(config.get("rt_subframes", 2))

    num_positions = int(config.get("num_positions", 1))
    base_seed = int(sampling_cfg.get("seed", 42))
    min_position_distance = float(sampling_cfg.get("min_position_distance_m", 3.0))
    character_min_obstacle = float(sampling_cfg.get("character_min_obstacle_distance_m", 0.2))
    max_attempts = max(1, int(sampling_cfg.get("max_attempts_per_position", 2)))
    body_width_m = float(score_field_cfg["occupancy_full_visibility_width_m"])
    max_cap = int(score_field_cfg.get("max_capture_candidates", 16))
    capture_score_min = float(score_field_cfg["capture_score_min"])
    capture_score_max = float(score_field_cfg["capture_score_max"])
    output_root = Path(config["backend_params"]["output_dir"])

    # ------------------------------------------------------------------
    # 1. 解析要采的场景列表
    # ------------------------------------------------------------------
    scene_configs = resolve_scene_configs(config)
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

        # --- 2.2 打开 stage + 场景级设置 ---
        print(f"[SDG] {scene_id}: loading stage {stage_url}")
        if not open_stage(stage_url):
            print(f"[SDG] {scene_id}: open_stage failed; skipping scene.")
            failed_scenes.append((scene_id, "open_stage failed"))
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

        warmup_updates = int(config.get("scene", {}).get("warmup_updates", 4))

        def _begin_render_pass(scene_mesh_visible: bool):
            """关 hydra → 切 /World/scene_collision → 开 hydra → warmup 数帧。渲染完必须配对 _end_render_pass。"""
            set_render_products_updates_enabled(render_products, False)
            set_prim_visibility(stage, "/World/scene_collision", visible=scene_mesh_visible)
            set_render_products_updates_enabled(render_products, True)
            for _ in range(warmup_updates):
                simulation_app.update()

        def _end_render_pass():
            set_render_products_updates_enabled(render_products, False)

        existing_world_points_xy: list[tuple[float, float]] = []
        person_prim_path = "/SDG/Person"
        person_initialized = False

        # ------------------------------------------------------------------
        # 3. 多 pos 循环(在当前 scene 内)
        # ------------------------------------------------------------------
        for pos_idx in range(num_positions):
            pos_tag = f"pos_{pos_idx:03d}"
            print(f"\n[SDG] ===== {scene_id} / {pos_tag} =====")
            pos_dir = output_root / scene_id / pos_tag
            pos_dir.mkdir(parents=True, exist_ok=True)

            # 每个 pos 都用低画质跑 score field(Pass 1/2 只要分割结果)
            _set_low_quality()

            # --- 3.1 放置/移动人物(按 max_attempts 重试,seed 每次换) ---
            person_xyz = None
            for attempt in range(max_attempts):
                attempt_seed = base_seed + pos_idx * 10007 + attempt * 17
                try:
                    debug_char = place_person(
                        stage=stage,
                        occupancy_map=occupancy_map,
                        prim_path=person_prim_path,
                        seed=attempt_seed,
                        character_usd_path=config["person"]["url"],
                        arm_drop_degrees=float(config.get("scene", {}).get("character_arm_drop_degrees", 75.0)),
                        min_obstacle_distance_m=character_min_obstacle,
                        existing_world_points_xy=existing_world_points_xy if existing_world_points_xy else None,
                        min_point_distance_m=min_position_distance,
                        reuse_existing_prim=person_initialized,
                    )
                    person_xyz = debug_char["position"]
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"[SDG]   place_person attempt {attempt} (seed={attempt_seed}) failed: {exc}")

            if person_xyz is None:
                print(f"[SDG] {pos_tag}: failed after {max_attempts} attempts; skipping pos.")
                continue
            print(f"[SDG] {pos_tag} person position: {person_xyz}")
            if not person_initialized:
                add_labels(person_prim_path, labels="person", taxonomy="class")
                person_initialized = True
            for _ in range(warmup_updates):
                simulation_app.update()

            # --- 3.2 环形采样 + occupancy shortcut ---
            ring_samples = list(
                iter_ring_camera_samples(
                    occupancy_map=occupancy_map,
                    person_position_xy=(person_xyz[0], person_xyz[1]),
                    camera_height_m=camera_z,
                    min_radius_m=float(score_field_cfg["min_radius_m"]),
                    max_radius_m=float(score_field_cfg["max_radius_m"]),
                    grid_step_m=float(score_field_cfg["grid_step_m"]),
                    min_obstacle_distance_m=float(score_field_cfg["camera_min_obstacle_distance_m"]),
                )
            )
            print(f"[SDG] {pos_tag} ring produced {len(ring_samples)} candidate poses")
            if not ring_samples:
                print(f"[SDG] {pos_tag}: no ring samples; skipping pos.")
                existing_world_points_xy.append((float(person_xyz[0]), float(person_xyz[1])))
                continue

            certain_indices: list[int] = []
            uncertain_indices: list[int] = []
            for i, pose in enumerate(ring_samples):
                if _has_full_width_occupancy_visibility(
                    occupancy_map=occupancy_map,
                    person_position_xy=(person_xyz[0], person_xyz[1]),
                    camera_position_xy=(pose["x"], pose["y"]),
                    body_width_m=body_width_m,
                ):
                    certain_indices.append(i)
                else:
                    uncertain_indices.append(i)
            uncertain_poses = [ring_samples[i] for i in uncertain_indices]
            print(
                f"[SDG] {pos_tag} occupancy shortcut: {len(certain_indices)} certain, "
                f"{len(uncertain_indices)} uncertain -> rendering."
            )

            # --- 3.3 Pass 1/2 ---
            uncertain_total_counts: list[int] = []
            uncertain_visible_counts: list[int] = []
            pass1_elapsed = 0.0
            pass2_elapsed = 0.0
            look_at_xyz = (float(person_xyz[0]), float(person_xyz[1]), float(person_xyz[2]) + 1.0)

            if uncertain_poses:
                print(f"[SDG] {pos_tag} Pass 1: hiding mesh, {len(uncertain_poses)} unoccluded views...")
                _begin_render_pass(scene_mesh_visible=False)
                pass1_start = time.perf_counter()
                for batch in iter_batches(uncertain_poses, num_cameras):
                    set_batch_poses(driver_cams, batch, look_at_xyz)
                    rep.orchestrator.step(rt_subframes=rt_subframes)
                    batch_counts = read_person_counts(seg_annotators)
                    uncertain_total_counts.extend(batch_counts[: len(batch)])
                pass1_elapsed = time.perf_counter() - pass1_start
                _end_render_pass()
                print(f"[SDG] {pos_tag} Pass 1 elapsed: {pass1_elapsed:.3f}s")

                print(f"[SDG] {pos_tag} Pass 2: capturing {len(uncertain_poses)} occluded views...")
                _begin_render_pass(scene_mesh_visible=True)
                pass2_start = time.perf_counter()
                for batch in iter_batches(uncertain_poses, num_cameras):
                    set_batch_poses(driver_cams, batch, look_at_xyz)
                    rep.orchestrator.step(rt_subframes=rt_subframes)
                    batch_counts = read_person_counts(seg_annotators)
                    uncertain_visible_counts.extend(batch_counts[: len(batch)])
                pass2_elapsed = time.perf_counter() - pass2_start
                _end_render_pass()
                print(f"[SDG] {pos_tag} Pass 2 elapsed: {pass2_elapsed:.3f}s")

            # --- 3.4 合并 score_field ---
            score_field: list[ScoreFieldPoint] = []
            poses_payload: list[dict | None] = [None] * len(ring_samples)

            for i in certain_indices:
                pose = ring_samples[i]
                sf = ScoreFieldPoint(
                    x=pose["x"], y=pose["y"], z=pose["z"],
                    camera_z=pose["camera_z"], yaw_rad=pose["yaw_rad"],
                    score=1.0, distance_m=pose["distance_m"],
                    visible_person_pixels=1, total_person_pixels=1,
                    scoring_mode="occupancy_full_visibility",
                )
                score_field.append(sf)
                poses_payload[i] = {"idx": int(i), **asdict(sf)}

            for idx_in_uncertain, i in enumerate(uncertain_indices):
                pose = ring_samples[i]
                vis = int(uncertain_visible_counts[idx_in_uncertain])
                total = int(uncertain_total_counts[idx_in_uncertain])
                score = float(vis) / float(total) if total > 0 else 0.0
                sf = ScoreFieldPoint(
                    x=pose["x"], y=pose["y"], z=pose["z"],
                    camera_z=pose["camera_z"], yaw_rad=pose["yaw_rad"],
                    score=score, distance_m=pose["distance_m"],
                    visible_person_pixels=vis, total_person_pixels=total,
                    scoring_mode="segmentation_visibility",
                )
                score_field.append(sf)
                poses_payload[i] = {"idx": int(i), **asdict(sf)}

            # --- 3.5 写 score_field.json ---
            score_field_payload = {
                "person_position": [float(v) for v in person_xyz],
                "num_poses": len(ring_samples),
                "num_certain": len(certain_indices),
                "num_uncertain": len(uncertain_indices),
                "pass1_elapsed_sec": float(pass1_elapsed),
                "pass2_elapsed_sec": float(pass2_elapsed),
                "occupancy_full_visibility_width_m": body_width_m,
                "ring_params": {
                    "min_radius_m": float(score_field_cfg["min_radius_m"]),
                    "max_radius_m": float(score_field_cfg["max_radius_m"]),
                    "grid_step_m": float(score_field_cfg["grid_step_m"]),
                    "camera_min_obstacle_distance_m": float(score_field_cfg["camera_min_obstacle_distance_m"]),
                },
                "poses": poses_payload,
            }
            score_field_path = pos_dir / "score_field.json"
            score_field_path.write_text(json.dumps(score_field_payload, indent=2), encoding="utf-8")
            print(f"[SDG] {pos_tag} wrote {score_field_path}")

            # --- 3.6 Stage 3: 挑 N 个 candidate + capture ---
            selected = select_capture_candidates(
                score_field=score_field,
                score_min=capture_score_min,
                score_max=capture_score_max,
                seed=base_seed + pos_idx * 7919,
                max_candidates=max_cap,
                fallback_to_nearest=False,
            )
            print(f"[SDG] {pos_tag} selected {len(selected)} / {len(score_field)} candidates for capture.")

            overlay_path = save_score_field_overlay(
                occupancy_map=occupancy_map,
                score_field=score_field,
                out_path=pos_dir / "score_field_overlay.png",
                person_position_xy=(float(person_xyz[0]), float(person_xyz[1])),
                selected_candidates=selected if selected else None,
            )
            print(f"[SDG] {pos_tag} wrote overlay to {overlay_path}")

            if not selected:
                print(f"[SDG] {pos_tag}: no candidates in score range; skipping capture.")
                existing_world_points_xy.append((float(person_xyz[0]), float(person_xyz[1])))
                continue

            capture_batch_input: list[dict] = []
            view_meta: list[dict] = []
            for sf in selected:
                yaw = _camera_yaw_from_person((sf.x, sf.y), (person_xyz[0], person_xyz[1]))
                quat = _yaw_to_world_quaternion(yaw)
                capture_batch_input.append(
                    {"x": float(sf.x), "y": float(sf.y), "camera_z": camera_z, "yaw_rad": float(yaw)}
                )
                view_meta.append(
                    {
                        "camera_position": (float(sf.x), float(sf.y), camera_z),
                        "camera_orientation_wxyz": quat,
                        "sampling_score": float(sf.score),
                        "scoring_mode": str(sf.scoring_mode),
                        "visible_person_pixels": int(sf.visible_person_pixels),
                        "total_person_pixels": int(sf.total_person_pixels),
                    }
                )

            # 捕获前切高画质
            print(f"[SDG] {pos_tag} switching render quality to PathTracing + DLSS execMode=2")
            _set_high_quality()

            # Pass A: scene_collision 隐藏 -> RGB + semantic seg (person bbox)
            print(f"[SDG] {pos_tag} Capture Pass A: {len(capture_batch_input)} views...")
            _begin_render_pass(scene_mesh_visible=False)
            rgb_frames: list = [None] * len(capture_batch_input)
            seg_frames: list = [None] * len(capture_batch_input)

            passA_start = time.perf_counter()
            view_idx = 0
            for batch_idx, batch in enumerate(iter_batches(capture_batch_input, num_cameras)):
                set_batch_poses_with_orientation(driver_cams, batch)
                rep.orchestrator.step(rt_subframes=rt_subframes)
                for i in range(len(batch)):
                    rgb_frames[view_idx + i] = np.asarray(rgb_annotators[i].get_data())
                    seg_frames[view_idx + i] = seg_annotators[i].get_data()
                view_idx += len(batch)
                print(f"[SDG]   Pass A batch {batch_idx}: {len(batch)} poses")
            passA_elapsed = time.perf_counter() - passA_start
            _end_render_pass()
            print(f"[SDG] {pos_tag} Pass A elapsed: {passA_elapsed:.3f}s")

            # Pass B: scene_collision 可见 -> depth
            print(f"[SDG] {pos_tag} Capture Pass B: {len(capture_batch_input)} views...")
            _begin_render_pass(scene_mesh_visible=True)
            depth_frames: list = [None] * len(capture_batch_input)

            passB_start = time.perf_counter()
            view_idx = 0
            for batch_idx, batch in enumerate(iter_batches(capture_batch_input, num_cameras)):
                set_batch_poses_with_orientation(driver_cams, batch)
                rep.orchestrator.step(rt_subframes=rt_subframes)
                for i in range(len(batch)):
                    depth_frames[view_idx + i] = np.asarray(
                        depth_annotators[i].get_data(), dtype=np.float32
                    )
                view_idx += len(batch)
                print(f"[SDG]   Pass B batch {batch_idx}: {len(batch)} poses")
            passB_elapsed = time.perf_counter() - passB_start
            _end_render_pass()
            print(f"[SDG] {pos_tag} Pass B elapsed: {passB_elapsed:.3f}s")

            # --- 逐视图后处理 + 落盘 ---
            observations: list[dict] = []
            scores_map: dict[str, float] = {}
            for idx, meta in enumerate(view_meta):
                cam_pos = meta["camera_position"]
                orient = meta["camera_orientation_wxyz"]
                rgb = rgb_frames[idx]
                depth = depth_frames[idx]
                seg_frame = seg_frames[idx]

                stem = f"{idx:03d}"
                rgb_path = pos_dir / "rgb" / f"{stem}.png"
                depth_png_path = pos_dir / "depth" / f"{stem}.png"
                depth_npy_path = pos_dir / "depth" / f"{stem}.npy"
                ground_mask_path = pos_dir / "ground_mask" / f"{stem}.png"
                score_map_path = pos_dir / "score_map" / f"{stem}.npy"
                valid_mask_path = pos_dir / "valid_mask" / f"{stem}.npy"
                yaw_map_path = pos_dir / "yaw_map" / f"{stem}.npy"
                person_bbox_path = pos_dir / "person_bbox" / f"{stem}.json"

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

                bbox = _semantic_bbox_xyxy(seg_frame, "person")
                _save_person_bbox_norm(
                    person_bbox_path,
                    bbox,
                    image_width=int(resolution[0]),
                    image_height=int(resolution[1]),
                )

                observations.append(
                    {
                        "idx": idx,
                        "rgb_path": str(rgb_path),
                        "depth_path": str(depth_png_path),
                        "depth_npy_path": str(depth_npy_path),
                        "ground_mask_path": str(ground_mask_path),
                        "score_map_path": str(score_map_path),
                        "valid_mask_path": str(valid_mask_path),
                        "yaw_map_path": str(yaw_map_path),
                        "person_bbox_path": str(person_bbox_path),
                        "camera_position": [float(v) for v in cam_pos],
                        "camera_orientation_wxyz": [float(v) for v in orient],
                        "sampling_score": float(meta["sampling_score"]),
                        "reid_score": float(meta["sampling_score"]),
                        "visible_person_pixels": int(meta["visible_person_pixels"]),
                        "total_person_pixels": int(meta["total_person_pixels"]),
                        "scoring_mode": meta["scoring_mode"],
                    }
                )
                scores_map[f"{stem}.png"] = float(meta["sampling_score"])

            metadata = {
                "scene_id": scene_id,
                "stage_url": stage_url,
                "pos_idx": pos_idx,
                "person_position": [float(v) for v in person_xyz],
                "person_prim_path": person_prim_path,
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
                "observations": observations,
            }
            (pos_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            (pos_dir / "scores.json").write_text(json.dumps(scores_map, indent=2), encoding="utf-8")
            print(f"[SDG] {pos_tag} wrote metadata.json + scores.json")

            # 记录该 pos 的人物位置,供下一个 pos 保持最小距离
            existing_world_points_xy.append((float(person_xyz[0]), float(person_xyz[1])))

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
