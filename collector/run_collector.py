from __future__ import annotations

import argparse
import gc
import json
from math import cos, sin
import random
import shutil
import sys
from pathlib import Path
from time import perf_counter

import yaml

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collector.scene_loader import load_scene
from collector.debug_viz import ensure_scene_query_support, place_debug_character
from collector.occupancy import load_interiorgs_occupancy_map, save_score_field_overlay, summarize_occupancy_map
from collector.camera_capture import (
    CameraCaptureConfig,
    capture_candidate_views,
    create_collector_camera,
    estimate_candidate_visibility_ratios_batch,
)
from collector.score_field import generate_segmentation_score_field, select_capture_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim point-nav ReID collector")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(output_root: str, scene_name: str) -> Path:
    scene_dir = Path(output_root) / scene_name
    legacy_debug_dir = scene_dir / "debug"
    if legacy_debug_dir.exists():
        shutil.rmtree(legacy_debug_dir)
    legacy_meta_dir = scene_dir / "meta"
    if legacy_meta_dir.exists():
        shutil.rmtree(legacy_meta_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    return scene_dir


def ensure_position_output_dirs(scene_root: Path, position_index: int) -> Path:
    pos_dir = scene_root / f"pos_{position_index:03d}"
    for rel in ("rgb", "depth", "ground_mask", "score_map", "valid_mask", "yaw_map"):
        (pos_dir / rel).mkdir(parents=True, exist_ok=True)
    return pos_dir


def _relpath(path: str | Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(root.resolve()))


def _build_camera_config(camera_cfg_raw: dict, debug_character: dict) -> CameraCaptureConfig:
    return CameraCaptureConfig(
        prim_path=camera_cfg_raw.get("prim_path", "/World/CollectorCamera"),
        resolution=tuple(camera_cfg_raw.get("resolution", [600, 400])),
        camera_height=float(camera_cfg_raw.get("camera_height", 0.4)),
        visibility_settle_updates=int(camera_cfg_raw.get("visibility_settle_updates", camera_cfg_raw.get("warmup_updates", 8))),
        focus_distance=float(camera_cfg_raw.get("focus_distance", 400.0)),
        focal_length=float(camera_cfg_raw.get("focal_length", 15.0)),
        horizontal_aperture=float(camera_cfg_raw.get("horizontal_aperture", 3.6)),
        vertical_aperture=float(camera_cfg_raw.get("vertical_aperture", 2.4)),
        near_clipping_distance=float(camera_cfg_raw.get("near_clipping_distance", 0.01)),
        depth_unit=str(camera_cfg_raw.get("depth_unit", "meters")),
        look_direction_mode=str(camera_cfg_raw.get("look_direction_mode", "look_at_target")),
        target_position_xy=(
            float(debug_character["position"][0]),
            float(debug_character["position"][1]),
        ),
        debug_logging=bool(camera_cfg_raw.get("debug_logging", False)),
        enable_rgb=bool(camera_cfg_raw.get("enable_rgb", True)),
        enable_depth=bool(camera_cfg_raw.get("enable_depth", True)),
        enable_instance_segmentation=bool(camera_cfg_raw.get("enable_instance_segmentation", True)),
        yaw_jitter_margin=float(camera_cfg_raw.get("yaw_jitter_margin", 0.0)),
    )


def _set_render_mode(simulation_app, mode: str, warmup_updates: int = 4) -> None:
    import carb.settings

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", str(mode))
    for _ in range(int(warmup_updates)):
        simulation_app.update()


def _collect_single_position(
    simulation_app,
    timeline,
    stage,
    occupancy_map,
    cfg: dict,
    scene_cfg: dict,
    scene_root: Path,
    position_index: int,
) -> tuple[Path, int]:
    sampling_cfg = cfg.get("sampling", {})
    base_camera_cfg_raw = cfg.get("camera", {})
    scoring_camera_cfg_raw = dict(cfg.get("scoring_camera", base_camera_cfg_raw))
    capture_camera_cfg_raw = dict(cfg.get("capture_camera", base_camera_cfg_raw))
    scoring_camera_cfg_raw.setdefault("prim_path", "/World/ScoringCamera")
    scoring_camera_cfg_raw.setdefault("enable_rgb", False)
    scoring_camera_cfg_raw.setdefault("enable_depth", False)
    scoring_camera_cfg_raw.setdefault("enable_instance_segmentation", True)
    capture_camera_cfg_raw.setdefault("prim_path", "/World/CollectorCamera")
    capture_camera_cfg_raw.setdefault("enable_rgb", True)
    capture_camera_cfg_raw.setdefault("enable_depth", True)
    capture_camera_cfg_raw.setdefault("enable_instance_segmentation", True)
    ring_cfg = cfg.get("score_field", {})
    pos_dir = ensure_position_output_dirs(scene_root, position_index)

    debug_character = place_debug_character(
        stage,
        occupancy_map,
        seed=int(sampling_cfg.get("seed", 0)) + 1000 + int(position_index),
        character_usd_path=str(
            scene_cfg.get(
                "character_usd_path",
                "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/"
                "Assets/Isaac/6.0/Isaac/People/Characters/F_Business_02/F_Business_02.usd",
            )
        ),
        arm_drop_degrees=float(scene_cfg.get("character_arm_drop_degrees", 75.0)),
    )
    print(f"  [Position {position_index:03d}] debug_character: {json.dumps(debug_character, ensure_ascii=True)}", flush=True)

    score_min = float(ring_cfg.get("capture_score_min", 0.2))
    score_max = float(ring_cfg.get("capture_score_max", 0.6))
    score_overlay_path = pos_dir / "score_field_overlay.png"
    scoring_camera_cfg = _build_camera_config(scoring_camera_cfg_raw, debug_character)
    capture_camera_cfg = _build_camera_config(capture_camera_cfg_raw, debug_character)
    scoring_camera = create_collector_camera(scoring_camera_cfg)
    collector_camera = create_collector_camera(capture_camera_cfg)
    print(f"[Step 3.{position_index:03d}] generating segmentation-based ring score field around person", flush=True)
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    scene_collision_exists = bool(
        stage.GetPrimAtPath("/World/scene_collision") and stage.GetPrimAtPath("/World/scene_collision").IsValid()
    )
    geometry_root = None
    for candidate_path in ("/World/volume/mesh", "/World/mesh", "/World/GroundPlane"):
        prim = stage.GetPrimAtPath(candidate_path)
        if prim and prim.IsValid():
            geometry_root = candidate_path
            break

    def _segmentation_visibility_batch_scorer(samples):
        poses = []
        for sample in samples:
            camera_orientation = (
                float(cos(float(sample["yaw_rad"]) * 0.5)),
                0.0,
                0.0,
                float(sin(float(sample["yaw_rad"]) * 0.5)),
            )
            poses.append(((sample["x"], sample["y"], sample["camera_z"]), camera_orientation))
        return estimate_candidate_visibility_ratios_batch(
            simulation_app=simulation_app,
            timeline=timeline,
            camera=scoring_camera,
            poses=poses,
            visibility_settle_updates=scoring_camera_cfg.visibility_settle_updates,
            person_prim_path=str(debug_character["prim_path"]),
            scene_collision_exists=scene_collision_exists,
            geometry_root=geometry_root,
        )

    score_field_start = perf_counter()
    score_field = generate_segmentation_score_field(
        occupancy_map=occupancy_map,
        person_position_xy=(
            float(debug_character["position"][0]),
            float(debug_character["position"][1]),
        ),
        camera_height_m=float(scoring_camera_cfg_raw.get("camera_height", 0.4)),
        min_radius_m=float(ring_cfg.get("min_radius_m", 0.8)),
        max_radius_m=float(ring_cfg.get("max_radius_m", 3.0)),
        grid_step_m=float(ring_cfg.get("grid_step_m", 0.2)),
        visibility_batch_scorer=_segmentation_visibility_batch_scorer,
        occupancy_full_visibility_width_m=float(ring_cfg.get("occupancy_full_visibility_width_m", 0.25)),
    )
    score_field_elapsed_sec = perf_counter() - score_field_start
    print(f"  [Position {position_index:03d}] score_field_elapsed_sec: {score_field_elapsed_sec:.3f}", flush=True)
    if not score_field:
        raise RuntimeError(f"Score field is empty for position {position_index:03d}")

    scores = [item.score for item in score_field]
    in_band_count = sum(1 for item in score_field if float(score_min) <= float(item.score) <= float(score_max))
    if in_band_count == 0:
        save_score_field_overlay(
            occupancy_map,
            score_field,
            score_overlay_path,
            person_position_xy=(
                float(debug_character["position"][0]),
                float(debug_character["position"][1]),
            ),
            selected_candidates=None,
        )
        print(f"  [Position {position_index:03d}] score_field_points: {len(score_field)}", flush=True)
        print(f"  [Position {position_index:03d}] score_range: [{min(scores):.3f}, {max(scores):.3f}]", flush=True)
        print(f"  [Position {position_index:03d}] in_band_candidates: 0", flush=True)
        print(
            f"  [Position {position_index:03d}] skipped: no candidates in score band "
            f"[{score_min:.3f}, {score_max:.3f}]",
            flush=True,
        )
        print(f"  [Position {position_index:03d}] score_overlay_path: {score_overlay_path}", flush=True)
        return pos_dir, 0

    capture_candidates = select_capture_candidates(
        score_field=score_field,
        score_min=score_min,
        score_max=score_max,
        seed=int(sampling_cfg.get("seed", 0)) + int(position_index),
        max_candidates=ring_cfg.get("max_capture_candidates"),
        fallback_to_nearest=False,
    )
    if not capture_candidates:
        raise RuntimeError(
            f"Failed to sample capture candidates in score band [{score_min:.3f}, {score_max:.3f}] "
            f"despite in_band_count={in_band_count} for position {position_index:03d}"
        )
    save_score_field_overlay(
        occupancy_map,
        score_field,
        score_overlay_path,
        person_position_xy=(
            float(debug_character["position"][0]),
            float(debug_character["position"][1]),
        ),
        selected_candidates=capture_candidates,
    )
    print(f"  [Position {position_index:03d}] score_field_points: {len(score_field)}", flush=True)
    print(f"  [Position {position_index:03d}] score_range: [{min(scores):.3f}, {max(scores):.3f}]", flush=True)
    print(f"  [Position {position_index:03d}] in_band_candidates: {in_band_count}", flush=True)
    print(f"  [Position {position_index:03d}] capture_candidates: {len(capture_candidates)}", flush=True)
    print(f"  [Position {position_index:03d}] score_overlay_path: {score_overlay_path}", flush=True)

    _set_render_mode(
        simulation_app=simulation_app,
        mode="PathTracing",
        warmup_updates=int(capture_camera_cfg_raw.get("warmup_updates", 8)),
    )
    print(f"  [Position {position_index:03d}] render_mode: PathTracing", flush=True)

    capture_records = capture_candidate_views(
        simulation_app=simulation_app,
        timeline=timeline,
        camera=collector_camera,
        candidates=capture_candidates,
        scene_dir=pos_dir,
        cfg=capture_camera_cfg,
        score_field=score_field,
        person_prim_path=str(debug_character["prim_path"]),
        occupancy_map=occupancy_map,
    )

    observation_items = []
    scores_payload = {}
    for idx, (candidate, record) in enumerate(zip(capture_candidates, capture_records)):
        matching_score = next(
            (
                item.score
                for item in score_field
                if abs(float(item.x) - float(candidate.x)) <= 1e-9
                and abs(float(item.y) - float(candidate.y)) <= 1e-9
                and abs(float(item.z) - float(candidate.z)) <= 1e-9
            ),
            0.0,
        )
        image_name = f"{idx:03d}.png"
        sampling_score = round(float(matching_score), 4)
        visibility_ratio = round(float(record.visibility_ratio), 4)
        observation_items.append(
            {
                "idx": int(idx),
                "rgb_path": _relpath(record.rgb_path, pos_dir),
                "depth_path": _relpath(record.depth_png_path, pos_dir),
                "depth_npy_path": _relpath(record.depth_npy_path, pos_dir),
                "ground_mask_path": _relpath(record.ground_mask_path, pos_dir),
                "score_map_path": _relpath(record.score_map_path, pos_dir),
                "valid_mask_path": _relpath(record.valid_mask_path, pos_dir),
                "yaw_map_path": _relpath(record.yaw_map_path, pos_dir),
                "reid_score": visibility_ratio,
                "sampling_score": sampling_score,
                "visible_person_pixels": int(record.visible_person_pixels),
                "total_person_pixels": int(record.total_person_pixels),
                "camera_position": [float(v) for v in record.camera_position],
                "camera_orientation_wxyz": [float(v) for v in record.camera_orientation_wxyz],
            }
        )
        scores_payload[image_name] = visibility_ratio

    metadata_path = pos_dir / "metadata.json"
    scores_path = pos_dir / "scores.json"
    metadata_payload = {
        "img_size": [int(capture_camera_cfg.resolution[1]), int(capture_camera_cfg.resolution[0])],
        "score_field_size": int(len(score_field)),
        "observations": observation_items,
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    scores_path.write_text(json.dumps(scores_payload, indent=2), encoding="utf-8")

    if capture_records:
        print(
            f"  [Position {position_index:03d}] first_capture:",
            json.dumps(
                {
                    "rgb_path": capture_records[0].rgb_path,
                    "depth_png_path": capture_records[0].depth_png_path,
                    "depth_npy_path": capture_records[0].depth_npy_path,
                    "ground_mask_path": capture_records[0].ground_mask_path,
                    "score_map_path": capture_records[0].score_map_path,
                    "valid_mask_path": capture_records[0].valid_mask_path,
                    "yaw_map_path": capture_records[0].yaw_map_path,
                    "metadata_path": str(metadata_path),
                    "scores_path": str(scores_path),
                    "visibility_ratio": capture_records[0].visibility_ratio,
                    "visible_person_pixels": capture_records[0].visible_person_pixels,
                    "total_person_pixels": capture_records[0].total_person_pixels,
                    "camera_position": capture_records[0].camera_position,
                    "camera_orientation_wxyz": capture_records[0].camera_orientation_wxyz,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    return pos_dir, len(capture_records)


def _merge_scene_cfg(base_scene_cfg: dict, scene_id: str, stage_root: str, interiorgs_root: str) -> dict:
    scene_cfg = dict(base_scene_cfg)
    scene_cfg["name"] = str(scene_id)
    scene_cfg["stage_url"] = str(Path(stage_root) / f"{scene_id}.usda")
    scene_cfg["interiorgs_root"] = str(interiorgs_root)
    return scene_cfg


def _collect_available_interiorgs_scene_ids(interiorgs_root: Path) -> set[str]:
    available_scene_ids: set[str] = set()
    for path in interiorgs_root.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if "_" not in name:
            continue
        scene_id = name.rsplit("_", 1)[-1].strip()
        if scene_id:
            available_scene_ids.add(scene_id)
    return available_scene_ids


def resolve_scene_configs(cfg: dict) -> list[dict]:
    scene_cfg = cfg.get("scene", {})
    dataset_cfg = cfg.get("dataset")
    if not dataset_cfg:
        return [scene_cfg]

    dataset_name = str(dataset_cfg.get("name", "dataset"))
    if dataset_name != "sage3d":
        raise ValueError(f"Unsupported dataset.name: {dataset_name}")

    stage_root = Path(dataset_cfg.get("stage_root", "/home/leo/FusionLab/DataSets/spatialverse/SAGE-3D_InteriorGS_usda"))
    interiorgs_root = Path(dataset_cfg.get("interiorgs_root", "/home/leo/FusionLab/DataSets/spatialverse/InteriorGS"))
    if not stage_root.exists():
        raise FileNotFoundError(f"Dataset stage_root does not exist: {stage_root}")
    if not interiorgs_root.exists():
        raise FileNotFoundError(f"Dataset interiorgs_root does not exist: {interiorgs_root}")

    selection_cfg = dataset_cfg.get("selection", {})
    mode = str(selection_cfg.get("mode", "single"))
    all_scene_ids = sorted(path.stem for path in stage_root.glob("*.usda"))
    if not all_scene_ids:
        raise RuntimeError(f"No .usda scenes found under {stage_root}")
    available_interiorgs_scene_ids = _collect_available_interiorgs_scene_ids(interiorgs_root)
    supported_scene_ids = sorted(scene_id for scene_id in all_scene_ids if scene_id in available_interiorgs_scene_ids)
    if not supported_scene_ids:
        raise RuntimeError(
            f"No scenes under {stage_root} have matching InteriorGS occupancy under {interiorgs_root}"
        )

    if mode == "single":
        explicit_ids = selection_cfg.get("scene_ids")
        if explicit_ids:
            scene_ids = [str(scene_id) for scene_id in explicit_ids]
        else:
            scene_id = selection_cfg.get("scene_id")
            if scene_id is None:
                raise ValueError("dataset.selection.mode=single requires scene_id or scene_ids")
            scene_ids = [str(scene_id)]
    elif mode == "random":
        random_count = int(selection_cfg.get("random_count", 1))
        random_seed = int(selection_cfg.get("random_seed", 0))
        if random_count <= 0:
            raise ValueError("dataset.selection.random_count must be > 0")
        if random_count > len(supported_scene_ids):
            raise ValueError(
                f"dataset.selection.random_count={random_count} exceeds supported scenes={len(supported_scene_ids)}"
            )
        rng = random.Random(random_seed)
        scene_ids = sorted(rng.sample(supported_scene_ids, random_count))
    elif mode == "all":
        scene_ids = supported_scene_ids
    else:
        raise ValueError(f"Unsupported dataset.selection.mode: {mode}")

    missing = [scene_id for scene_id in scene_ids if scene_id not in all_scene_ids]
    if missing:
        raise FileNotFoundError(f"Requested scene ids not found under {stage_root}: {missing}")
    if mode == "single":
        missing_occupancy = [scene_id for scene_id in scene_ids if scene_id not in available_interiorgs_scene_ids]
        if missing_occupancy:
            raise FileNotFoundError(
                f"Requested scene ids do not have matching InteriorGS occupancy under {interiorgs_root}: "
                f"{missing_occupancy}"
            )

    return [
        _merge_scene_cfg(scene_cfg, scene_id, str(stage_root), str(interiorgs_root))
        for scene_id in scene_ids
    ]


def collect_scene(simulation_app, cfg: dict, scene_cfg: dict, keep_scene_open: bool) -> None:
    scene_name = scene_cfg.get("name", "unnamed_scene")
    output_root = cfg.get("output", {}).get("root_dir", "./outputs")
    scene_root = ensure_output_dirs(output_root, scene_name)

    print(f"[Step 1] output directory ready: {scene_root}", flush=True)
    print(f"[Step 2] loading scene: {scene_cfg.get('stage_url')}", flush=True)
    result = load_scene(scene_cfg, simulation_app)

    print("[Step 2] scene loaded successfully", flush=True)
    print(f"  stage_url: {result.stage_url}", flush=True)
    print(f"  has_physics_scene: {result.has_physics_scene}", flush=True)
    print(f"  physics_scene_path: {result.physics_scene_path}", flush=True)
    print(f"  stage_prim_count: {result.stage_prim_count}", flush=True)

    import omni.timeline
    import omni.usd

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(int(scene_cfg.get("warmup_updates", 10))):
        simulation_app.update()
    timeline.pause()
    print("[Step 2] scene warmup finished", flush=True)

    print("[Step 3] loading InteriorGS occupancy", flush=True)
    occupancy_map = load_interiorgs_occupancy_map(scene_cfg)
    occupancy_summary = summarize_occupancy_map(occupancy_map)
    print("[Step 3] occupancy loaded", flush=True)
    print(f"  occupancy_meta_path: {occupancy_summary.yaml_path}", flush=True)
    print(f"  occupancy_image_path: {occupancy_summary.image_path}", flush=True)
    print(f"  occupancy_resolution: {occupancy_summary.resolution}", flush=True)
    print(f"  occupancy_origin_xy: {occupancy_summary.origin_xy}", flush=True)
    print(
        f"  occupancy_transform: flip_x={occupancy_map.flip_x}, "
        f"flip_y={occupancy_map.flip_y}, negate_xy={occupancy_map.negate_xy}",
        flush=True,
    )
    print(f"  occupancy_size: {occupancy_summary.width} x {occupancy_summary.height}", flush=True)
    print(f"  occupancy_free_cells: {occupancy_summary.free_cells}", flush=True)
    print(f"  occupancy_occupied_cells: {occupancy_summary.occupied_cells}", flush=True)
    print(f"  occupancy_unknown_cells: {occupancy_summary.unknown_cells}", flush=True)
    print(f"  occupancy_sampled_world_points: {occupancy_summary.sampled_world_points}", flush=True)

    stage = omni.usd.get_context().get_stage()
    stage.Load()

    physics_scene_path, physics_scene_created = ensure_scene_query_support(stage)
    print(f"  debug_physics_scene_path: {physics_scene_path}", flush=True)
    print(f"  debug_physics_scene_created: {physics_scene_created}", flush=True)

    timeline.play()
    for _ in range(3):
        simulation_app.update()
    timeline.pause()

    num_positions = int(cfg.get("num_positions", 1))
    if num_positions <= 0:
        raise ValueError("num_positions must be > 0")
    print(f"[Step 3] collecting {num_positions} person positions in scene", flush=True)
    total_captures = 0
    for position_index in range(num_positions):
        print(f"[Position {position_index:03d}] start", flush=True)
        pos_dir, capture_count = _collect_single_position(
            simulation_app=simulation_app,
            timeline=timeline,
            stage=stage,
            occupancy_map=occupancy_map,
            cfg=cfg,
            scene_cfg=scene_cfg,
            scene_root=scene_root,
            position_index=position_index,
        )
        total_captures += capture_count
        print(f"[Position {position_index:03d}] saved to {pos_dir} ({capture_count} views)", flush=True)

    print(f"[Step 4] scene capture complete: {num_positions} positions, {total_captures} views total", flush=True)

    if keep_scene_open:
        timeline.play()
        print("[Step 5] capture finished; keeping scene open for inspection. Close the Isaac Sim window to exit.", flush=True)
        while simulation_app.is_running():
            simulation_app.update()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    scene_cfgs = resolve_scene_configs(cfg)
    launch_cfg = {"headless": bool(cfg.get("headless", False))}

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(launch_config=launch_cfg)
    try:
        print(f"[Config] resolved scenes: {len(scene_cfgs)}", flush=True)
        for index, scene_cfg in enumerate(scene_cfgs, start=1):
            print(
                f"[Config] scene {index}/{len(scene_cfgs)}: "
                f"{scene_cfg.get('name', 'unnamed_scene')} ({scene_cfg.get('stage_url')})",
                flush=True,
            )

        multiple_scenes = len(scene_cfgs) > 1
        for index, scene_cfg in enumerate(scene_cfgs, start=1):
            print(f"\n=== Scene {index}/{len(scene_cfgs)}: {scene_cfg.get('name', 'unnamed_scene')} ===", flush=True)
            keep_scene_open = bool(scene_cfg.get("keep_scene_open_after_capture", True)) and not multiple_scenes
            collect_scene(simulation_app, cfg, scene_cfg, keep_scene_open=keep_scene_open)
            if multiple_scenes:
                try:
                    import omni.usd

                    ctx = omni.usd.get_context()
                    if ctx is not None:
                        ctx.close_stage()
                except Exception:
                    pass
                gc.collect()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        try:
            import omni.usd

            ctx = omni.usd.get_context()
            if ctx is not None:
                ctx.close_stage()
        except Exception:
            pass
        gc.collect()
        simulation_app.close()


if __name__ == "__main__":
    main()
