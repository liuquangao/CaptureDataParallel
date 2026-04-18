from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
from PIL import Image
import yaml
from isaacsim import SimulationApp

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from collector.debug_viz import place_debug_character
from collector.occupancy import load_interiorgs_occupancy_map
from collector.score_field import generate_segmentation_score_field, select_capture_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Replicator RGB capture from occupancy-derived camera candidates")
    parser.add_argument("--config", type=str, default=str(THIS_DIR / "configs" / "sage3d.yaml"))
    parser.add_argument("--scene-id", type=str, default="839875")
    parser.add_argument("--camera-count", type=int, default=32)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/leo/FusionLab/CaptureData/outputs/replicator_debug_839875_with_person",
    )
    return parser.parse_args()


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_scene_cfg(cfg: dict, scene_id: str) -> tuple[dict, dict, dict, dict]:
    dataset_cfg = dict(cfg.get("dataset", {}))
    scene_cfg = dict(cfg.get("scene", {}))
    scoring_camera_cfg = dict(cfg.get("scoring_camera", {}))
    capture_camera_cfg = dict(cfg.get("capture_camera", cfg.get("camera", {})))

    stage_root = Path(dataset_cfg.get("stage_root", "/home/leo/FusionLab/DataSets/spatialverse/SAGE-3D_InteriorGS_usda"))
    interiorgs_root = Path(dataset_cfg.get("interiorgs_root", "/home/leo/FusionLab/DataSets/spatialverse/InteriorGS"))
    scene_cfg["name"] = str(scene_id)
    scene_cfg["stage_url"] = str(stage_root / f"{scene_id}.usda")
    scene_cfg["interiorgs_root"] = str(interiorgs_root)
    return scene_cfg, dataset_cfg, scoring_camera_cfg, capture_camera_cfg


def _rgb_data_to_numpy(rgb_data) -> np.ndarray:
    if isinstance(rgb_data, np.ndarray):
        arr = rgb_data
    else:
        arr = None
        if hasattr(rgb_data, "detach") and hasattr(rgb_data, "cpu"):
            arr = rgb_data.detach().cpu().numpy()
        elif hasattr(rgb_data, "cpu") and hasattr(rgb_data, "numpy"):
            arr = rgb_data.cpu().numpy()
        elif hasattr(rgb_data, "__dlpack__"):
            import torch

            arr = torch.utils.dlpack.from_dlpack(rgb_data).cpu().numpy()
        elif hasattr(rgb_data, "numpy"):
            arr = rgb_data.numpy()
        else:
            arr = np.asarray(rgb_data)

    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    scene_cfg, _, scoring_camera_cfg, capture_camera_cfg = _merge_scene_cfg(cfg, args.scene_id)
    stage_path = Path(scene_cfg["stage_url"])
    if not stage_path.exists():
        raise FileNotFoundError(f"Stage file does not exist: {stage_path}")

    simulation_app = SimulationApp(
        launch_config={
            "renderer": "RealTimePathTracing",
            "headless": bool(args.headless),
        }
    )

    try:
        import carb.settings
        import omni.replicator.core as rep
        from isaacsim.core.utils.stage import get_current_stage, open_stage

        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
        rep.orchestrator.set_capture_on_play(False)

        if not open_stage(str(stage_path)):
            raise RuntimeError(f"Failed to open stage: {stage_path}")
        for _ in range(int(scene_cfg.get("warmup_updates", 10))):
            simulation_app.update()

        stage = get_current_stage()
        print("加载完成", flush=True)

        occupancy_map = load_interiorgs_occupancy_map(scene_cfg)
        debug_character = place_debug_character(
            stage=stage,
            occupancy_map=occupancy_map,
            seed=int(cfg.get("sampling", {}).get("seed", 2026)),
            character_usd_path=str(
                scene_cfg.get(
                    "character_usd_path",
                    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/"
                    "Assets/Isaac/6.0/Isaac/People/Characters/F_Business_02/F_Business_02.usd",
                )
            ),
            arm_drop_degrees=float(scene_cfg.get("character_arm_drop_degrees", 75.0)),
        )
        person_position = tuple(float(v) for v in debug_character["position"])
        print(f"人物位置: {person_position}", flush=True)

        ring_cfg = cfg.get("score_field", {})
        scoring_results_default = [(0.0, 0, 0)]

        def _dummy_visibility_batch_scorer(samples):
            return scoring_results_default * len(samples)

        score_field = generate_segmentation_score_field(
            occupancy_map=occupancy_map,
            person_position_xy=(person_position[0], person_position[1]),
            camera_height_m=float(scoring_camera_cfg.get("camera_height", 0.4)),
            min_radius_m=float(ring_cfg.get("min_radius_m", 1.2)),
            max_radius_m=float(ring_cfg.get("max_radius_m", 4.5)),
            grid_step_m=float(ring_cfg.get("grid_step_m", 0.2)),
            visibility_batch_scorer=_dummy_visibility_batch_scorer,
            occupancy_full_visibility_width_m=float(ring_cfg.get("occupancy_full_visibility_width_m", 0.25)),
        )
        if not score_field:
            raise RuntimeError("Score field is empty")

        capture_candidates = select_capture_candidates(
            score_field=score_field,
            score_min=float(ring_cfg.get("capture_score_min", 0.1)),
            score_max=1.0,
            seed=int(cfg.get("sampling", {}).get("seed", 2026)),
            max_candidates=int(args.camera_count),
            fallback_to_nearest=True,
        )
        if len(capture_candidates) < int(args.camera_count):
            raise RuntimeError(
                f"Only found {len(capture_candidates)} capture candidates, expected {args.camera_count}"
            )

        stage.DefinePrim("/DebugCandidateCameras", "Scope")
        cameras = []
        render_products = []
        rgb_annotators = []
        resolution = (
            int(capture_camera_cfg.get("resolution", [600, 400])[0]),
            int(capture_camera_cfg.get("resolution", [600, 400])[1]),
        )

        for index in range(int(args.camera_count)):
            camera = rep.functional.create.camera(
                focus_distance=float(capture_camera_cfg.get("focus_distance", 400.0)),
                focal_length=18.0,
                clipping_range=(float(capture_camera_cfg.get("near_clipping_distance", 0.01)), 10000000.0),
                name=f"CandidateCam_{index:03d}",
                parent="/DebugCandidateCameras",
            )
            cameras.append(camera)

        print("初始化完成", flush=True)

        look_at_target = (
            float(person_position[0]),
            float(person_position[1]),
            float(person_position[2]) + 0.9,
        )
        camera_height = float(capture_camera_cfg.get("camera_height", 0.4))
        for index, (camera, candidate) in enumerate(zip(cameras, capture_candidates)):
            position = (float(candidate.x), float(candidate.y), camera_height)
            rep.functional.modify.pose(
                camera,
                position_value=position,
                look_at_value=look_at_target,
                look_at_up_axis=(0.0, 0.0, 1.0),
            )
            render_product = rep.create.render_product(camera, resolution, name=f"RGB_{index:03d}")
            render_product.hydra_texture.set_updates_enabled(False)
            render_products.append(render_product)
            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cuda", do_array_copy=False)
            rgb_annotator.attach([render_product.path])
            rgb_annotators.append(rgb_annotator)

        print("相机位置设定完成", flush=True)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for render_product in render_products:
            render_product.hydra_texture.set_updates_enabled(True)

        print("开始采集图像", flush=True)
        capture_start = perf_counter()
        rep.orchestrator.step(delta_time=0.0, rt_subframes=32)
        rep.orchestrator.wait_until_complete()
        capture_elapsed = perf_counter() - capture_start
        for index, rgb_annotator in enumerate(rgb_annotators):
            rgb = _rgb_data_to_numpy(rgb_annotator.get_data())
            Image.fromarray(rgb, mode="RGB").save(output_dir / f"{index:03d}.png")
        print(f"结束采集图像，耗时 {capture_elapsed:.3f} 秒", flush=True)

        for rgb_annotator in rgb_annotators:
            rgb_annotator.detach()
        for render_product in render_products:
            render_product.destroy()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
