#!/usr/bin/env python3
"""
Capture 8 orbital reference views for each character for ReID scoring.

For each character, places it on an empty stage, orbits a camera around it at
8 equally-spaced angles, and saves tight person crops.

Person detection: depth-based (empty stage → only person has finite depth).

Output:
    <output_dir>/F_Business_02/ref_00.png ... ref_07.png
    <output_dir>/male_adult_medical_01/ref_00.png ... ref_07.png

Usage:
    /home/leo/FusionLab/isaacsim/_build/linux-x86_64/release/python.sh \\
        script/capture_reference_views.py \\
        [--output_dir outputs_parallel-v2-raycasted/reference] \\
        [--warmup_steps 30]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHARACTERS = {
    "F_Business_02": (
        "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/6.0/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
    ),
    "male_adult_medical_01": (
        "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/6.0/Isaac/People/Characters/original_male_adult_medical_01/"
        "male_adult_medical_01.usd"
    ),
}

_NUM_VIEWS = 8
_ORBIT_RADIUS_M = 2.0
_CAMERA_HEIGHT_M = 1.0
_LOOK_AT_HEIGHT = 0.85      # approx person torso center (m)
_RENDER_W, _RENDER_H = 512, 512
_ARM_DROP_DEGREES = 75.0
_CROP_PADDING = 0.08
_DEPTH_MAX_M = 8.0          # pixels beyond this are background


def _load_module(name: str, rel_path: str):
    """Load a project module by absolute file path, bypassing sys.path."""
    path = _PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _depth_bbox_xyxy(depth: "np.ndarray", max_depth_m: float = _DEPTH_MAX_M):
    """Return (u_min, v_min, u_max, v_max) for pixels with 0 < depth < max_depth_m."""
    import numpy as np
    mask = np.isfinite(depth) & (depth > 0.01) & (depth < max_depth_m)
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture 8 reference views per character")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_parallel-v2-raycasted/reference",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=30,
        help="Simulation update steps before capturing each frame",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({
        "renderer": "RayTracedLighting",
        "headless": True,
        "anti_aliasing": 1,
    })

    import carb.settings
    carb.settings.get_settings().set("/log/level", "error")
    carb.settings.get_settings().set("/log/outputStreamLevel", "Error")

    import numpy as np
    import omni.timeline
    import omni.usd
    import omni.replicator.core as rep
    from isaacsim.core.utils.stage import get_current_stage
    from PIL import Image
    from pxr import UsdLux

    timeline = omni.timeline.get_timeline_interface()

    _pp = _load_module("_person_placement", "utils/person_placement.py")
    _add_usd_reference_xform = _pp._add_usd_reference_xform
    _apply_static_standing_pose = _pp._apply_static_standing_pose

    for char_name, char_url in CHARACTERS.items():
        print(f"\n=== {char_name} ===")
        char_out = output_dir / char_name
        char_out.mkdir(parents=True, exist_ok=True)

        # ---- empty stage ----
        omni.usd.get_context().new_stage()
        stage = get_current_stage()

        dome = stage.DefinePrim("/World/DomeLight", "DomeLight")
        UsdLux.DomeLight(dome).GetIntensityAttr().Set(1000.0)

        PERSON_PRIM = "/World/Person"
        _add_usd_reference_xform(stage, char_url, PERSON_PRIM, (0.0, 0.0, 0.0), yaw_rad=0.0)
        _apply_static_standing_pose(stage, PERSON_PRIM, arm_drop_degrees=_ARM_DROP_DEGREES)

        # Timeline must be playing for simulation_app.update() to trigger rendering
        if not timeline.is_playing():
            timeline.play()

        # Warmup: allow remote USD to download and load
        print(f"  Loading character ({args.warmup_steps} warmup steps)...")
        for _ in range(args.warmup_steps):
            simulation_app.update()

        # ---- camera + annotators ----
        cam = rep.functional.create.camera(
            focal_length=8.0,
            focus_distance=_ORBIT_RADIUS_M,
            clipping_range=(0.01, 1e7),
            name="RefCam",
            parent="/World",
        )
        rp = rep.create.render_product(cam, (_RENDER_W, _RENDER_H))
        rp.hydra_texture.set_updates_enabled(False)  # disabled until we're ready

        rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_ann.attach(rp)

        depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth_ann.attach(rp)

        # Enable rendering and warm up the render product before first capture
        rp.hydra_texture.set_updates_enabled(True)
        print(f"  Warming up render product ({args.warmup_steps} steps)...")
        for _ in range(args.warmup_steps):
            simulation_app.update()

        # ---- orbit 8 angles ----
        for view_idx in range(_NUM_VIEWS):
            angle_rad = (2.0 * math.pi * view_idx) / _NUM_VIEWS
            cx = _ORBIT_RADIUS_M * math.cos(angle_rad)
            cy = _ORBIT_RADIUS_M * math.sin(angle_rad)

            rep.functional.modify.pose(
                cam,
                position_value=(cx, cy, _CAMERA_HEIGHT_M),
                look_at_value=(0.0, 0.0, _LOOK_AT_HEIGHT),
                look_at_up_axis=(0, 0, 1),
            )

            for _ in range(10):
                simulation_app.update()

            rgb_data = np.asarray(rgb_ann.get_data())
            if rgb_data.size == 0:
                print(f"  WARNING: render returned empty data at view {view_idx}, skipping")
                continue
            rgb = rgb_data[:, :, :3] if rgb_data.shape[-1] == 4 else rgb_data
            depth = np.asarray(depth_ann.get_data(), dtype=np.float32)

            bbox = _depth_bbox_xyxy(depth)
            if bbox is None:
                finite = int(np.isfinite(depth).sum()) if depth.size > 0 else 0
                dmin = float(depth[np.isfinite(depth)].min()) if finite > 0 else float("nan")
                dmax = float(depth[np.isfinite(depth)].max()) if finite > 0 else float("nan")
                print(f"  WARNING: no person pixels at view {view_idx} "
                      f"(angle={math.degrees(angle_rad):.0f}°)")
                print(f"    depth shape={depth.shape} finite={finite} "
                      f"min={dmin:.2f} max={dmax:.2f}")
                continue

            u_min, v_min, u_max, v_max = bbox
            pad_v = max(1, int((v_max - v_min) * _CROP_PADDING))
            pad_u = max(1, int((u_max - u_min) * _CROP_PADDING))
            v_min = max(0, v_min - pad_v)
            v_max = min(_RENDER_H - 1, v_max + pad_v)
            u_min = max(0, u_min - pad_u)
            u_max = min(_RENDER_W - 1, u_max + pad_u)

            crop = rgb[v_min:v_max + 1, u_min:u_max + 1]
            out_path = char_out / f"ref_{view_idx:02d}.png"
            Image.fromarray(crop.astype(np.uint8)).save(out_path)
            print(f"  ref_{view_idx:02d}.png  {crop.shape[1]}×{crop.shape[0]}px  "
                  f"angle={math.degrees(angle_rad):.0f}°")

        # ---- cleanup ----
        rgb_ann.detach()
        depth_ann.detach()
        rp.destroy()

    simulation_app.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
