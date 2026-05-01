# Copyright (c) 2026, Fusion Intelligence Labs, University of Exeter. All rights reserved.
#
# Open Isaac Sim, load scene 839874, place one static standing person, and keep
# the interactive window alive.

from __future__ import annotations

import argparse
import importlib.util
import resource
import sys
from pathlib import Path


_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


DEFAULT_STAGE_ROOT = Path("/home/leo/FusionLab/DataSets/spatialverse/SAGE-3D_InteriorGS_usda")
DEFAULT_INTERIORGS_ROOT = Path("/home/leo/FusionLab/DataSets/spatialverse/InteriorGS")
DEFAULT_CHARACTER_USD = (
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open scene 839874 in Isaac Sim and place one person.")
    parser.add_argument("--scene-id", default="839874", help="SAGE-3D scene id to open.")
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--interiorgs-root", type=Path, default=DEFAULT_INTERIORGS_ROOT)
    parser.add_argument("--person-url", default=DEFAULT_CHARACTER_USD)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--arm-drop-degrees", type=float, default=75.0)
    parser.add_argument("--min-obstacle-distance-m", type=float, default=0.2)
    parser.add_argument(
        "--show-gauss",
        action="store_true",
        help="Show /World/gauss. This is the default; the flag is kept for compatibility.",
    )
    parser.add_argument(
        "--hide-gauss",
        action="store_true",
        help="Hide /World/gauss if you only want to inspect the collision mesh.",
    )
    parser.add_argument(
        "--hide-scene-collision",
        action="store_true",
        help="Hide /World/scene_collision recursively.",
    )
    return parser.parse_args()


def _load_project_module(module_name: str, relative_path: str):
    """Load a local project module without using the top-level ``utils`` name."""
    module_path = THIS_DIR / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _set_preview_camera(stage, person_xyz: list[float]) -> str | None:
    import omni.replicator.core as rep
    from pxr import UsdGeom

    person_x = float(person_xyz[0])
    person_y = float(person_xyz[1])
    person_z = float(person_xyz[2])
    camera_z = person_z + 1.55
    camera = rep.functional.create.camera(
        focal_length=18.0,
        clipping_range=(0.01, 100000.0),
        name="PreviewCamera",
        parent="/SDG",
    )
    rep.functional.modify.pose(
        camera,
        position_value=(person_x - 3.0, person_y - 3.0, camera_z),
        look_at_value=(person_x, person_y, person_z + 0.95),
        look_at_up_axis=(0, 0, 1),
    )

    camera_prim = UsdGeom.Camera(camera).GetPrim()
    camera_path = str(camera_prim.GetPath())
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera_path
    except Exception as exc:  # noqa: BLE001
        print(f"[OPEN-839874] Could not switch active viewport camera: {exc}")

    return camera_path if stage.GetPrimAtPath(camera_path).IsValid() else None


def _add_label_if_valid(stage, add_labels, prim_path: str, label: str) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"[OPEN-839874] Semantic label skipped; prim not found: {prim_path}")
        return False
    add_labels(prim_path, labels=label, taxonomy="class")
    print(f"[OPEN-839874] Semantic label: {prim_path} -> class={label}")
    return True


def main() -> None:
    args = parse_args()
    stage_url = args.stage_root / f"{args.scene_id}.usda"
    scene_cfg = {
        "name": str(args.scene_id),
        "stage_url": str(stage_url),
        "interiorgs_root": str(args.interiorgs_root),
        "occupancy_flip_x": True,
        "occupancy_flip_y": True,
        "occupancy_negate_xy": True,
    }

    if not stage_url.exists():
        raise FileNotFoundError(f"Scene USD does not exist: {stage_url}")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        launch_config={
            "renderer": "RealTimePathTracing",
            "headless": False,
            "disable_viewport_updates": False,
            "anti_aliasing": 1,
        }
    )

    import carb.settings
    from isaacsim.core.experimental.utils.semantics import add_labels, remove_all_labels
    from isaacsim.core.utils.stage import get_current_stage, open_stage

    occupancy_module = _load_project_module("capture_data_parallel_occupancy_map", "utils/occupancy_map.py")
    person_module = _load_project_module("capture_data_parallel_person_placement", "utils/person_placement.py")
    replicator_tools_module = _load_project_module(
        "capture_data_parallel_replicator_tools",
        "utils/replicator_tools.py",
    )
    load_interiorgs_occupancy_map = occupancy_module.load_interiorgs_occupancy_map
    place_person = person_module.place_person
    set_prim_visibility = replicator_tools_module.set_prim_visibility

    carb_settings = carb.settings.get_settings()
    carb_settings.set("/log/level", "error")
    carb_settings.set("/log/outputStreamLevel", "Error")
    carb_settings.set("/rtx/rendermode", "RayTracedLighting")
    carb_settings.set("/rtx/post/aa/op", 1)
    carb_settings.set("/rtx/post/dlss/execMode", 0)

    print(f"[OPEN-839874] Loading occupancy map for scene {args.scene_id}")
    occupancy_map = load_interiorgs_occupancy_map(scene_cfg)

    print(f"[OPEN-839874] Opening stage: {stage_url}")
    if not open_stage(str(stage_url)):
        simulation_app.close()
        raise RuntimeError(f"open_stage failed: {stage_url}")

    stage = get_current_stage()
    for prim in stage.Traverse():
        remove_all_labels(prim, include_descendants=True)
    stage.DefinePrim("/SDG", "Scope")

    set_prim_visibility(stage, "/World/gauss", visible=not bool(args.hide_gauss))
    set_prim_visibility(stage, "/World/scene_collision", visible=not bool(args.hide_scene_collision))
    _add_label_if_valid(stage, add_labels, "/World/gauss", "scene")
    _add_label_if_valid(stage, add_labels, "/World/scene_collision", "scene")

    person = place_person(
        stage=stage,
        occupancy_map=occupancy_map,
        prim_path="/SDG/Person",
        seed=int(args.seed),
        character_usd_path=str(args.person_url),
        arm_drop_degrees=float(args.arm_drop_degrees),
        min_obstacle_distance_m=float(args.min_obstacle_distance_m),
    )
    _add_label_if_valid(stage, add_labels, person["prim_path"], "person")
    camera_path = _set_preview_camera(stage, person["position"])

    print(f"[OPEN-839874] Person prim: {person['prim_path']}")
    print(f"[OPEN-839874] Person position: {person['position']}")
    print(f"[OPEN-839874] Pose info: {person['pose_info']}")
    if camera_path is not None:
        print(f"[OPEN-839874] Viewport camera: {camera_path}")
    print("[OPEN-839874] Window will stay open. Press Ctrl+C in this terminal to close.")

    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        print("[OPEN-839874] Closing Isaac Sim.")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
