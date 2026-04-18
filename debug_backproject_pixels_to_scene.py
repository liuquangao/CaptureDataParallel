#!/home/leo/FusionLab/isaacsim/_build/linux-x86_64/release/python.sh

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import yaml

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from collector.occupancy import load_interiorgs_occupancy_map, save_occupancy_overlay
from collector.scene_loader import load_scene
from debug_render_occupancy_in_scene import add_occupancy_overlay


@dataclass
class DebugPoint:
    index: int
    variant: str
    pixel_u: int
    pixel_v: int
    depth_m: float
    world_x: float
    world_y: float
    world_z: float
    occ_row: int
    occ_col: int
    occ_state: str

    @property
    def x(self) -> float:
        return self.world_x

    @property
    def y(self) -> float:
        return self.world_y

    @property
    def z(self) -> float:
        return self.world_z


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backproject selected RGB pixels into Isaac world space and verify them against occupancy."
    )
    parser.add_argument(
        "--rgb",
        type=Path,
        default=Path("/home/leo/FusionLab/CaptureData/outputs/839875/pos_000/rgb/015.png"),
        help="RGB image path.",
    )
    parser.add_argument(
        "--depth-npy",
        type=Path,
        default=None,
        help="Depth .npy path. If omitted, resolve from metadata.json.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="metadata.json path. If omitted, infer from the RGB parent directory.",
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path("/home/leo/FusionLab/DataSets/spatialverse/SAGE-3D_InteriorGS_usda"),
        help="Directory containing SAGE-3D .usda stages.",
    )
    parser.add_argument(
        "--interiorgs-root",
        type=Path,
        default=Path("/home/leo/FusionLab/DataSets/spatialverse/InteriorGS"),
        help="InteriorGS root for occupancy lookup.",
    )
    parser.add_argument(
        "--pixels",
        nargs="*",
        default=["120,330", "280,320", "470,310", "300,260"],
        help="Pixel coordinates as 'u,v'. Defaults target floor-like pixels in 015.png.",
    )
    parser.add_argument(
        "--prim-path",
        type=str,
        default="/World/DebugBackprojectedPixels",
        help="Root prim path for backprojected world markers.",
    )
    parser.add_argument(
        "--occupancy-prim-path",
        type=str,
        default="/World/DebugOccupancy",
        help="Root prim path for the in-scene occupancy overlay.",
    )
    parser.add_argument(
        "--occupancy-z",
        type=float,
        default=0.02,
        help="Z value used for rendering occupancy cells in scene.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Launch Isaac Sim headless.",
    )
    return parser.parse_args()


def _parse_pixel_specs(specs: list[str]) -> list[tuple[int, int]]:
    pixels: list[tuple[int, int]] = []
    for raw in specs:
        u_str, v_str = raw.split(",", 1)
        pixels.append((int(u_str), int(v_str)))
    return pixels


def _load_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json does not exist: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_observation(metadata: dict, rgb_relpath: str) -> dict:
    for item in metadata.get("observations", []):
        if item.get("rgb_path") == rgb_relpath:
            return item
    raise KeyError(f"Could not find observation for rgb_path={rgb_relpath}")


def _resolve_scene_id_from_rgb(rgb_path: Path) -> str:
    pos_dir = rgb_path.parent.parent
    scene_dir = pos_dir.parent
    return scene_dir.name


def _camera_intrinsics(width: int, height: int, focal_length: float, horizontal_aperture: float, vertical_aperture: float) -> tuple[float, float, float, float]:
    fx = width * float(focal_length) / max(float(horizontal_aperture), 1e-6)
    fy = height * float(focal_length) / max(float(vertical_aperture), 1e-6)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return fx, fy, cx, cy


def _backproject_pixel_to_world(
    u: int,
    v: int,
    depth_m: float,
    camera_position: tuple[float, float, float],
    camera_orientation_wxyz: tuple[float, float, float, float],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    invert_lateral: bool = False,
) -> tuple[float, float, float]:
    forward = float(depth_m)
    lateral = (float(u) - cx) * forward / max(fx, 1e-6)
    if invert_lateral:
        lateral = -lateral
    vertical = -(float(v) - cy) * forward / max(fy, 1e-6)

    yaw_rad = 2.0 * np.arctan2(float(camera_orientation_wxyz[3]), float(camera_orientation_wxyz[0]))
    cos_yaw = float(np.cos(float(yaw_rad)))
    sin_yaw = float(np.sin(float(yaw_rad)))

    world_x = float(camera_position[0]) + forward * cos_yaw - lateral * sin_yaw
    world_y = float(camera_position[1]) + forward * sin_yaw + lateral * cos_yaw
    world_z = float(camera_position[2]) + vertical
    return world_x, world_y, world_z


def _occupancy_state(occupancy_map, row: int, col: int) -> str:
    if not (0 <= row < occupancy_map.height and 0 <= col < occupancy_map.width):
        return "out_of_bounds"
    if bool(occupancy_map.free_mask[row, col]):
        return "free"
    if bool(occupancy_map.occupied_mask[row, col]):
        return "occupied"
    if bool(occupancy_map.unknown_mask[row, col]):
        return "unknown"
    return "other"


def _save_rgb_overlay(rgb_path: Path, debug_points: list[DebugPoint], out_path: Path) -> None:
    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    variant_colors = {
        "base": (255, 64, 64),
        "flip_lateral": (0, 220, 0),
    }
    radius = 6
    for point in debug_points:
        color = variant_colors.get(point.variant, (255, 255, 255))
        u, v = point.pixel_u, point.pixel_v
        offset = -3 if point.variant == "base" else 3
        draw.ellipse((u - radius, v - radius + offset, u + radius, v + radius + offset), outline=color, width=2)
        draw.text((u + radius + 2, v - radius + offset), f"{point.index}:{point.variant}", fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _add_world_markers(stage, prim_path: str, debug_points: list[DebugPoint]) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(prim_path)
    if root and root.IsValid():
        stage.RemovePrim(prim_path)
    UsdGeom.Xform.Define(stage, prim_path)

    variant_to_rgb = {
        "base": (0.95, 0.15, 0.15),
        "flip_lateral": (0.10, 0.90, 0.20),
    }
    for point in debug_points:
        point_path = f"{prim_path}/P_{point.index:02d}_{point.variant}"
        sphere = UsdGeom.Sphere.Define(stage, point_path)
        sphere.CreateRadiusAttr(0.06)
        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(point.world_x), float(point.world_y), float(point.world_z)))

        material_path = f"{point_path}/Material"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(variant_to_rgb.get(point.variant, (1.0, 1.0, 1.0)))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(material)


def main() -> None:
    args = parse_args()

    rgb_path = args.rgb.resolve()
    metadata_path = args.metadata.resolve() if args.metadata is not None else rgb_path.parent.parent / "metadata.json"
    metadata = _load_metadata(metadata_path)
    rgb_relpath = str(rgb_path.resolve().relative_to(rgb_path.parent.parent.resolve()))
    observation = _find_observation(metadata, rgb_relpath)

    depth_npy_path = args.depth_npy.resolve() if args.depth_npy is not None else (rgb_path.parent.parent / observation["depth_npy_path"]).resolve()
    if not depth_npy_path.exists():
        raise FileNotFoundError(f"Depth npy does not exist: {depth_npy_path}")

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.asarray(np.load(depth_npy_path), dtype=np.float32)
    height, width = rgb.shape[:2]
    if depth.shape != (height, width):
        raise ValueError(f"Depth shape {depth.shape} does not match RGB shape {(height, width)}")

    camera_position = tuple(float(v) for v in observation["camera_position"])
    camera_orientation = tuple(float(v) for v in observation["camera_orientation_wxyz"])
    pixels = _parse_pixel_specs(args.pixels)

    scene_id = _resolve_scene_id_from_rgb(rgb_path)
    scene_cfg = {
        "stage_url": str((args.stage_root / f"{scene_id}.usda").resolve()),
        "interiorgs_root": str(args.interiorgs_root.resolve()),
        "occupancy_flip_x": True,
        "occupancy_flip_y": True,
        "occupancy_negate_xy": True,
    }
    occupancy_map = load_interiorgs_occupancy_map(scene_cfg)

    camera_cfg_path = THIS_DIR / "configs" / "sage3d.yaml"
    with camera_cfg_path.open("r", encoding="utf-8") as f:
        camera_cfg = yaml.safe_load(f)["camera"]

    fx, fy, cx, cy = _camera_intrinsics(
        width=width,
        height=height,
        focal_length=float(camera_cfg["focal_length"]),
        horizontal_aperture=float(camera_cfg["horizontal_aperture"]),
        vertical_aperture=float(camera_cfg["vertical_aperture"]),
    )

    debug_points: list[DebugPoint] = []
    for index, (u, v) in enumerate(pixels):
        if not (0 <= u < width and 0 <= v < height):
            raise ValueError(f"Pixel {(u, v)} is out of bounds for image size {(width, height)}")
        depth_m = float(depth[v, u])
        for variant, invert_lateral in (("base", False), ("flip_lateral", True)):
            world_x, world_y, world_z = _backproject_pixel_to_world(
                u=u,
                v=v,
                depth_m=depth_m,
                camera_position=camera_position,
                camera_orientation_wxyz=camera_orientation,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                invert_lateral=invert_lateral,
            )
            occ_row, occ_col = occupancy_map.world_to_grid(world_x, world_y)
            debug_points.append(
                DebugPoint(
                    index=index,
                    variant=variant,
                    pixel_u=u,
                    pixel_v=v,
                    depth_m=depth_m,
                    world_x=world_x,
                    world_y=world_y,
                    world_z=world_z,
                    occ_row=int(occ_row),
                    occ_col=int(occ_col),
                    occ_state=_occupancy_state(occupancy_map, int(occ_row), int(occ_col)),
                )
            )

    out_dir = rgb_path.parent.parent / "backproject_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_overlay_path = out_dir / f"{rgb_path.stem}_rgb_points.png"
    occ_overlay_path = out_dir / f"{rgb_path.stem}_occupancy_points.png"
    _save_rgb_overlay(rgb_path, debug_points, rgb_overlay_path)
    save_occupancy_overlay(
        occupancy_map=occupancy_map,
        candidates=debug_points,
        out_path=occ_overlay_path,
        person_position_xy=None,
    )

    print(f"[Backproject] rgb={rgb_path}")
    print(f"[Backproject] depth_npy={depth_npy_path}")
    print(f"[Backproject] metadata={metadata_path}")
    print(f"[Backproject] rgb_overlay={rgb_overlay_path}")
    print(f"[Backproject] occupancy_overlay={occ_overlay_path}")
    for point in debug_points:
        print(
            "[Backproject] "
            f"idx={point.index} variant={point.variant} pixel=({point.pixel_u},{point.pixel_v}) "
            f"depth={point.depth_m:.4f} "
            f"world=({point.world_x:.4f},{point.world_y:.4f},{point.world_z:.4f}) "
            f"occ=({point.occ_row},{point.occ_col}) state={point.occ_state}",
            flush=True,
        )

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        load_scene(scene_cfg, simulation_app=simulation_app)
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Stage is not available after scene load")
        add_occupancy_overlay(
            stage=stage,
            occupancy_map=occupancy_map,
            prim_path=args.occupancy_prim_path,
            z_value=float(args.occupancy_z),
            thickness=0.02,
            show_free=False,
            hide_unknown=False,
        )
        _add_world_markers(stage, args.prim_path, debug_points)
        print("[Backproject] Scene loaded. Red spheres use the current formula; green spheres flip the lateral sign. Close Isaac Sim to exit.", flush=True)
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
