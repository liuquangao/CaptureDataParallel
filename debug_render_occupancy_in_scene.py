#!/home/leo/FusionLab/isaacsim/_build/linux-x86_64/release/python.sh

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from collector.occupancy import load_interiorgs_occupancy_map
from collector.scene_loader import load_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a SAGE-3D stage in Isaac Sim and render the loaded occupancy map in-scene."
    )
    parser.add_argument(
        "--stage-url",
        type=str,
        default="/home/leo/FusionLab/DataSets/spatialverse/SAGE-3D_InteriorGS_usda/839875.usda",
        help="Path to the `.usda` stage file to open.",
    )
    parser.add_argument(
        "--interiorgs-root",
        type=str,
        default="/home/leo/FusionLab/DataSets/spatialverse/InteriorGS",
        help="InteriorGS root used to resolve the matching occupancy.json/png.",
    )
    parser.add_argument(
        "--prim-path",
        type=str,
        default="/World/DebugOccupancy",
        help="Root prim path for the rendered occupancy overlay.",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=0.02,
        help="Z height in Isaac world coordinates where the occupancy layer is rendered.",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=0.02,
        help="Thickness of each occupancy cell cuboid in meters.",
    )
    parser.add_argument(
        "--show-free",
        action="store_true",
        help="Also render free cells. By default only occupied and unknown cells are shown.",
    )
    parser.add_argument(
        "--hide-unknown",
        action="store_true",
        help="Do not render unknown cells.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Launch Isaac Sim headless. Default is interactive so RGB rendering can be inspected.",
    )
    return parser.parse_args()


def _remove_existing_prim(stage, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


def _define_preview_material(stage, material_path: str, rgb: tuple[float, float, float], opacity: float):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(float(v) for v in rgb))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mask_to_positions(occupancy_map, mask: np.ndarray, z_value: float):
    from pxr import Gf

    positions = []
    rows, cols = np.where(mask)
    for row, col in zip(rows.tolist(), cols.tolist()):
        world_x, world_y = occupancy_map.grid_to_world(int(row), int(col))
        positions.append(Gf.Vec3f(float(world_x), float(world_y), float(z_value)))
    return positions


def _create_instancer(
    stage,
    root_path: str,
    name: str,
    occupancy_map,
    mask: np.ndarray,
    z_value: float,
    thickness: float,
    rgb: tuple[float, float, float],
    opacity: float,
) -> int:
    from pxr import Gf, UsdGeom, UsdShade, Vt

    positions = _mask_to_positions(occupancy_map, mask, z_value)
    if not positions:
        return 0

    instancer = UsdGeom.PointInstancer.Define(stage, f"{root_path}/{name}")
    prototype = UsdGeom.Cube.Define(stage, f"{root_path}/{name}/Cell")
    prototype.CreateSizeAttr(1.0)

    material = _define_preview_material(stage, f"{root_path}/{name}/Material", rgb=rgb, opacity=opacity)
    UsdShade.MaterialBindingAPI(prototype.GetPrim()).Bind(material)

    proto_indices = Vt.IntArray([0] * len(positions))
    scales = Vt.Vec3fArray(
        [Gf.Vec3f(float(occupancy_map.resolution), float(occupancy_map.resolution), float(thickness))] * len(positions)
    )
    orientations = Vt.QuathArray([Gf.Quath(1.0, Gf.Vec3h(0.0, 0.0, 0.0))] * len(positions))

    instancer.CreatePrototypesRel().SetTargets([prototype.GetPath()])
    instancer.CreateProtoIndicesAttr().Set(proto_indices)
    instancer.CreatePositionsAttr().Set(Vt.Vec3fArray(positions))
    instancer.CreateScalesAttr().Set(scales)
    instancer.CreateOrientationsAttr().Set(orientations)
    return len(positions)


def add_occupancy_overlay(stage, occupancy_map, prim_path: str, z_value: float, thickness: float, show_free: bool, hide_unknown: bool) -> dict:
    from pxr import UsdGeom

    _remove_existing_prim(stage, prim_path)
    UsdGeom.Xform.Define(stage, prim_path)

    counts: dict[str, int] = {}
    if show_free:
        counts["free"] = _create_instancer(
            stage,
            prim_path,
            "free",
            occupancy_map,
            occupancy_map.free_mask,
            z_value,
            thickness,
            rgb=(0.20, 0.85, 0.35),
            opacity=0.20,
        )
    counts["occupied"] = _create_instancer(
        stage,
        prim_path,
        "occupied",
        occupancy_map,
        occupancy_map.occupied_mask,
        z_value,
        thickness,
        rgb=(0.95, 0.15, 0.15),
        opacity=0.85,
    )
    if not hide_unknown:
        counts["unknown"] = _create_instancer(
            stage,
            prim_path,
            "unknown",
            occupancy_map,
            occupancy_map.unknown_mask,
            z_value,
            thickness,
            rgb=(0.95, 0.75, 0.10),
            opacity=0.55,
        )
    return counts


def main() -> None:
    args = parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        scene_cfg = {
            "stage_url": str(args.stage_url),
            "interiorgs_root": str(args.interiorgs_root),
            "occupancy_flip_x": True,
            "occupancy_flip_y": True,
            "occupancy_negate_xy": True,
        }

        load_result = load_scene(scene_cfg, simulation_app=simulation_app)
        occupancy_map = load_interiorgs_occupancy_map(scene_cfg)

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Stage is not available after load_scene")

        counts = add_occupancy_overlay(
            stage=stage,
            occupancy_map=occupancy_map,
            prim_path=str(args.prim_path),
            z_value=float(args.z),
            thickness=float(args.thickness),
            show_free=bool(args.show_free),
            hide_unknown=bool(args.hide_unknown),
        )

        print(
            "[OccupancyOverlay] "
            f"stage={load_result.stage_url} "
            f"prim={args.prim_path} "
            f"z={args.z:.3f} thickness={args.thickness:.3f} "
            f"counts={counts}",
            flush=True,
        )
        print(
            "[OccupancyOverlay] colors: free=green occupied=red unknown=yellow",
            flush=True,
        )
        print(
            "[OccupancyOverlay] Close the Isaac Sim window to exit.",
            flush=True,
        )

        while simulation_app.is_running():
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
