from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SceneLoadResult:
    stage_url: str
    has_physics_scene: bool
    physics_scene_path: str | None
    stage_prim_count: int


def _open_stage(stage_url: str, simulation_app=None):
    import omni.usd

    ctx = omni.usd.get_context()
    ctx.open_stage(stage_url)

    # open_stage is async — pump frames until the stage is ready
    if simulation_app is not None:
        for _ in range(20):
            simulation_app.update()
            stage = ctx.get_stage()
            if stage is not None:
                return stage

    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError(f"Failed to open stage after pumping frames: {stage_url}")
    return stage


def _set_physics_scene_synchronous(stage) -> tuple[bool, str | None]:
    from pxr import PhysxSchema, UsdPhysics

    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
            physx_scene.GetUpdateTypeAttr().Set("Synchronous")
            return True, str(prim.GetPath())
    return False, None


def load_scene(scene_cfg: dict[str, Any], simulation_app=None) -> SceneLoadResult:
    stage_url = scene_cfg.get("stage_url")
    if not stage_url:
        raise ValueError("Missing required config field: stage_url")
    if not Path(stage_url).exists():
        raise FileNotFoundError(f"Scene file does not exist: {stage_url}")

    stage = _open_stage(stage_url, simulation_app)
    has_physics_scene, physics_scene_path = _set_physics_scene_synchronous(stage)

    stage_prim_count = sum(1 for _ in stage.Traverse())
    return SceneLoadResult(
        stage_url=stage_url,
        has_physics_scene=has_physics_scene,
        physics_scene_path=physics_scene_path,
        stage_prim_count=stage_prim_count,
    )
