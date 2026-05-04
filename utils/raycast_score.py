"""Skeleton-joint raycast visibility scoring."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RaycastScore:
    score: float
    visible_joints: int
    total_joints: int


def _find_first_skeleton_under(stage, root_path: str):
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Cannot search skeleton under invalid prim: {root_path}")

    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == "Skeleton":
            return prim
    raise RuntimeError(f"No Skeleton prim found under {root_path}")


def get_skeleton_joint_world_positions(stage, person_prim_path: str) -> dict[str, tuple[float, float, float]]:
    from pxr import Gf, Usd, UsdGeom

    skeleton_prim = _find_first_skeleton_under(stage, person_prim_path)
    joints_attr = skeleton_prim.GetAttribute("joints")
    rest_attr = skeleton_prim.GetAttribute("restTransforms")
    if not joints_attr or not rest_attr:
        raise RuntimeError(f"Skeleton is missing joints/restTransforms: {skeleton_prim.GetPath()}")

    joints = [str(joint) for joint in joints_attr.Get()]
    rest_transforms = list(rest_attr.Get())
    if len(joints) != len(rest_transforms):
        raise RuntimeError(
            "Skeleton joints/restTransforms length mismatch: "
            f"{skeleton_prim.GetPath()}, joints={len(joints)}, restTransforms={len(rest_transforms)}"
        )

    joint_to_index = {joint: index for index, joint in enumerate(joints)}
    local_world_by_joint: dict[str, Gf.Matrix4d] = {}

    def _local_world(joint: str) -> Gf.Matrix4d:
        if joint in local_world_by_joint:
            return local_world_by_joint[joint]
        index = joint_to_index[joint]
        parent = joint.rsplit("/", 1)[0] if "/" in joint else ""
        if parent and parent in joint_to_index:
            matrix = rest_transforms[index] * _local_world(parent)
        else:
            matrix = rest_transforms[index]
        local_world_by_joint[joint] = matrix
        return matrix

    skeleton_to_world = UsdGeom.Xformable(skeleton_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    positions: dict[str, tuple[float, float, float]] = {}
    for joint in joints:
        world_matrix = _local_world(joint) * skeleton_to_world
        translation = world_matrix.ExtractTranslation()
        positions[joint] = (float(translation[0]), float(translation[1]), float(translation[2]))
    return positions


def _is_path_under(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def _raycast_hit_path(hit) -> str:
    from pxr import PhysicsSchemaTools

    rigid_body = hit.rigid_body
    if isinstance(rigid_body, str):
        return rigid_body
    if isinstance(rigid_body, int):
        return PhysicsSchemaTools.intToSdfPath(rigid_body).pathString
    raise RuntimeError(f"Unsupported RaycastHit.rigid_body type: {type(rigid_body).__name__}")


def _visible_to_joint(
    query,
    camera_xyz: tuple[float, float, float],
    joint_xyz: tuple[float, float, float],
    target_prim_path: str,
) -> bool:
    start = tuple(float(v) for v in camera_xyz)
    end = tuple(float(v) for v in joint_xyz)
    direction = (
        end[0] - start[0],
        end[1] - start[1],
        end[2] - start[2],
    )
    distance = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    if distance <= 1e-6:
        raise RuntimeError("Raycast joint distance is zero")
    unit_dir = (direction[0] / distance, direction[1] / distance, direction[2] / distance)

    blocking_hit = {"blocked": False}

    def _report_hit(hit):
        path = _raycast_hit_path(hit)
        if _is_path_under(path, target_prim_path):
            return True
        blocking_hit["blocked"] = True
        return False

    query.raycast_all(start, unit_dir, distance, _report_hit)
    return not bool(blocking_hit["blocked"])


def _joint_in_camera_frame(
    camera_xyz: tuple[float, float, float],
    yaw_rad: float,
    joint_xyz: tuple[float, float, float],
    resolution: tuple[int, int],
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    margin_px: float = 0.0,
) -> bool:
    dx = float(joint_xyz[0]) - float(camera_xyz[0])
    dy = float(joint_xyz[1]) - float(camera_xyz[1])
    dz = float(joint_xyz[2]) - float(camera_xyz[2])
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    forward = dx * cos_yaw + dy * sin_yaw
    if forward <= 1e-6:
        return False

    lateral = -dx * sin_yaw + dy * cos_yaw
    vertical = dz
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


def score_target_joint_visibility(
    query,
    camera_xyz: tuple[float, float, float],
    camera_yaw_rad: float,
    resolution: tuple[int, int],
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    target_prim_path: str,
    joints: dict[str, tuple[float, float, float]],
    frame_margin_px: float = 0.0,
) -> RaycastScore:
    visible = 0
    for joint_xyz in joints.values():
        if not _joint_in_camera_frame(
            camera_xyz=camera_xyz,
            yaw_rad=float(camera_yaw_rad),
            joint_xyz=joint_xyz,
            resolution=resolution,
            focal_length=float(focal_length),
            horizontal_aperture=float(horizontal_aperture),
            vertical_aperture=float(vertical_aperture),
            margin_px=float(frame_margin_px),
        ):
            continue
        if _visible_to_joint(
            query=query,
            camera_xyz=camera_xyz,
            joint_xyz=joint_xyz,
            target_prim_path=target_prim_path,
        ):
            visible += 1
    total = len(joints)
    if total <= 0:
        raise RuntimeError("Skeleton contains no joints")
    return RaycastScore(
        score=float(visible) / float(total),
        visible_joints=int(visible),
        total_joints=int(total),
    )
