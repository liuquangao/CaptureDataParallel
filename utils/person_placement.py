"""在 stage 中放置人物:

- `ensure_scene_query_support` 确保物理场景支持 scene query(允许 raycast)
- `place_debug_character` 从占用图中采一个符合约束的世界点,落人物 USD
"""
from __future__ import annotations

import math
import random


DEFAULT_CHARACTER_USD_PATH = (
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
)


def _iter_grid_line_cells(row0: int, col0: int, row1: int, col1: int):
    d_row = abs(int(row1) - int(row0))
    d_col = abs(int(col1) - int(col0))
    step_row = 1 if int(row0) < int(row1) else -1
    step_col = 1 if int(col0) < int(col1) else -1
    err = d_col - d_row
    row, col = int(row0), int(col0)

    while True:
        yield row, col
        if row == int(row1) and col == int(col1):
            break
        err2 = 2 * err
        if err2 > -d_row:
            err -= d_row
            col += step_col
        if err2 < d_col:
            err += d_col
            row += step_row


def _is_free_line_in_occupancy(
    occupancy_map,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
) -> bool:
    start_row, start_col = occupancy_map.world_to_grid(float(start_xy[0]), float(start_xy[1]))
    end_row, end_col = occupancy_map.world_to_grid(float(end_xy[0]), float(end_xy[1]))
    if not (0 <= start_row < occupancy_map.height and 0 <= start_col < occupancy_map.width):
        return False
    if not (0 <= end_row < occupancy_map.height and 0 <= end_col < occupancy_map.width):
        return False

    for row, col in _iter_grid_line_cells(start_row, start_col, end_row, end_col):
        if not (0 <= row < occupancy_map.height and 0 <= col < occupancy_map.width):
            return False
        if not bool(occupancy_map.free_mask[row, col]):
            return False
    return True


def sample_person_near_anchor(
    occupancy_map,
    anchor_position_xy: tuple[float, float],
    rng: random.Random,
    min_distance_m: float,
    max_distance_m: float,
    min_obstacle_distance_m: float = 0.0,
    require_pair_connectivity: bool = True,
) -> tuple[float, float]:
    base_mask = occupancy_map.room_free_mask if occupancy_map.room_free_mask is not None else occupancy_map.free_mask
    candidate_cells = []
    anchor_x = float(anchor_position_xy[0])
    anchor_y = float(anchor_position_xy[1])
    min_distance_m = float(min_distance_m)
    max_distance_m = float(max_distance_m)

    for row, col in zip(*base_mask.nonzero()):
        row_i, col_i = int(row), int(col)
        if not occupancy_map._is_cell_clear_of_obstacles(row_i, col_i, float(min_obstacle_distance_m)):
            continue
        cand_x, cand_y = occupancy_map.grid_to_world(row_i, col_i)
        dx = float(cand_x) - anchor_x
        dy = float(cand_y) - anchor_y
        distance_m = (dx * dx + dy * dy) ** 0.5
        if distance_m < min_distance_m or distance_m > max_distance_m:
            continue
        if require_pair_connectivity and not _is_free_line_in_occupancy(
            occupancy_map,
            (anchor_x, anchor_y),
            (float(cand_x), float(cand_y)),
        ):
            continue
        candidate_cells.append((float(cand_x), float(cand_y)))

    if not candidate_cells:
        raise RuntimeError("No valid secondary-person candidates found near anchor position")
    rng.shuffle(candidate_cells)
    return candidate_cells[0]


def place_person_near_anchor(
    stage,
    occupancy_map,
    anchor_position_xy: tuple[float, float],
    prim_path: str,
    seed: int = 0,
    character_usd_path: str = DEFAULT_CHARACTER_USD_PATH,
    arm_drop_degrees: float = 75.0,
    min_distance_m: float = 1.5,
    max_distance_m: float = 3.5,
    min_obstacle_distance_m: float = 0.2,
    require_pair_connectivity: bool = True,
    reuse_existing_prim: bool = False,
) -> dict:
    rng = random.Random(int(seed))
    char_x, char_y = sample_person_near_anchor(
        occupancy_map=occupancy_map,
        anchor_position_xy=anchor_position_xy,
        rng=rng,
        min_distance_m=float(min_distance_m),
        max_distance_m=float(max_distance_m),
        min_obstacle_distance_m=float(min_obstacle_distance_m),
        require_pair_connectivity=bool(require_pair_connectivity),
    )
    floor_z = 0.0
    character_yaw_rad = float(rng.uniform(-math.pi, math.pi))

    existing_prim = stage.GetPrimAtPath(prim_path)
    if reuse_existing_prim and existing_prim and existing_prim.IsValid():
        person_prim_path = _set_xform_translation_and_yaw(
            stage=stage,
            prim_path=prim_path,
            translation_xyz=(float(char_x), float(char_y), floor_z),
            yaw_rad=character_yaw_rad,
        )
        pose_info = None
    else:
        person_prim_path = _add_usd_reference_xform(
            stage=stage,
            usd_path=character_usd_path,
            prim_path=prim_path,
            translation_xyz=(float(char_x), float(char_y), floor_z),
            yaw_rad=character_yaw_rad,
        )
        pose_info = _apply_static_standing_pose(stage, person_prim_path, arm_drop_degrees=arm_drop_degrees)
    return {
        "prim_path": person_prim_path,
        "position": [float(char_x), float(char_y), floor_z],
        "yaw_rad": character_yaw_rad,
        "character_usd_path": character_usd_path,
        "pose_info": pose_info,
    }


def ensure_scene_query_support(stage) -> tuple[str, bool]:
    from pxr import PhysxSchema, UsdPhysics

    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
            physx_scene.GetUpdateTypeAttr().Set("Synchronous")
            if physx_scene.GetEnableSceneQuerySupportAttr().Get() is None:
                physx_scene.CreateEnableSceneQuerySupportAttr(True)
            else:
                physx_scene.GetEnableSceneQuerySupportAttr().Set(True)
            return str(prim.GetPath()), False

    physics_scene_path = "/World/PhysicsScene"
    scene_prim = UsdPhysics.Scene.Define(stage, physics_scene_path).GetPrim()
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    physx_scene.GetUpdateTypeAttr().Set("Synchronous")
    physx_scene.CreateEnableSceneQuerySupportAttr(True)
    return physics_scene_path, True


def _add_usd_reference_xform(
    stage,
    usd_path: str,
    prim_path: str,
    translation_xyz: tuple[float, float, float],
    yaw_rad: float = 0.0,
) -> str:
    from pxr import Gf, UsdGeom

    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    xform = UsdGeom.Xform.Define(stage, prim_path)
    xformable = UsdGeom.Xformable(xform.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation_xyz]))
    xformable.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(0.0, 0.0, math.degrees(float(yaw_rad)))
    )

    asset_path = f"{prim_path}/Asset"
    asset_xform = UsdGeom.Xform.Define(stage, asset_path)
    asset_xform.GetPrim().GetReferences().AddReference(str(usd_path))
    return prim_path


def _set_xform_translation_and_yaw(
    stage,
    prim_path: str,
    translation_xyz: tuple[float, float, float],
    yaw_rad: float = 0.0,
) -> str:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    xformable = UsdGeom.Xformable(xform.GetPrim())
    translate_op = None
    rotate_op = None

    for op in xformable.GetOrderedXformOps():
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        elif op_type == UsdGeom.XformOp.TypeRotateXYZ and rotate_op is None:
            rotate_op = op

    if translate_op is None or rotate_op is None:
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp()
        rotate_op = xformable.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble)

    translate_op.Set(Gf.Vec3d(*[float(v) for v in translation_xyz]))
    rotate_op.Set(Gf.Vec3d(0.0, 0.0, math.degrees(float(yaw_rad))))
    return prim_path


def _find_first_skeleton_under(stage, root_path: str):
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Cannot search skeleton under invalid prim: {root_path}")

    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == "Skeleton":
            return prim
    raise RuntimeError(f"No Skeleton prim found under {root_path}")


def _rotation_matrix(axis_xyz: tuple[float, float, float], degrees: float):
    from pxr import Gf

    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Rotation(Gf.Vec3d(*axis_xyz), float(degrees)))
    return matrix


def _apply_static_standing_pose(stage, person_root_path: str, arm_drop_degrees: float = 75.0) -> dict:
    skeleton_prim = _find_first_skeleton_under(stage, person_root_path)
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

    suffix_to_index = {joint.split("/")[-1]: index for index, joint in enumerate(joints)}
    required_joints = ("L_Upperarm", "R_Upperarm")
    missing = [name for name in required_joints if name not in suffix_to_index]
    if missing:
        raise RuntimeError(f"Skeleton does not contain expected upper-arm joints: {missing}")

    left_rotation = _rotation_matrix((0.0, 0.0, 1.0), -float(arm_drop_degrees))
    right_rotation = _rotation_matrix((0.0, 0.0, 1.0), float(arm_drop_degrees))
    modified_transforms = list(rest_transforms)
    modified_transforms[suffix_to_index["L_Upperarm"]] = (
        left_rotation * modified_transforms[suffix_to_index["L_Upperarm"]]
    )
    modified_transforms[suffix_to_index["R_Upperarm"]] = (
        right_rotation * modified_transforms[suffix_to_index["R_Upperarm"]]
    )
    rest_attr.Set(modified_transforms)

    return {
        "mode": "static_standing_pose",
        "skeleton_path": skeleton_prim.GetPath().pathString,
        "modified_joints": [joints[suffix_to_index[name]] for name in required_joints],
        "arm_drop_degrees": float(arm_drop_degrees),
    }


def place_person(
    stage,
    occupancy_map,
    prim_path: str = "/SDG/Person",
    seed: int = 0,
    character_usd_path: str = DEFAULT_CHARACTER_USD_PATH,
    arm_drop_degrees: float = 75.0,
    min_obstacle_distance_m: float = 0.2,
    existing_world_points_xy: list[tuple[float, float]] | None = None,
    min_point_distance_m: float = 3.0,
    reuse_existing_prim: bool = False,
) -> dict:
    rng = random.Random(seed)
    char_x, char_y = occupancy_map.sample_room_free_world_point_with_constraints(
        rng=rng,
        min_obstacle_distance_m=float(min_obstacle_distance_m),
        existing_world_points_xy=existing_world_points_xy,
        min_point_distance_m=float(min_point_distance_m),
    )
    floor_z = 0.0
    character_yaw_rad = float(rng.uniform(-math.pi, math.pi))

    existing_prim = stage.GetPrimAtPath(prim_path)
    if reuse_existing_prim and existing_prim and existing_prim.IsValid():
        person_prim_path = _set_xform_translation_and_yaw(
            stage=stage,
            prim_path=prim_path,
            translation_xyz=(float(char_x), float(char_y), floor_z),
            yaw_rad=character_yaw_rad,
        )
        pose_info = None
    else:
        person_prim_path = _add_usd_reference_xform(
            stage=stage,
            usd_path=character_usd_path,
            prim_path=prim_path,
            translation_xyz=(float(char_x), float(char_y), floor_z),
            yaw_rad=character_yaw_rad,
        )
        pose_info = _apply_static_standing_pose(stage, person_prim_path, arm_drop_degrees=arm_drop_degrees)
    return {
        "prim_path": person_prim_path,
        "position": [float(char_x), float(char_y), floor_z],
        "yaw_rad": character_yaw_rad,
        "character_usd_path": character_usd_path,
        "pose_info": pose_info,
    }


def place_person_pair(
    stage,
    occupancy_map,
    first_prim_path: str = "/SDG/Persons/person_000",
    second_prim_path: str = "/SDG/Persons/person_001",
    seed: int = 0,
    character_usd_path: str = DEFAULT_CHARACTER_USD_PATH,
    arm_drop_degrees: float = 75.0,
    first_person_min_obstacle_distance_m: float = 0.2,
    second_person_min_obstacle_distance_m: float = 0.2,
    existing_world_points_xy: list[tuple[float, float]] | None = None,
    min_point_distance_m: float = 3.0,
    min_pair_distance_m: float = 1.5,
    max_pair_distance_m: float = 3.5,
    require_pair_connectivity: bool = True,
    max_second_person_attempts: int = 12,
    reuse_existing_first_prim: bool = False,
    reuse_existing_second_prim: bool = False,
) -> dict:
    first_result = place_person(
        stage=stage,
        occupancy_map=occupancy_map,
        prim_path=first_prim_path,
        seed=int(seed),
        character_usd_path=character_usd_path,
        arm_drop_degrees=arm_drop_degrees,
        min_obstacle_distance_m=float(first_person_min_obstacle_distance_m),
        existing_world_points_xy=existing_world_points_xy,
        min_point_distance_m=float(min_point_distance_m),
        reuse_existing_prim=reuse_existing_first_prim,
    )

    rng = random.Random(int(seed) + 1)
    second_result = None
    first_position = first_result["position"]
    for _ in range(max(1, int(max_second_person_attempts))):
        second_result = place_person_near_anchor(
            stage=stage,
            occupancy_map=occupancy_map,
            anchor_position_xy=(float(first_position[0]), float(first_position[1])),
            prim_path=second_prim_path,
            seed=int(rng.randrange(1 << 30)),
            character_usd_path=character_usd_path,
            arm_drop_degrees=arm_drop_degrees,
            min_distance_m=float(min_pair_distance_m),
            max_distance_m=float(max_pair_distance_m),
            min_obstacle_distance_m=float(second_person_min_obstacle_distance_m),
            require_pair_connectivity=bool(require_pair_connectivity),
            reuse_existing_prim=reuse_existing_second_prim,
        )
        break

    if second_result is None:
        raise RuntimeError("Failed to place secondary person near anchor person")

    return {
        "persons": [
            {
                "instance_id": "person_000",
                **first_result,
            },
            {
                "instance_id": "person_001",
                **second_result,
            },
        ]
    }
