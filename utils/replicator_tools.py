"""Replicator 多相机并行采集的原子工具。

所有函数都需要 SimulationApp 启动之后才能使用(omni.* 模块在运行时才可 import)。
共同约定:pose 字典至少含 ``x``, ``y``, ``camera_z`` 三个键。
"""
from __future__ import annotations

from typing import Iterable


_ANNOTATOR_INIT_PARAMS = {
    "semantic_segmentation": {"colorize": False},
    "instance_id_segmentation": {"colorize": False},
    "rgb": None,
    "distance_to_image_plane": None,
}


def set_prim_visibility(stage, path: str, visible: bool, recursive: bool = True) -> bool:
    """切换 prim(以及可选子树)的 Imageable 可见性。"""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return False

    def _apply(p):
        im = UsdGeom.Imageable(p)
        if not im:
            return
        (im.MakeVisible() if visible else im.MakeInvisible())

    _apply(prim)
    if recursive:
        for child in Usd.PrimRange(prim):
            if child != prim:
                _apply(child)
    return True


def create_camera_pool(
    num_cameras: int,
    resolution: tuple[int, int],
    focal_length: float,
    focus_distance: float,
    vertical_aperture: float,
    near_clipping_distance: float = 0.01,
    far_clipping_distance: float = 1e7,
    parent: str = "/SDG",
    scope_name: str = "Cameras",
    name_prefix: str = "Cam",
    view_prefix: str = "View",
    annotator_names: Iterable[str] = ("semantic_segmentation",),
):
    """创建 N 个 driver camera + N 个 render product + 多组 annotators。

    Returns
    -------
    driver_cams : list[pxr.Usd.Prim]
        N 个 driver camera prim。
    render_products : list
        N 个 render_product。
    annotators_by_name : dict[str, list]
        ``annotators_by_name[annotator_name][i]`` 是第 ``i`` 个相机上
        对应 annotator 的句柄,可直接 ``get_data()``。
    """
    import omni.replicator.core as rep
    from pxr import UsdGeom

    rep.functional.create.scope(name=scope_name, parent=parent)
    scope_path = f"{parent}/{scope_name}"

    driver_cams = [
        rep.functional.create.camera(
            focus_distance=float(focus_distance),
            focal_length=float(focal_length),
            clipping_range=(float(near_clipping_distance), float(far_clipping_distance)),
            name=f"{name_prefix}_{i:02d}",
            parent=scope_path,
        )
        for i in range(int(num_cameras))
    ]

    # rep 只支持 horizontal_aperture 参数;vertical_aperture 通过 USD 属性补上
    for cam_prim in driver_cams:
        usd_cam = UsdGeom.Camera(cam_prim)
        if usd_cam:
            usd_cam.GetVerticalApertureAttr().Set(float(vertical_aperture))

    render_products = [
        rep.create.render_product(cam, tuple(resolution), name=f"{view_prefix}_{i:02d}")
        for i, cam in enumerate(driver_cams)
    ]

    annotators_by_name: dict[str, list] = {}
    for name in annotator_names:
        init_params = _ANNOTATOR_INIT_PARAMS.get(name)
        bucket = []
        for rp in render_products:
            if init_params is None:
                ann = rep.AnnotatorRegistry.get_annotator(name)
            else:
                ann = rep.AnnotatorRegistry.get_annotator(name, init_params=init_params)
            ann.attach(rp)
            bucket.append(ann)
        annotators_by_name[name] = bucket

    return driver_cams, render_products, annotators_by_name


def set_render_products_updates_enabled(render_products: Iterable, enabled: bool) -> None:
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(bool(enabled))


def teardown_camera_pool(
    stage,
    annotators_by_name: dict[str, list],
    render_products: Iterable,
    scope_path: str = "/SDG/Cameras",
) -> None:
    """拆相机池:detach 所有 annotator + 关 render_product 更新 + 删 scope prim。

    多场景循环里每开新 stage 前调一次,避免残留 hydra handle 指向旧 stage。
    """
    for bucket in annotators_by_name.values():
        for ann in bucket:
            try:
                ann.detach()
            except Exception:
                pass
    try:
        set_render_products_updates_enabled(render_products, False)
    except Exception:
        pass
    if stage is not None:
        prim = stage.GetPrimAtPath(scope_path)
        if prim and prim.IsValid():
            stage.RemovePrim(scope_path)


def set_batch_poses_with_orientation(driver_cams, batch: list[dict]) -> None:
    """显式 yaw→quaternion 方式设置位姿。``batch[i]`` 需含 ``x,y,camera_z,yaw_rad``。"""
    import omni.replicator.core as rep

    for cam, pose in zip(driver_cams[: len(batch)], batch):
        yaw = float(pose["yaw_rad"])
        import math
        # look_at 沿相机前向取一个远点,避免 rep 内部再做坐标变换导致 up 轴漂移
        fx = float(pose["x"]) + math.cos(yaw)
        fy = float(pose["y"]) + math.sin(yaw)
        rep.functional.modify.pose(
            cam,
            position_value=(float(pose["x"]), float(pose["y"]), float(pose["camera_z"])),
            look_at_value=(fx, fy, float(pose["camera_z"])),
            look_at_up_axis=(0, 0, 1),
        )


def iter_batches(items: list, batch_size: int):
    for start in range(0, len(items), int(batch_size)):
        yield items[start : start + int(batch_size)]
