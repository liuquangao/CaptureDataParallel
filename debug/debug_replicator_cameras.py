"""调试多摄像头采集"""

import argparse
import json
import math
import os
import time

import numpy as np
import yaml
from isaacsim import SimulationApp

import resource
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))


# 这里是各种参数的默认配置
config = {
    "launch_config": {
        "renderer": "RealTimePathTracing",
        "headless": False,
    },
    "resolution": [640, 640],
    "rt_subframes": 4,
    "num_frames": 10,
    "env_url": "/home/leo/FusionLab/DataSets/spatialverse/SAGE-3D_InteriorGS_usda/839912.usda",
    "writer": "BasicWriter",
    "backend_type": "DiskBackend",
    "backend_params": {
        "output_dir": "/home/leo/FusionLab/CaptureDataParallel",
    },
    "writer_config": {
        "rgb": False,
        "bounding_box_2d_tight": False,
        "semantic_segmentation": True,
        "distance_to_image_plane": False,
        "bounding_box_3d": False,
        "occlusion": False,
    },
    "clear_previous_semantics": True,
    "person": {
        "url": "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/"
                "Assets/Isaac/6.0/Isaac/People/Characters/F_Business_02/F_Business_02.usd",
        "class": "person",
    },
    "close_app_after_run": True,
}

import carb

# Parse command line arguments for optional config file
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
args, unknown = parser.parse_known_args()

# 初始化仿APP
simulation_app = SimulationApp(launch_config=config["launch_config"])

import carb.settings

# 必须在 SimulationApp 之后才能访问 carb settings；把日志级别压到 error 以屏蔽 hydra/ warning 噪声
carb.settings.get_settings().set("/log/level", "error")
# 或者更精细地关某些 channel:
# carb.settings.get_settings().set("/log/channels/carb", "error")

# Runtime modules (must import after SimulationApp creation)
import omni.replicator.core as rep
import omni.usd
# import scene_based_sdg_utils
from isaacsim.core.experimental.utils.semantics import add_labels, remove_all_labels
from isaacsim.core.utils.stage import get_current_stage, open_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Usd, UsdGeom


NUM_CAMERAS = 16


def read_person_counts(annotators):
    """从每个 annotator 读一张 seg mask,返回每个视图里 person 的像素数。"""
    counts = []
    for ann in annotators:
        data = ann.get_data()
        seg = data["data"]  # HxW uint32, 每个像素是 class id
        id_to_labels = data["info"]["idToLabels"]  # {"0": {"class": "..."}, ...}
        person_ids = [int(k) for k, v in id_to_labels.items() if v.get("class") == "person"]
        counts.append(int(np.isin(seg, person_ids).sum()) if person_ids else 0)
    return counts

# 语义分割模式:隐藏 3DGS 体积,显示 mesh,并给 mesh 和人物各打一个 class 标签
def _set_prim_visibility(path: str, visible: bool, recursive: bool = True) -> bool:
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

# 得到资产根路径
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not get nucleus server path, closing application..")
    simulation_app.close()

# 加载环境的舞台
print(f"[SDG] Loading Stage {config['env_url']}")
if not open_stage(config["env_url"]):
    carb.log_error(f"Could not open stage {config['env_url']}, closing application..")
    simulation_app.close()


# 随机
rep.set_global_seed(42)
rng = np.random.default_rng(42)

# Configure replicator for manual triggering
rep.orchestrator.set_capture_on_play(False)


carb.settings.get_settings().set("/rtx/rendermode", "RayTracedLighting")
carb.settings.get_settings().set("rtx/post/dlss/execMode", 0)
# # 0: Disabled, 1: TAA, 2: FXAA, 3: DLSS, 4:RTXAA
# carb.settings.get_settings().set("/rtx/post/aa/op", 1)  # 关掉所有抗锯齿,保持像素纯净


# Clear previous semantic labels
if config["clear_previous_semantics"]:
    for prim in get_current_stage().Traverse():
        remove_all_labels(prim, include_descendants=True)

# Create SDG scope for organizing all generated objects
stage = get_current_stage()
sdg_scope = stage.DefinePrim("/SDG", "Scope")


# 加载 occupancy + 采样人物落点 + 采样 16 个相机点
import yaml, sys
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from collector.debug_viz import place_debug_character
from collector.occupancy import load_interiorgs_occupancy_map
from collector.score_field import _has_full_width_occupancy_visibility, iter_ring_camera_samples

dataset_cfg = yaml.safe_load(open(THIS_DIR / "configs" / "sage3d.yaml"))
scene_cfg = {**dataset_cfg["dataset"], **dataset_cfg["scene"], "stage_url": config["env_url"]}

occ = load_interiorgs_occupancy_map(scene_cfg)

# 采样 + 放置人物(和 run_collector 保持一致)
debug_char = place_debug_character(
    stage=stage,
    occupancy_map=occ,
    prim_path="/SDG/Person",
    seed=42,
    character_usd_path=config["person"]["url"],
    min_obstacle_distance_m=dataset_cfg["sampling"]["character_min_obstacle_distance_m"],
)
person_xyz = debug_char["position"]  # [x, y, z=0]



_set_prim_visibility("/World/gauss", visible=True)
_set_prim_visibility("/World/scene_collision", visible=False)

add_labels("/World/scene_collision", labels="scene", taxonomy="class")
add_labels("/SDG/Person", labels="person", taxonomy="class")


ring = list(iter_ring_camera_samples(
    occupancy_map=occ,
    person_position_xy=(person_xyz[0], person_xyz[1]),
    camera_height_m=dataset_cfg["capture_camera"]["camera_height"],
    min_radius_m=dataset_cfg["score_field"]["min_radius_m"],
    max_radius_m=dataset_cfg["score_field"]["max_radius_m"],
    grid_step_m=dataset_cfg["score_field"]["grid_step_m"],
    min_obstacle_distance_m=dataset_cfg["score_field"]["camera_min_obstacle_distance_m"],
))
camera_poses = list(ring)
print(f"[SDG] Ring produced {len(camera_poses)} candidate poses; will evaluate in batches of {NUM_CAMERAS}.")

# 32 个摄像头,分别命名为 ScoreCam_00, ScoreCam_01, ..., ScoreCam_31
# 每批最多 32 个 pose 复用这些相机；最后一批不满 32 时只更新前 len(batch) 个
rep.functional.create.scope(name="Cameras", parent="/SDG")

driver_cams = [
    rep.functional.create.camera(
        focus_distance=400.0,
        focal_length=8.0,
        clipping_range=(0.01, 10000000.0),
        name=f"ScoreCam_{i:02d}",
        parent="/SDG/Cameras",
    )
    for i in range(NUM_CAMERAS)
]

# 设置渲染:为 16 个相机各建一个 render product,并先关闭其更新(采集时再开)
resolution = config.get("resolution", (128, 128))
render_products = [
    rep.create.render_product(cam, resolution, name=f"ScoreView_{i:02d}")
    for i, cam in enumerate(driver_cams)
]

for rp in render_products:
    rp.hydra_texture.set_updates_enabled(False)

# 为每个 render product 挂一个 semantic_segmentation annotator
# 这里不是在采集，只是挂好 annotator，等采集时开更新
seg_annotators = []
for rp in render_products:
    ann = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
    ann.attach(rp)
    seg_annotators.append(ann)

rt_subframes = config.get("rt_subframes", -1)


def _set_batch_poses(batch):
    """只更新前 len(batch) 个相机的位姿；多余相机保留上一批位姿,结果会在读取时忽略。"""
    for cam, pose in zip(driver_cams[: len(batch)], batch):
        rep.functional.modify.pose(
            cam,
            position_value=(pose["x"], pose["y"], pose["camera_z"]),
            look_at_value=(person_xyz[0], person_xyz[1], person_xyz[2] + 1.0),
            look_at_up_axis=(0, 0, 1),
        )


def _iter_batches(poses, batch_size):
    for start in range(0, len(poses), batch_size):
        yield poses[start : start + batch_size]


# 预过滤：用 occupancy 判定"全身宽度完全可见"的短路点 -> 直接 score=1.0,不进相机
body_width_m = float(dataset_cfg["score_field"]["occupancy_full_visibility_width_m"])
certain_indices: list[int] = []
uncertain_indices: list[int] = []
for i, pose in enumerate(camera_poses):
    if _has_full_width_occupancy_visibility(
        occupancy_map=occ,
        person_position_xy=(person_xyz[0], person_xyz[1]),
        camera_position_xy=(pose["x"], pose["y"]),
        body_width_m=body_width_m,
    ):
        certain_indices.append(i)
    else:
        uncertain_indices.append(i)
uncertain_poses = [camera_poses[i] for i in uncertain_indices]
print(
    f"[SDG] Occupancy shortcut: {len(certain_indices)} certain (score=1.0), "
    f"{len(uncertain_indices)} uncertain -> rendering."
)


uncertain_total_counts: list[int] = []
uncertain_visible_counts: list[int] = []
pass1_elapsed = 0.0
pass2_elapsed = 0.0

for rp in render_products:
    rp.hydra_texture.set_updates_enabled(True)

# headless 下 hydra 纹理需要先空转几帧完成初始化,否则 orchestrator.step 会触发 GPU pagefault
for _ in range(4):
    simulation_app.update()

if uncertain_poses:
    # Pass 1: 藏 mesh,拍"无遮挡"下的人物像素(作为分母 total)
    print(f"[SDG] Pass 1: hiding mesh and capturing {len(uncertain_poses)} unoccluded views (total_counts)...")
    _set_prim_visibility("/World/scene_collision", visible=False)
    
    pass1_start = time.perf_counter()
    for batch_idx, batch in enumerate(_iter_batches(uncertain_poses, NUM_CAMERAS)):
        _set_batch_poses(batch)
        rep.orchestrator.step(rt_subframes=2)
        batch_counts = read_person_counts(seg_annotators)
        uncertain_total_counts.extend(batch_counts[: len(batch)])
        print(f"[SDG] Pass 1 batch {batch_idx}: {len(batch)} poses")
    for _ in range(2):
        simulation_app.update()


    pass1_elapsed = time.perf_counter() - pass1_start
    print(f"[SDG] Pass 1 elapsed: {pass1_elapsed:.6f}s")

    # for rp in render_products:
    #         rp.hydra_texture.set_updates_enabled(False)
    # 切换可见性(整个流程中只切一次)
    print("[SDG] Toggling scene_collision visibility: False -> True")
    _set_prim_visibility("/World/scene_collision", visible=True)
    # for rp in render_products:
    #         rp.hydra_texture.set_updates_enabled(True)

    # Pass 2: mesh 可见,拍"有遮挡"下的人物像素(分子 visible)
    print(f"[SDG] Pass 2: capturing {len(uncertain_poses)} occluded views (visible_counts)...")
    pass2_start = time.perf_counter()
    for batch_idx, batch in enumerate(_iter_batches(uncertain_poses, NUM_CAMERAS)):
        _set_batch_poses(batch)
        rep.orchestrator.step(rt_subframes=2)
        batch_counts = read_person_counts(seg_annotators)
        uncertain_visible_counts.extend(batch_counts[: len(batch)])
        print(f"[SDG] Pass 2 batch {batch_idx}: {len(batch)} poses")
    for _ in range(2):
        simulation_app.update()
    pass2_elapsed = time.perf_counter() - pass2_start
    print(f"[SDG] Pass 2 elapsed: {pass2_elapsed:.6f}s")
    print(f"[SDG] Pass delta (pass2 - pass1): {pass2_elapsed - pass1_elapsed:+.6f}s")
else:
    print("[SDG] No uncertain poses; skipping Pass1/Pass2.")

# for rp in render_products:
#     rp.hydra_texture.set_updates_enabled(False)

# 合并 certain + uncertain 的结果,写 JSON
poses_payload = [None] * len(camera_poses)
for i in certain_indices:
    pose = camera_poses[i]
    poses_payload[i] = {
        "idx": int(i),
        "x": float(pose["x"]),
        "y": float(pose["y"]),
        "z": float(pose["z"]),
        "camera_z": float(pose["camera_z"]),
        "yaw_rad": float(pose["yaw_rad"]),
        "distance_m": float(pose["distance_m"]),
        "scoring_mode": "occupancy_full_visibility",
        "score": 1.0,
        "visible_person_pixels": 1,
        "total_person_pixels": 1,
    }
for idx_in_uncertain, i in enumerate(uncertain_indices):
    pose = camera_poses[i]
    vis = int(uncertain_visible_counts[idx_in_uncertain])
    total = int(uncertain_total_counts[idx_in_uncertain])
    score = float(vis) / float(total) if total > 0 else 0.0
    poses_payload[i] = {
        "idx": int(i),
        "x": float(pose["x"]),
        "y": float(pose["y"]),
        "z": float(pose["z"]),
        "camera_z": float(pose["camera_z"]),
        "yaw_rad": float(pose["yaw_rad"]),
        "distance_m": float(pose["distance_m"]),
        "scoring_mode": "segmentation_visibility",
        "score": score,
        "visible_person_pixels": vis,
        "total_person_pixels": total,
    }

output_dir = Path(config["backend_params"]["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
score_field_path = output_dir / "debug_score_field.json"
payload = {
    "person_position": [float(v) for v in person_xyz],
    "num_poses": len(camera_poses),
    "num_certain": len(certain_indices),
    "num_uncertain": len(uncertain_indices),
    "pass1_elapsed_sec": float(pass1_elapsed),
    "pass2_elapsed_sec": float(pass2_elapsed),
    "occupancy_full_visibility_width_m": body_width_m,
    "ring_params": {
        "min_radius_m": float(dataset_cfg["score_field"]["min_radius_m"]),
        "max_radius_m": float(dataset_cfg["score_field"]["max_radius_m"]),
        "grid_step_m": float(dataset_cfg["score_field"]["grid_step_m"]),
        "camera_min_obstacle_distance_m": float(
            dataset_cfg["score_field"]["camera_min_obstacle_distance_m"]
        ),
    },
    "poses": poses_payload,
}
score_field_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"[SDG] Wrote score field to {score_field_path}")

# 保持 GUI 打开,直到用户关闭窗口
print("[SDG] GUI 保持打开,关闭窗口退出。")
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
