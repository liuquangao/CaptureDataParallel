"""场景选择 / 过滤。

- `resolve_scene_configs(cfg)`:根据 ``dataset.selection`` (single/random/all)
  枚举目标 scene_id,并返回带 ``stage_url`` / ``interiorgs_root`` 的
  per-scene 配置列表,供外层 for 循环使用。
- `check_scene_filter(occupancy_map, cfg)`:根据 ``scene_filter.max_free_ratio``
  判断当前占用图是否像"未封闭场景",若是则跳过。调用方应先 load 占用图,
  避免重复 IO。
"""
from __future__ import annotations

import random
from pathlib import Path


def _collect_available_interiorgs_scene_ids(interiorgs_root: Path) -> set[str]:
    available: set[str] = set()
    for path in interiorgs_root.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if "_" not in name:
            continue
        scene_id = name.rsplit("_", 1)[-1].strip()
        if scene_id:
            available.add(scene_id)
    return available


def _merge_scene_cfg(base_scene_cfg: dict, scene_id: str, stage_root: str, interiorgs_root: str) -> dict:
    merged = dict(base_scene_cfg)
    merged["name"] = str(scene_id)
    merged["stage_url"] = str(Path(stage_root) / f"{scene_id}.usda")
    merged["interiorgs_root"] = str(interiorgs_root)
    return merged


def resolve_scene_configs(cfg: dict) -> list[dict]:
    scene_cfg = cfg.get("scene", {}) or {}
    dataset_cfg = cfg.get("dataset")
    if not dataset_cfg:
        # 单场景模式:直接用 cfg["env_url"]
        env_url = cfg.get("env_url")
        if not env_url:
            raise ValueError("Missing dataset config and env_url")
        merged = dict(scene_cfg)
        merged["name"] = Path(env_url).stem
        merged["stage_url"] = str(env_url)
        return [merged]

    stage_root = Path(dataset_cfg.get("stage_root", ""))
    interiorgs_root = Path(dataset_cfg.get("interiorgs_root", ""))
    if not stage_root.exists():
        raise FileNotFoundError(f"dataset.stage_root does not exist: {stage_root}")
    if not interiorgs_root.exists():
        raise FileNotFoundError(f"dataset.interiorgs_root does not exist: {interiorgs_root}")

    selection_cfg = dataset_cfg.get("selection", {}) or {}
    mode = str(selection_cfg.get("mode", "single"))

    all_scene_ids = sorted(path.stem for path in stage_root.glob("*.usda"))
    if not all_scene_ids:
        raise RuntimeError(f"No .usda scenes found under {stage_root}")
    available = _collect_available_interiorgs_scene_ids(interiorgs_root)
    supported = sorted(s for s in all_scene_ids if s in available)
    if not supported:
        raise RuntimeError(
            f"No scenes under {stage_root} have matching InteriorGS occupancy under {interiorgs_root}"
        )

    if mode == "single":
        explicit_ids = selection_cfg.get("scene_ids")
        if explicit_ids:
            scene_ids = [str(sid) for sid in explicit_ids]
        else:
            sid = selection_cfg.get("scene_id")
            if sid is None:
                raise ValueError("dataset.selection.mode=single requires scene_id or scene_ids")
            scene_ids = [str(sid)]
    elif mode == "random":
        random_count = int(selection_cfg.get("random_count", 1))
        random_seed = int(selection_cfg.get("random_seed", 0))
        if random_count <= 0:
            raise ValueError("dataset.selection.random_count must be > 0")
        if random_count > len(supported):
            raise ValueError(
                f"dataset.selection.random_count={random_count} exceeds supported scenes={len(supported)}"
            )
        rng = random.Random(random_seed)
        scene_ids = sorted(rng.sample(supported, random_count))
    elif mode == "all":
        scene_ids = supported
    else:
        raise ValueError(f"Unsupported dataset.selection.mode: {mode}")

    excluded = {str(sid) for sid in selection_cfg.get("exclude_scene_ids", [])}
    if excluded:
        scene_ids = [sid for sid in scene_ids if sid not in excluded]

    missing = [sid for sid in scene_ids if sid not in all_scene_ids]
    if missing:
        raise FileNotFoundError(f"Requested scene ids not found under {stage_root}: {missing}")
    if mode == "single":
        missing_occ = [sid for sid in scene_ids if sid not in available]
        if missing_occ:
            raise FileNotFoundError(
                f"Requested scene ids do not have matching InteriorGS occupancy under {interiorgs_root}: {missing_occ}"
            )

    return [
        _merge_scene_cfg(scene_cfg, sid, str(stage_root), str(interiorgs_root))
        for sid in scene_ids
    ]


def check_scene_filter(occupancy_map, cfg: dict) -> tuple[bool, str | None]:
    """根据 scene_filter.max_free_ratio 判断是否跳过当前场景。"""
    scene_filter_cfg = cfg.get("scene_filter", {}) or {}
    max_free_ratio = scene_filter_cfg.get("max_free_ratio")
    total = int(occupancy_map.width) * int(occupancy_map.height)
    if max_free_ratio is not None and total > 0:
        free_ratio = float(int(occupancy_map.free_mask.sum())) / float(total)
        if free_ratio > float(max_free_ratio):
            return True, (
                f"occupancy free_ratio={free_ratio:.3f} > max_free_ratio={float(max_free_ratio):.3f} "
                f"(likely unclosed / unbounded scene)"
            )

    min_room_free_area_m2 = scene_filter_cfg.get("min_room_free_area_m2")
    if min_room_free_area_m2 is not None:
        usable_mask = occupancy_map.room_free_mask if occupancy_map.room_free_mask is not None else occupancy_map.free_mask
        usable_area_m2 = float(int(usable_mask.sum())) * float(occupancy_map.resolution) ** 2
        if usable_area_m2 < float(min_room_free_area_m2):
            return True, (
                f"room_free_area_m2={usable_area_m2:.3f} < "
                f"min_room_free_area_m2={float(min_room_free_area_m2):.3f}"
            )
    return False, None
