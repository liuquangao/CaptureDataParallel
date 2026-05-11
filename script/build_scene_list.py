"""
预先生成通过筛选的场景列表（不需要 Isaac Sim）。

用法：
    python script/build_scene_list.py \
        --config configs/sage3d_parallel.yaml \
        --output scene_list.txt

筛选条件与 run_collector.py 一致：
  - exclude_scene_ids（配置文件里已有 839929）
  - interiorgs_root 下存在匹配占用图
  - scene_filter.max_free_ratio
  - scene_filter.min_room_free_area_m2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.occupancy_map import load_interiorgs_occupancy_map
from utils.scene_selection import check_scene_filter, resolve_scene_configs


def _split(ids: list[str], train_ratio: float, seed: int = 42) -> tuple[list[str], list[str]]:
    import random
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_train = round(len(shuffled) * train_ratio)
    return sorted(shuffled[:n_train]), sorted(shuffled[n_train:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None, help="全量列表写入路径，默认打印到 stdout")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="train 比例，默认 0.9")
    parser.add_argument("--seed", type=int, default=42, help="shuffle 随机种子")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    scene_configs = resolve_scene_configs(config)
    print(f"[list] {len(scene_configs)} candidates after exclude_scene_ids / mode filtering", flush=True)

    passed: list[str] = []
    filtered: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for scene_cfg in scene_configs:
        scene_id = str(scene_cfg.get("name", Path(scene_cfg["stage_url"]).stem))
        try:
            occupancy_map = load_interiorgs_occupancy_map(scene_cfg)
        except Exception as exc:
            print(f"[list] FAIL  {scene_id}: {exc}")
            failed.append((scene_id, str(exc)))
            continue

        skip, reason = check_scene_filter(occupancy_map, config)
        if skip:
            print(f"[list] SKIP  {scene_id}: {reason}")
            filtered.append((scene_id, reason or ""))
            continue

        usable_mask = (
            occupancy_map.room_free_mask
            if occupancy_map.room_free_mask is not None
            else occupancy_map.free_mask
        )
        area_m2 = float(int(usable_mask.sum())) * float(occupancy_map.resolution) ** 2
        print(f"[list] OK    {scene_id}  (room_free_area={area_m2:.1f} m²)")
        passed.append(scene_id)

    print(f"\n[list] passed={len(passed)}  filtered={len(filtered)}  failed={len(failed)}")

    train_ids, test_ids = _split(passed, args.train_ratio, args.seed)
    print(f"[list] train={len(train_ids)}  test={len(test_ids)}  (ratio={args.train_ratio})")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(passed) + "\n", encoding="utf-8")
        train_path = out.parent / (out.stem + "_train" + out.suffix)
        test_path = out.parent / (out.stem + "_test" + out.suffix)
        train_path.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
        test_path.write_text("\n".join(test_ids) + "\n", encoding="utf-8")
        print(f"[list] written: {out}  {train_path}  {test_path}")
    else:
        print("\n--- train ---")
        print("\n".join(train_ids))
        print("\n--- test ---")
        print("\n".join(test_ids))


if __name__ == "__main__":
    main()