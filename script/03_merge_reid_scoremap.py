#!/usr/bin/env python3
"""
Merge per-reference reid_score_maps into one averaged file per observation.

Each pos_NNN/reid_score_map/person_NNN/ has subdirs ref_00..ref_07, each
containing 000.npy..NNN.npy.  This script averages across all ref views and
writes pos_NNN/reid_score_map/person_NNN/000.npy (etc.) in-place.

Training code (activate_reid_dataset.py) expects the merged files.

Usage:
    python script/03_merge_reid_scoremap.py --dataset_dir outputs_parallel-v2-raycasted
    python script/03_merge_reid_scoremap.py --dataset_dir outputs_parallel-v2-raycasted --force
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def merge_scene(scene_dir: Path, force: bool) -> int:
    merged = 0
    for reid_dir in sorted(scene_dir.glob("pos_*/reid_score_map/person_*")):
        ref_dirs = sorted(d for d in reid_dir.iterdir() if d.is_dir() and d.name.startswith("ref_"))
        if not ref_dirs:
            continue
        # Collect observation indices from first ref dir
        obs_files = sorted(ref_dirs[0].glob("*.npy"))
        for obs_file in obs_files:
            out_path = reid_dir / obs_file.name
            if out_path.exists() and not force:
                continue
            maps = []
            for ref_dir in ref_dirs:
                p = ref_dir / obs_file.name
                if p.exists():
                    maps.append(np.load(str(p)).astype(np.float32))
            if not maps:
                continue
            merged_map = np.mean(maps, axis=0)
            np.save(str(out_path), merged_map)
            merged += 1
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--force", action="store_true", help="overwrite existing merged files")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    scene_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    print(f"Processing {len(scene_dirs)} scenes in {dataset_dir}")

    total = 0
    for scene_dir in scene_dirs:
        n = merge_scene(scene_dir, args.force)
        if n:
            print(f"  {scene_dir.name}: merged {n} maps")
        total += n

    print(f"Done. Total merged: {total}")


if __name__ == "__main__":
    main()
