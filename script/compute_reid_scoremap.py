#!/usr/bin/env python3
"""
Compute per-reference ReID score maps from score_field_views.

For each pos_XXX / person_YYY:
  1. Load 8 reference crops  →  extract features F_ref[0..7]
  2. Batch-extract features from all score_field_views candidates (bbox crop)
  3. For each ref_j: cosine_sim(F_cand, F_ref[j])  →  score_by_xy dict
  4. _compute_score_map  →  reid_score_map (same coordinate system as raycast score_map)
  5. Save to  pos_XXX/reid_score_map/person_YYY/ref_XX/NNN.npy

Sanity check: reid_score_map should have nonzero values at the same pixels as
the existing raycast score_map (same valid_mask).

Usage:
    python script/compute_reid_scoremap.py \
        --dataset_dir outputs_parallel-v2-raycasted \
        --reference_dir outputs_parallel-v2-raycasted/reference \
        [--reid_model osnet_ain_x1_0] \
        [--batch_size 64] \
        [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# Project root on sys.path so utils can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.capture_outputs import _compute_score_map

_SSL_REPO = Path("/home/leo/FusionLab/ReID/TransReID-SSL/transreid_pytorch")
sys.path.insert(0, str(_SSL_REPO))
from model.backbones.vit_pytorch import vit_base_patch16_224_TransReID

_REID_WEIGHT = Path("/home/leo/FusionLab/ReID/weights/transformer_120.pth")
_IMG_SIZE = (384, 128)

# person_id → reference character folder name (matches run_collector config order)
PERSON_TO_CHAR = {
    "person_000": "F_Business_02",
    "person_001": "male_adult_medical_01",
}
NUM_REFS = 1
REF_IDX  = 6   # use only ref_06.png


# ── ReID feature extractor ────────────────────────────────────────────────────

class ReIDScorer:
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        model = vit_base_patch16_224_TransReID(
            img_size=_IMG_SIZE, stride_size=16, camera=0, view=0,
            local_feature=False, sie_xishu=1.5, drop_path_rate=0.1,
            stem_conv=True,
        )
        ckpt = torch.load(_REID_WEIGHT, map_location="cpu")
        backbone_state = {k[len("base."):]: v for k, v in ckpt.items() if k.startswith("base.")}
        model.load_state_dict(backbone_state, strict=False)
        self.model = model.to(self.device).eval()

        self.transform = T.Compose([
            T.Resize(_IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def extract_batch(self, imgs: list[Image.Image]) -> torch.Tensor:
        """Returns (N, D) normalized feature tensor."""
        tensors = torch.stack([self.transform(img) for img in imgs]).to(self.device)
        with torch.no_grad():
            feats = self.model(tensors, cam_label=None, view_label=None)
        return F.normalize(feats, dim=1)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_ref_features(scorer: ReIDScorer, ref_dir: Path) -> Optional[torch.Tensor]:
    """Load ref_00.png..ref_07.png, return (8, D) normalized tensor or None."""
    p = ref_dir / f"ref_{REF_IDX:02d}.png"
    if not p.exists():
        print(f"  [WARN] missing reference image: {p}")
        return None
    return scorer.extract_batch([Image.open(p).convert("RGB")])   # (1, D)


def crop_bbox(rgb_path: str, bbox_path: str) -> Optional[Image.Image]:
    bbox_data = json.loads(Path(bbox_path).read_text())
    xyxy = bbox_data.get("xyxy_norm")
    if not xyxy:
        return None
    img = Image.open(rgb_path).convert("RGB")
    w, h = img.size
    x1 = max(0, min(w, int(round(xyxy[0] * w))))
    y1 = max(0, min(h, int(round(xyxy[1] * h))))
    x2 = max(0, min(w, int(round(xyxy[2] * w))))
    y2 = max(0, min(h, int(round(xyxy[3] * h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((x1, y1, x2, y2))


@dataclass
class Candidate:
    key: tuple[float, float]  # (x, y) candidate_key
    x: float
    y: float
    z: float
    yaw_rad: float
    feat: Optional[torch.Tensor]  # (1, D) or None if no valid bbox


def load_candidates(scorer: ReIDScorer,
                    sfv_dir: Path,
                    batch_size: int = 64) -> list[Candidate]:
    """
    Load all candidates from score_field_views/person_YYY/,
    extract ReID features in batches for valid-bbox candidates.
    """
    meta_files = sorted((sfv_dir / "metadata").iterdir())

    candidates: list[Candidate] = []
    valid_indices: list[int] = []
    valid_crops: list[Image.Image] = []

    for mf in meta_files:
        m = json.loads(mf.read_text())
        cand = Candidate(
            key=tuple(m["candidate_key"]),
            x=float(m["x"]),
            y=float(m["y"]),
            z=float(m["z"]),
            yaw_rad=float(m["yaw_rad"]),
            feat=None,
        )
        candidates.append(cand)
        if m.get("bbox_valid") and m.get("rgb_path") and m.get("bbox_path"):
            crop = crop_bbox(m["rgb_path"], m["bbox_path"])
            if crop is not None:
                valid_indices.append(len(candidates) - 1)
                valid_crops.append(crop)

    if valid_crops:
        all_feats = []
        for bs in range(0, len(valid_crops), batch_size):
            all_feats.append(scorer.extract_batch(valid_crops[bs:bs + batch_size]))
        feats = torch.cat(all_feats, dim=0)   # (N_valid, D)
        for feat_i, cand_i in enumerate(valid_indices):
            candidates[cand_i].feat = feats[feat_i:feat_i + 1]  # (1, D)

    return candidates


# ── Score map proxy ───────────────────────────────────────────────────────────

@dataclass
class _ScoreItem:
    """Thin proxy matching the attribute API expected by _compute_score_map."""
    x: float
    y: float
    z: float
    score: float
    yaw_rad: float


def load_ground_mask(path: str) -> np.ndarray:
    """Load ground_mask PNG (0/255) and convert to 0/1 uint8 as _compute_score_map expects."""
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 0).astype(np.uint8)


# ── Per-pos processing ────────────────────────────────────────────────────────

def process_pos(pos_dir: Path,
                reference_dir: Path,
                scorer: ReIDScorer,
                batch_size: int,
                force: bool) -> None:

    for person_id, char_name in PERSON_TO_CHAR.items():
        sfv_dir = pos_dir / "score_field_views" / person_id
        meta_path = pos_dir / "metadata" / f"{person_id}.json"

        if not sfv_dir.exists() or not meta_path.exists():
            continue

        # Check if already done
        out_base = pos_dir / "reid_score_map" / person_id
        if not force and out_base.exists():
            existing = list(out_base.glob("ref_*/*.npy"))
            if existing:
                print(f"    [SKIP] {person_id}: reid_score_map already exists")
                continue

        # ── 1. Reference features ─────────────────────────────────────────
        ref_dir = reference_dir / char_name
        ref_feats = load_ref_features(scorer, ref_dir)   # (8, D) or None
        if ref_feats is None:
            print(f"    [SKIP] {person_id}: reference images missing in {ref_dir}")
            continue

        # ── 2. Candidate features ─────────────────────────────────────────
        print(f"    {person_id}: extracting candidate features...", end=" ", flush=True)
        candidates = load_candidates(scorer, sfv_dir, batch_size=batch_size)
        n_valid = sum(1 for c in candidates if c.feat is not None)
        print(f"{len(candidates)} candidates, {n_valid} with valid bbox")

        # ── 3. Load observation metadata ──────────────────────────────────
        meta = json.loads(meta_path.read_text())
        camera = meta["camera"]
        focal_length      = float(camera["focal_length"])
        h_aperture        = float(camera["horizontal_aperture"])
        v_aperture        = float(camera["vertical_aperture"])
        resolution        = tuple(camera["resolution"])   # (W, H)
        observations      = meta["observations"]

        # ── 4. For each reference, build score_maps ───────────────────────
        # Use only REF_IDX; NUM_REFS=1 so ref_j=0 always.
        ref_feat = ref_feats[0:1]   # (1, D)

        # Cosine similarity for every candidate; build key→score lookup
        score_field: list[_ScoreItem] = []
        key_to_reid: dict[tuple, float] = {}
        for cand in candidates:
            if cand.feat is not None:
                score = float((cand.feat @ ref_feat.T).squeeze())
                score = max(0.0, score)
            else:
                score = 0.0
            score_field.append(_ScoreItem(
                x=cand.x, y=cand.y, z=cand.z,
                score=score, yaw_rad=cand.yaw_rad,
            ))
            key_to_reid[cand.key] = score

        # One score_map per observation frame
        for obs in observations:
            stem = f"{obs['idx']:03d}"
            out_path = out_base / f"ref_{REF_IDX:02d}" / f"{stem}.npy"
            if not force and out_path.exists():
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)

            depth       = np.load(obs["depth_npy_path"]).astype(np.float32)
            ground_mask = load_ground_mask(obs["ground_mask_path"])

            score_map, valid_mask, _ = _compute_score_map(
                score_field=score_field,
                depth_m=depth,
                ground_mask=ground_mask,
                camera_position=tuple(obs["camera_position"]),
                camera_orientation_wxyz=tuple(obs["camera_orientation_wxyz"]),
                resolution=resolution,
                focal_length=focal_length,
                horizontal_aperture=h_aperture,
                vertical_aperture=v_aperture,
            )
            np.save(str(out_path), score_map)

        print(f"    {person_id}: saved reid_score_map/ref_{REF_IDX:02d}/")

        # ── 5. Write back reid scores ─────────────────────────────────────
        reid_scores_dict: dict[str, float] = {}
        for obs in observations:
            key = tuple(obs["candidate_key"])
            reid = key_to_reid.get(key, 0.0)
            obs["reid_score"] = reid
            reid_scores_dict[f"{obs['idx']:03d}.png"] = reid

        # Save reid_scores/person_XXX.json
        reid_scores_dir = pos_dir / "reid_scores"
        reid_scores_dir.mkdir(exist_ok=True)
        reid_scores_path = reid_scores_dir / f"{person_id}.json"
        reid_scores_path.write_text(json.dumps(reid_scores_dict, indent=2))

        # Write back metadata
        meta_path.write_text(json.dumps(meta, indent=2))

        # ── 6. Sanity check: compare coverage with raycast score_map ──────
        _sanity_check(pos_dir, person_id, observations)


def _sanity_check(pos_dir: Path, person_id: str, observations: list) -> None:
    """Print IoU of nonzero pixels between reid_score_map and raycast score_map."""
    ref0_maps = sorted((pos_dir / "reid_score_map" / person_id / f"ref_{REF_IDX:02d}").glob("*.npy"))
    if not ref0_maps:
        return
    ious = []
    for obs in observations:
        stem = f"{obs['idx']:03d}"
        raycast_path = Path(obs["score_map_path"])
        reid_path    = pos_dir / "reid_score_map" / person_id / f"ref_{REF_IDX:02d}" / f"{stem}.npy"
        if not raycast_path.exists() or not reid_path.exists():
            continue
        raycast = np.load(str(raycast_path)) > 0
        reid    = np.load(str(reid_path))    > 0
        inter   = int((raycast & reid).sum())
        union   = int((raycast | reid).sum())
        ious.append(inter / union if union > 0 else 1.0)
    if ious:
        print(f"    {person_id}: coverage IoU with raycast score_map: "
              f"mean={np.mean(ious):.3f}  min={np.min(ious):.3f}  max={np.max(ious):.3f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str,
                   default="outputs_parallel-v2-raycasted")
    p.add_argument("--reference_dir", type=str,
                   default="outputs_parallel-v2-raycasted/reference")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--force", action="store_true")
    p.add_argument("--scene_id", type=str, default=None,
                   help="Process only this scene (e.g. 839873)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir   = Path(args.dataset_dir)
    reference_dir = Path(args.reference_dir)

    print(f"Loading ReID model: TransReID-SSL ViT-B/16 ({_REID_WEIGHT.name})")
    scorer = ReIDScorer(device=args.device)

    if args.scene_id:
        scene_dirs = [dataset_dir / args.scene_id]
    else:
        scene_dirs = sorted(
            p for p in dataset_dir.iterdir()
            if p.is_dir() and p.name not in {"reference"}
        )

    for scene_dir in scene_dirs:
        pos_dirs = sorted(p for p in scene_dir.iterdir()
                          if p.is_dir() and p.name.startswith("pos_"))
        if not pos_dirs:
            continue
        print(f"\nScene {scene_dir.name}: {len(pos_dirs)} positions")
        for pos_dir in pos_dirs:
            print(f"  {pos_dir.name}")
            process_pos(pos_dir, reference_dir, scorer,
                        batch_size=args.batch_size, force=args.force)


if __name__ == "__main__":
    main()
