#!/usr/bin/env python3
"""
Compute ReID similarity before/after for AHO active eval outputs.

For each scene/pos/view:
  - Crop person from before/rgb.png using before/bbox_person_XXX.json
  - Crop person from after_person_XXX/rgb.png using after_person_XXX/bbox_person_XXX.json
  - Compute cosine similarity with reference ref_06.png
  - Report delta = after_reid - before_reid

Usage:
    python script/compute_reid_eval.py \
        --eval_dir outputs_aho_active_eval_isaac_two_people \
        --reference_dir outputs_parallel-v2-raycasted/reference \
        [--ref_idx 6] \
        [--reid_model osnet_ain_x1_0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

_SSL_REPO = Path("/home/leo/FusionLab/ReID/TransReID-SSL/transreid_pytorch")
sys.path.insert(0, str(_SSL_REPO))
from model.backbones.vit_pytorch import vit_base_patch16_224_TransReID

_REID_WEIGHT = Path("/home/leo/FusionLab/ReID/weights/transformer_120.pth")
_IMG_SIZE = (384, 128)

PERSON_TO_CHAR = {
    "person_000": "F_Business_02",
    "person_001": "male_adult_medical_01",
}


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

    def extract(self, imgs: list[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([self.transform(img) for img in imgs]).to(self.device)
        with torch.no_grad():
            feats = self.model(tensors, cam_label=None, view_label=None)
        return F.normalize(feats, dim=1)


def crop_bbox(rgb_path: Path, bbox_path: Path) -> Image.Image | None:
    if not bbox_path.exists() or not rgb_path.exists():
        return None
    data = json.loads(bbox_path.read_text())
    xyxy = data.get("xyxy_norm")
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


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b.T).squeeze())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str,
                        default="outputs_aho_active_eval_isaac_two_people")
    parser.add_argument("--reference_dir", type=str,
                        default="outputs_parallel-v2-raycasted/reference")
    parser.add_argument("--ref_idx", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    ref_dir = Path(args.reference_dir)

    print(f"Loading ReID model: TransReID-SSL ViT-B/16 ({_REID_WEIGHT.name})")
    scorer = ReIDScorer(device=args.device)

    # Load reference features once per person
    ref_feats: dict[str, torch.Tensor] = {}
    for person_id, char_name in PERSON_TO_CHAR.items():
        ref_path = ref_dir / char_name / f"ref_{args.ref_idx:02d}.png"
        if not ref_path.exists():
            print(f"[ERROR] reference not found: {ref_path}")
            sys.exit(1)
        ref_img = Image.open(ref_path).convert("RGB")
        ref_feats[person_id] = scorer.extract([ref_img])  # (1, D)
        print(f"  Loaded reference for {person_id}: {ref_path}")

    # Collect per-person results
    results: dict[str, list[dict]] = {p: [] for p in PERSON_TO_CHAR}

    scene_dirs = sorted(p for p in eval_dir.iterdir() if p.is_dir() and p.name != "_metrics")
    for scene_dir in scene_dirs:
        pos_dirs = sorted(p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith("pos_"))
        for pos_dir in pos_dirs:
            view_dirs = sorted(p for p in pos_dir.iterdir() if p.is_dir() and p.name.startswith("view_"))
            for view_dir in view_dirs:
                for person_id in PERSON_TO_CHAR:
                    before_rgb = view_dir / "before" / "rgb.png"
                    before_bbox = view_dir / "before" / f"bbox_{person_id}.json"
                    after_rgb = view_dir / f"after_{person_id}" / "rgb.png"
                    after_bbox = view_dir / f"after_{person_id}" / f"bbox_{person_id}.json"

                    # Skip if before score >= 0.8 (already good, not informative)
                    record = json.loads((view_dir / "record.json").read_text())
                    before_score = record.get("targets", {}).get(person_id, {}).get("before_score")
                    if before_score is None or before_score >= 0.8:
                        continue

                    before_crop = crop_bbox(before_rgb, before_bbox)
                    after_crop = crop_bbox(after_rgb, after_bbox)

                    if before_crop is None or after_crop is None:
                        continue

                    feats = scorer.extract([before_crop, after_crop])
                    ref_f = ref_feats[person_id]
                    before_sim = cosine_sim(feats[0:1], ref_f)
                    after_sim = cosine_sim(feats[1:2], ref_f)

                    results[person_id].append({
                        "scene": scene_dir.name,
                        "pos": pos_dir.name,
                        "view": view_dir.name,
                        "before_reid": before_sim,
                        "after_reid": after_sim,
                        "delta": after_sim - before_sim,
                    })

    # Print summary
    print(f"\n{'='*55}")
    print(f"ReID similarity vs ref_{args.ref_idx:02d}.png")
    print(f"{'='*55}")

    all_deltas = []
    for person_id in PERSON_TO_CHAR:
        rows = results[person_id]
        if not rows:
            continue
        befores = [r["before_reid"] for r in rows]
        afters  = [r["after_reid"]  for r in rows]
        deltas  = [r["delta"]       for r in rows]
        success_rate = float(np.mean([d > 0 for d in deltas]))
        all_deltas.extend(deltas)
        print(f"\n{person_id} ({len(rows)} views):")
        print(f"  before  mean={np.mean(befores):.4f}  std={np.std(befores):.4f}")
        print(f"  after   mean={np.mean(afters):.4f}  std={np.std(afters):.4f}")
        print(f"  delta   mean={np.mean(deltas):+.4f}  std={np.std(deltas):.4f}")
        print(f"  success rate: {success_rate:.1%}")

    if all_deltas:
        print(f"\n{'─'*55}")
        print(f"Overall ({len(all_deltas)} views, both persons combined):")
        print(f"  delta   mean={np.mean(all_deltas):+.4f}  std={np.std(all_deltas):.4f}")
        print(f"  success rate: {float(np.mean([d > 0 for d in all_deltas])):.1%}")

    # Save results
    out_path = eval_dir / "_metrics" / "reid_eval.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "ref_idx": args.ref_idx,
        "by_person": {
            person_id: {
                "n": len(results[person_id]),
                "before_mean": float(np.mean([r["before_reid"] for r in results[person_id]])) if results[person_id] else None,
                "after_mean":  float(np.mean([r["after_reid"]  for r in results[person_id]])) if results[person_id] else None,
                "delta_mean":  float(np.mean([r["delta"]       for r in results[person_id]])) if results[person_id] else None,
                "delta_std":   float(np.std( [r["delta"]       for r in results[person_id]])) if results[person_id] else None,
                "success_rate": float(np.mean([r["delta"] > 0  for r in results[person_id]])) if results[person_id] else None,
            }
            for person_id in PERSON_TO_CHAR
        },
        "overall": {
            "n": len(all_deltas),
            "delta_mean": float(np.mean(all_deltas)) if all_deltas else None,
            "delta_std":  float(np.std(all_deltas))  if all_deltas else None,
            "success_rate": float(np.mean([d > 0 for d in all_deltas])) if all_deltas else None,
        },
    }, indent=2), encoding="utf-8")
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()