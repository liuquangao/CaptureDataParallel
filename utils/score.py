"""从 Replicator segmentation annotators 读目标像素数。"""
from __future__ import annotations

import numpy as np


def read_person_counts(annotators) -> list[int]:
    """对每个 annotator 取一张 seg mask,返回对应视图里 person 的像素数。

    annotator 的 `get_data()` 结果里:
    - `data`: HxW uint32,每个像素是 class id
    - `info["idToLabels"]`: {"0": {"class": "..."}, ...}
    """
    counts: list[int] = []
    for ann in annotators:
        data = ann.get_data()
        seg = data["data"]
        id_to_labels = data["info"]["idToLabels"]
        person_ids = [int(k) for k, v in id_to_labels.items() if v.get("class") == "person"]
        counts.append(int(np.isin(seg, person_ids).sum()) if person_ids else 0)
    return counts


def read_instance_counts(annotators, prim_path_prefix: str) -> list[int]:
    """对每个 instance_id_segmentation annotator 统计目标 prim 前缀像素数。"""
    counts: list[int] = []
    prefix = str(prim_path_prefix).rstrip("/")
    for ann in annotators:
        data = ann.get_data()
        seg = np.asarray(data["data"])
        if seg.ndim == 3 and seg.shape[-1] == 1:
            seg = seg[..., 0]
        id_to_labels = data.get("info", {}).get("idToLabels", {})
        target_ids = []
        for raw_id, label_info in id_to_labels.items():
            text = str(label_info)
            if text == prefix or text.startswith(prefix + "/"):
                target_ids.append(int(raw_id))
        counts.append(int(np.isin(seg, target_ids).sum()) if target_ids else 0)
    return counts


def read_semantic_counts(annotators, class_name: str) -> list[int]:
    """对每个 semantic_segmentation annotator 统计指定 class 像素数。"""
    counts: list[int] = []
    target_class = str(class_name)
    for ann in annotators:
        data = ann.get_data()
        seg = np.asarray(data["data"])
        if seg.ndim == 3 and seg.shape[-1] == 1:
            seg = seg[..., 0]
        id_to_labels = data.get("info", {}).get("idToLabels", {})
        target_ids = [
            int(raw_id)
            for raw_id, label_info in id_to_labels.items()
            if isinstance(label_info, dict) and label_info.get("class") == target_class
        ]
        counts.append(int(np.isin(seg, target_ids).sum()) if target_ids else 0)
    return counts
