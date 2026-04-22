"""从 Replicator semantic_segmentation annotators 读 person 像素数。"""
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
