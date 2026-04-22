"""YAML 配置加载
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file root must be a mapping: {path}")

    return data
