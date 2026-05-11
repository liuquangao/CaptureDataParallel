"""YAML 配置加载
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply 'dot.key=value' overrides onto cfg in-place. Returns cfg."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set override must be 'key=value', got: {item!r}")
        key_path, _, raw_value = item.partition("=")
        keys = key_path.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        # try to parse as YAML scalar so numbers/booleans work
        try:
            parsed = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            parsed = raw_value
        node[keys[-1]] = parsed
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file root must be a mapping: {path}")

    return data
