"""ActiveHumanObservation inference helper for captured Isaac Sim frames."""
from __future__ import annotations

import json
import importlib.util
import struct
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image


_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
_DEPTH_MAX_M = 15.0


def _load_safetensors_fallback(path: Path, torch, device) -> dict:
    path = Path(path)
    dtype_map = {
        "F64": np.float64,
        "F32": np.float32,
        "F16": np.float16,
        "I64": np.int64,
        "I32": np.int32,
        "I16": np.int16,
        "I8": np.int8,
        "U8": np.uint8,
        "BOOL": np.bool_,
    }

    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data = f.read()

    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype_name = info["dtype"]
        shape = tuple(int(v) for v in info["shape"])
        start, end = [int(v) for v in info["data_offsets"]]

        if dtype_name == "BF16":
            raw = np.frombuffer(data[start:end], dtype=np.uint16).copy()
            tensor = torch.from_numpy(raw).view(torch.bfloat16).reshape(shape)
        else:
            if dtype_name not in dtype_map:
                raise ValueError(f"Unsupported safetensors dtype without safetensors package: {dtype_name}")
            arr = np.frombuffer(data[start:end], dtype=dtype_map[dtype_name]).copy().reshape(shape)
            tensor = torch.from_numpy(arr)
        tensors[name] = tensor.to(device)

    return tensors


def _load_checkpoint(path: Path, torch, device) -> dict:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors

            return load_safetensors(str(path), device=str(device))
        except ModuleNotFoundError:
            print("[AHO] safetensors package not found; using built-in safetensors reader")
            return _load_safetensors_fallback(path, torch=torch, device=device)

    return torch.load(str(path), map_location=device)


def _load_active_human_observation_class(aho_root: Path):
    aho_pkg_path = aho_root / "aho"
    models_pkg_path = aho_pkg_path / "models"

    if "aho" not in sys.modules:
        aho_pkg = types.ModuleType("aho")
        aho_pkg.__path__ = [str(aho_pkg_path)]
        sys.modules["aho"] = aho_pkg

    if "aho.models" not in sys.modules:
        models_pkg = types.ModuleType("aho.models")
        models_pkg.__path__ = [str(models_pkg_path)]
        sys.modules["aho.models"] = models_pkg

    module_name = "aho.models.aho_model"
    if module_name in sys.modules:
        return sys.modules[module_name].ActiveHumanObservation

    spec = importlib.util.spec_from_file_location(module_name, models_pkg_path / "aho_model.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load AHO model module from {models_pkg_path / 'aho_model.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.ActiveHumanObservation


class AHOInferenceRunner:
    def __init__(self, config: dict):
        import torch

        aho_root = Path(config["project_root"]).expanduser().resolve()
        if str(aho_root) not in sys.path:
            sys.path.insert(0, str(aho_root))

        ActiveHumanObservation = _load_active_human_observation_class(aho_root)

        device_name = str(config.get("device", "auto"))
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.torch = torch

        self.image_width = int(config.get("image_width", 384))
        self.image_height = int(config.get("image_height", 288))

        backbone_variant = str(config.get("backbone_variant", "large"))
        pretrained_dformer = Path(config["pretrained_dformer"]).expanduser().resolve()
        checkpoint = Path(config["checkpoint"]).expanduser().resolve()

        model_config = {
            "pretrained_dformer": str(pretrained_dformer),
            "backbone_variant": backbone_variant,
            "image_size": (self.image_height, self.image_width),
            "flow_hidden": int(config.get("flow_hidden", 32)),
            "v_dim": int(config.get("v_dim", 256)),
            "roi_size": int(config.get("roi_size", 7)),
            "roi_level": int(config.get("roi_level", 2)),
            "score_map_loss_weight": 1.0,
            "zero_loss_weight": 0.5,
            "quality_loss_weight": 1.0,
        }

        self.model = ActiveHumanObservation(model_config).to(self.device)
        self.model.eval()

        state_dict = _load_checkpoint(checkpoint, torch=torch, device=self.device)
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(
            "[AHO] loaded model "
            f"checkpoint={checkpoint}, device={self.device}, "
            f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
        )

    def _rgb_tensor(self, rgb: np.ndarray):
        arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        image = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="RGB")
        image = image.resize((self.image_width, self.image_height), _BILINEAR)
        rgb_np = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_np = (rgb_np - mean) / std
        rgb_np = np.transpose(rgb_np, (2, 0, 1))
        return self.torch.from_numpy(rgb_np).unsqueeze(0).to(self.device)

    def _depth_tensor(self, depth_m: np.ndarray):
        depth_arr = np.asarray(depth_m, dtype=np.float32)
        valid = np.isfinite(depth_arr) & (depth_arr > 0.0) & (depth_arr < _DEPTH_MAX_M)
        depth_norm = np.zeros_like(depth_arr, dtype=np.float32)
        if valid.any():
            d_min = float(depth_arr[valid].min())
            d_max = float(depth_arr[valid].max())
            depth_norm[valid] = (depth_arr[valid] - d_min) / (d_max - d_min + 1e-6)
        depth_norm = (depth_norm - 0.48) / 0.28
        image = Image.fromarray(depth_norm, mode="F")
        image = image.resize((self.image_width, self.image_height), _BILINEAR)
        resized = np.asarray(image, dtype=np.float32)
        return self.torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(self.device)

    def predict(self, rgb: np.ndarray, depth_m: np.ndarray, bbox_norm: list[float] | tuple[float, ...]) -> dict:
        bbox_np = np.asarray(bbox_norm, dtype=np.float32)
        if bbox_np.shape != (4,):
            raise ValueError(f"Expected bbox_norm shape (4,), got {bbox_np.shape}")
        bbox_np = np.clip(bbox_np, 0.0, 1.0)

        rgb_tensor = self._rgb_tensor(rgb)
        depth_tensor = self._depth_tensor(depth_m)
        bbox_tensor = self.torch.from_numpy(bbox_np).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            pred = self.model.sample(rgb_tensor, depth_tensor, bbox_tensor)

        score_map = pred["score_map"][0, 0].detach().cpu().float().numpy()
        quality = float(pred["quality"][0].detach().cpu().float().item())
        max_row, max_col = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
        return {
            "score_map": score_map,
            "quality": quality,
            "max_score": float(score_map[max_row, max_col]),
            "max_pixel": [int(max_col), int(max_row)],
            "image_size": [self.image_height, self.image_width],
        }
