"""Shared helpers for command-line smoke tools."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import load_config  # noqa: E402
from symm_template_reg.models import build_loss, build_model, register_all_modules  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg  # noqa: E402


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if hasattr(value, "to") and callable(value.to):
        return value.to(device)
    return value


def resolve_device(name: str) -> torch.device | None:
    if name == "cuda" and not torch.cuda.is_available():
        return None
    return torch.device(name)


def build_real_batch(config_path: str, num_samples: int = 2):
    config = load_config(config_path)
    config["dataset"]["fragment_mesh_cache_dir"] = "/tmp/symm_template_reg_tool_cache"
    register_all_modules()
    dataset = build_from_cfg(config["dataset"], DATASETS)
    candidate_count = min(max(num_samples * 8, 16), len(dataset))
    candidates = [dataset[index] for index in range(candidate_count)]
    selected = [candidates[0]]
    selected_indices = {0}
    first_length = len(candidates[0]["observed"]["points_C"])
    for candidate_index, sample in enumerate(candidates[1:], start=1):
        if len(selected) >= num_samples:
            break
        if len(sample["observed"]["points_C"]) != first_length or len(selected) > 1:
            selected.append(sample)
            selected_indices.add(candidate_index)
    for candidate_index, sample in enumerate(candidates):
        if len(selected) >= num_samples:
            break
        if candidate_index not in selected_indices:
            selected.append(sample)
            selected_indices.add(candidate_index)
    collate_cfg = config.get("collate", {"type": "FragmentTemplateCollator", "mode": "packed"})
    collate = build_from_cfg(collate_cfg, COLLATE_FUNCTIONS)
    batch = collate(selected)
    lengths = [len(sample["observed"]["points_C"]) for sample in selected]
    if num_samples >= 2 and len(set(lengths)) < 2:
        raise RuntimeError("smoke batch did not contain two distinct observed point counts")
    return config, dataset, batch, lengths


def tensor_shapes(prediction: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in prediction.as_dict().items():
        if isinstance(value, Tensor):
            result[key] = list(value.shape)
        elif isinstance(value, list):
            result[key] = [
                {name: list(item.shape) for name, item in layer.items() if isinstance(item, Tensor)}
                for layer in value
            ]
        elif isinstance(value, dict):
            result[key] = {
                name: list(item.shape)
                for name, item in value.items()
                if isinstance(item, Tensor)
            }
        else:
            result[key] = None
    return result
