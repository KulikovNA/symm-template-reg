"""Shared helpers for correspondence diagnostics (not a command-line tool)."""

from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch

from symm_template_reg.config import load_config
from symm_template_reg.engine.evaluator import move_to_device
from symm_template_reg.models import register_all_modules
from symm_template_reg.models.geometry.point_ops import farthest_point_indices
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg


def statistics_mm(distance_m: torch.Tensor) -> dict[str, float]:
    value = torch.as_tensor(distance_m).float().flatten() * 1000.0
    return {
        "p50_mm": float(torch.quantile(value, .50)),
        "p95_mm": float(torch.quantile(value, .95)),
        "max_mm": float(value.max()),
        "fraction_within_1mm": float((value <= 1).float().mean()),
        "fraction_within_2mm": float((value <= 2).float().mean()),
        "fraction_within_5mm": float((value <= 5).float().mean()),
    }


def build_dataset(config_path: str | Path, output_dir: str | Path, *, shell_only: bool = True):
    config = load_config(config_path)
    register_all_modules()
    cfg = copy.deepcopy(config["dataset"])
    cfg["fragment_mesh_filter"] = copy.deepcopy(config["data"]["fragment_mesh_filter"])
    cfg["observed_filter"] = copy.deepcopy(config["data"]["observed_filter"])
    cfg["symmetry_region_activity"] = copy.deepcopy(config["data"].get("symmetry_region_activity", {}))
    cfg["fragment_mesh_cache_dir"] = str(Path(output_dir) / "cache")
    cfg["registration_point_selection"] = (
        "shell_only" if shell_only else "all_fragment_points"
    )
    return config, build_from_cfg(cfg, DATASETS)


def manifest_samples(dataset: Any, manifest_path: str | Path, frames=(4, 8)):
    manifest = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
    records = {record.sample_id: i for i, record in enumerate(dataset.sample_records)}
    selected = []
    for entry in manifest["samples"]:
        if int(entry["frame_id"]) in set(frames):
            selected.append(dataset[records[str(entry["sample_id"])]])
    if {int(sample["frame_id"]) for sample in selected} != set(frames):
        raise ValueError(f"manifest does not contain requested frames {frames}")
    return manifest, sorted(selected, key=lambda sample: int(sample["frame_id"]))


def actual_template_anchors(sample: dict[str, Any], count: int = 512) -> torch.Tensor:
    points = sample["template"].get("fine_points_O", sample["template"]["points_O"])
    mask = torch.ones((1, len(points)), dtype=torch.bool)
    indices, selected = farthest_point_indices(points.unsqueeze(0), mask, count)
    return points[indices[0, selected[0]]]


def nearest_sample_distance(query: torch.Tensor, support: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    values = []
    for start in range(0, len(query), chunk):
        values.append(torch.cdist(query[start:start+chunk].float(), support.float()).amin(-1))
    return torch.cat(values)


def tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def collated(config: dict, samples: list[dict[str, Any]], device: torch.device):
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    return move_to_device(collate(samples), device), collate


__all__ = ["actual_template_anchors", "build_dataset", "collated", "manifest_samples", "nearest_sample_distance", "statistics_mm", "tensor_sha256"]
