"""Shared deterministic loading for ten-view scratch audits."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from symm_template_reg.config import load_config
from symm_template_reg.engine.manifest import load_and_validate_manifest
from symm_template_reg.engine.seed import seed_everything
from symm_template_reg.models import build_model, register_all_modules
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg


EXPECTED_FRAMES = tuple(range(10))


def load_scratch_context(config_path, manifest_path, output_dir, device):
    config = load_config(config_path)
    if str(config.get("initialization_mode")) != "scratch":
        raise ValueError("ten-view scratch audit requires initialization_mode=scratch")
    if config.get("pretrained_checkpoint") is not None:
        raise ValueError("scratch audit forbids pretrained checkpoints")
    if bool(config.get("frozen_feature_cache", {}).get("enabled", False)):
        raise ValueError("scratch audit forbids frozen-feature cache")
    seed = int(config.get("seed", 0))
    if torch.device(device).type == "cuda" and hasattr(
        torch.backends.cuda, "enable_flash_sdp"
    ):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    seed_everything(seed)
    register_all_modules()
    dataset_cfg = deepcopy(config["dataset"])
    data_cfg = config["data"]
    dataset_cfg["fragment_mesh_filter"] = deepcopy(data_cfg["fragment_mesh_filter"])
    dataset_cfg["observed_filter"] = deepcopy(data_cfg["observed_filter"])
    dataset_cfg["symmetry_region_activity"] = deepcopy(
        data_cfg.get("symmetry_region_activity", {})
    )
    dataset_cfg["fragment_mesh_cache_dir"] = str(
        Path(output_dir) / "cache" / "fragment_mesh_metadata"
    )
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    manifest, _ = load_and_validate_manifest(
        str(Path(manifest_path).expanduser().resolve()), config, dataset
    )
    frames = tuple(int(row["frame_id"]) for row in manifest["samples"])
    if frames != EXPECTED_FRAMES:
        raise ValueError(f"expected frames 0..9 in order, got {frames}")
    indices = {record.sample_id: i for i, record in enumerate(dataset.sample_records)}
    samples = [dataset[indices[str(row["sample_id"])]] for row in manifest["samples"]]
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    model = build_model(config["model"]).to(device)
    return config, manifest, dataset, samples, collate, model


__all__ = ["EXPECTED_FRAMES", "load_scratch_context"]
