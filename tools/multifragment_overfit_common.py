"""Shared deterministic loading for the four-fragment/four-view experiment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from symm_template_reg.config import load_config
from symm_template_reg.engine.multifragment_overfit import (
    EXPECTED_FRAGMENTS, EXPECTED_FRAMES, validate_multifragment_config,
)
from symm_template_reg.engine.overfit_manifest import load_faces840_manifest
from symm_template_reg.engine.seed import seed_everything
from symm_template_reg.models import build_model, register_all_modules
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg


def load_multifragment_context(config_path, manifest_path, output_dir, device="cpu"):
    config = load_config(config_path)
    validate_multifragment_config(config)
    seed_everything(int(config.get("seed", 0)))
    register_all_modules()
    dataset_cfg = deepcopy(config["dataset"])
    data_cfg = config["data"]
    dataset_cfg["fragment_mesh_filter"] = deepcopy(data_cfg["fragment_mesh_filter"])
    dataset_cfg["observed_filter"] = deepcopy(data_cfg["observed_filter"])
    dataset_cfg["symmetry_region_activity"] = deepcopy(data_cfg.get("symmetry_region_activity", {}))
    dataset_cfg["fragment_mesh_cache_dir"] = str(Path(output_dir) / "cache" / "fragment_mesh_metadata")
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    manifest, file_sha = load_faces840_manifest(manifest_path, config, dataset)
    identities = [(int(row["fragment_id"]), int(row["frame_id"])) for row in manifest["samples"]]
    expected = [(fragment, frame) for fragment in EXPECTED_FRAGMENTS for frame in EXPECTED_FRAMES]
    if identities != expected:
        raise ValueError(f"manifest sample order must be fragment-major 4x4, got {identities}")
    indices = {record.sample_id: index for index, record in enumerate(dataset.sample_records)}
    samples = [dataset[indices[str(row["sample_id"])]] for row in manifest["samples"]]
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    model = build_model(config["model"]).to(torch.device(device))
    runtime_static_cache = getattr(model, "_static_geometry_cache", None)
    if runtime_static_cache is not None:
        runtime_static_cache.manifest_sha256 = str(
            manifest.get("manifest_sha256", file_sha)
        )
    return config, manifest, file_sha, dataset, samples, collate, model


__all__ = ["load_multifragment_context"]
