#!/usr/bin/env python3
"""Run a short real-dataset loader, cache, transform and collate smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import load_config
from symm_template_reg.datasets import (
    FragmentTemplateRegistrationDataset,
    packed_collate,
    padded_collate,
)
from symm_template_reg.registry import DATASETS, build_from_cfg


def _find_dataset_cfg(config: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("dataset", "test_dataset", "train_dataset", "dataset_cfg"):
        value = config.get(key)
        if isinstance(value, Mapping) and "type" in value:
            return dict(value)
    data = config.get("data")
    if isinstance(data, Mapping):
        for key in ("test", "val", "train"):
            value = data.get(key)
            if isinstance(value, Mapping) and "type" in value:
                return dict(value)
    for key in ("test_dataloader", "val_dataloader", "train_dataloader"):
        loader = config.get(key)
        if isinstance(loader, Mapping):
            value = loader.get("dataset")
            if isinstance(value, Mapping) and "type" in value:
                return dict(value)
    raise KeyError("config has no dataset config with a registered 'type'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--num-samples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = _find_dataset_cfg(config)
    if args.dataset_root is not None:
        for key in ("dataset_root", "root", "data_root"):
            dataset_cfg.pop(key, None)
        dataset_cfg["dataset_root"] = str(args.dataset_root)
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    if not isinstance(dataset, FragmentTemplateRegistrationDataset):
        raise TypeError(f"expected FragmentTemplateRegistrationDataset, got {type(dataset).__name__}")
    count = min(args.num_samples, len(dataset))
    samples = [dataset[index] for index in range(count)]
    if count < 2:
        raise RuntimeError("smoke requires at least two samples")
    lengths = [len(sample["observed"]["points_C"]) for sample in samples]
    if len(set(lengths)) < 2:
        second = next(
            dataset[index]
            for index in range(count, len(dataset))
            if len(dataset[index]["observed"]["points_C"]) != lengths[0]
        )
        samples[-1] = second
        lengths[-1] = len(second["observed"]["points_C"])
    max_error = 0.0
    for sample in samples:
        points_O = sample["gt"]["points_O_corresponding"]
        if points_O is not None:
            transform = sample["gt"]["T_C_from_O"]
            points_C = points_O @ transform[:3, :3].T + transform[:3, 3]
            max_error = max(
                max_error,
                float((points_C - sample["observed"]["points_C"]).abs().max()),
            )
    if max_error > 1e-5:
        raise RuntimeError(
            f"GT transform round-trip error {max_error:.6g} m exceeds 1e-5 m"
        )
    packed = packed_collate(samples)
    padded = padded_collate(samples)
    packed["observed"].validate()
    packed["template"].validate()
    if packed["observed"].lengths.tolist() != lengths:
        raise RuntimeError("packed lengths disagree with individual samples")
    if padded["observed"]["valid_mask"].sum(1).tolist() != lengths:
        raise RuntimeError("padded valid_mask disagrees with individual samples")
    cache_counts = {
        model_id: dataset.template_repository.load_count(model_id)
        for model_id in sorted({sample["object_model_id"] for sample in samples})
    }
    bad_cache_counts = {key: value for key, value in cache_counts.items() if value != 1}
    if bad_cache_counts:
        raise RuntimeError(f"templates were not loaded exactly once: {bad_cache_counts}")
    summary = {
        "status": "ok",
        "dataset_samples_after_min_filter": len(dataset),
        "skipped_below_min_observed_points": dataset.skipped_too_small,
        "smoke_samples": count,
        "observed_lengths": lengths,
        "variable_N_confirmed": len(set(lengths)) >= 2,
        "packed_observed_shape": list(packed["observed"].points.shape),
        "packed_offsets": packed["observed"].offsets.tolist(),
        "padded_observed_shape": list(padded["observed"]["points_C"].shape),
        "padded_valid_counts": padded["observed"]["valid_mask"].sum(1).tolist(),
        "template_cache_load_count": next(iter(cache_counts.values())) if len(cache_counts) == 1 else None,
        "template_cache_load_counts": cache_counts,
        "symmetry_available": [sample["meta"]["symmetry_available"] for sample in samples],
        "T_C_from_O_roundtrip_max_abs_error_m": max_error,
        "torch_version": torch.__version__,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
