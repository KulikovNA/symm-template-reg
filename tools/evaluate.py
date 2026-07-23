#!/usr/bin/env python3
"""Evaluate one production checkpoint explicitly on val or test."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import apply_overrides, load_config  # noqa: E402
from symm_template_reg.engine.production_evaluator import (  # noqa: E402
    evaluate_production,
    write_evaluation_report,
)
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.registry import (  # noqa: E402
    COLLATE_FUNCTIONS,
    DATASETS,
    build_from_cfg,
)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cfg-options", nargs="*")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.cfg_options)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if config.get("runtime") != "production_evaluation":
        raise ValueError("tools/evaluate.py accepts the production eval config only")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    register_all_modules()
    dataset_cfg = deepcopy(config["data"]["validation"])
    dataset_cfg.update(
        dataset_root=args.dataset_root,
        split=args.split,
        index_cache_dir=str(output / "dataset_index"),
        boundary_augmentation={"enabled": False},
    )
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    workers = int(config["data"].get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"].get("validation_batch_size", 2)),
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(config["data"].get("pin_memory", True)),
        persistent_workers=bool(
            config["data"].get("persistent_workers", True) and workers > 0
        ),
        collate_fn=collate,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    model = build_model(config["model"]).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model", payload), strict=True)
    summary, rows = evaluate_production(
        model,
        loader,
        device,
        source_dataset_split=args.split,
        evaluation_role=(
            "test_evaluation" if args.split == "test" else "validation"
        ),
        max_batches=args.max_batches,
        candidate_k=int(config.get("evaluation", {}).get("candidate_k", 16)),
        projection_chunk_size=int(
            config.get("evaluation", {}).get("projection_chunk_size", 64)
        ),
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output),
        "dataset_index_fingerprint": dataset.index_fingerprint,
        "test_results_must_not_be_used_for_model_selection": args.split == "test",
        **summary,
    }
    write_evaluation_report(output, report, rows)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
