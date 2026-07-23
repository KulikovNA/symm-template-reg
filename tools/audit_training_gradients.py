#!/usr/bin/env python3
"""Run one real V2 batch and verify gradients reach every required module."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import apply_overrides, load_config  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.overfit_trainer import (  # noqa: E402
    _build_dataset,
    _build_pose_criterion,
    _loss_values,
)
from symm_template_reg.engine.trainer import resolve_device  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, build_from_cfg  # noqa: E402


def _gradient_norm(
    model: torch.nn.Module, predicate: callable
) -> tuple[float, list[str]]:
    squared = 0.0
    names = []
    for name, parameter in model.named_parameters():
        if predicate(name):
            names.append(name)
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(squared), names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output")
    parser.add_argument("--cfg-options", nargs="*")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.cfg_options)
    if float(config["loss"].get("observed_region_weight", 0.0)) <= 0.0:
        raise ValueError("observed_region_weight must be positive for gradient audit")
    if float(config["loss"].get("active_region_weight", 0.0)) <= 0.0:
        raise ValueError("active_region_weight must be positive for gradient audit")
    device = resolve_device(args.device)
    register_all_modules()
    dataset = _build_dataset(config)
    batch_size = min(int(config["data"].get("train_batch_size", 2)), len(dataset))
    samples = [dataset[index] for index in range(batch_size)]
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    batch = move_to_device(collate(samples), device)
    model = build_model(config["model"]).to(device).train()
    criterion = _build_pose_criterion(config)
    prediction = model(batch)
    total, losses = _loss_values(prediction, batch, criterion, config["loss"])
    if not bool(torch.isfinite(total)):
        raise RuntimeError("gradient audit loss is NaN/Inf")
    total.backward()

    selectors = {
        "observed_encoder": lambda name: name.startswith("observed_encoder."),
        "template_encoder": lambda name: name.startswith("template_encoder."),
        "interaction_transformer": lambda name: name.startswith(
            "interaction_transformer."
        ),
        "pose_decoder": lambda name: name.startswith("pose_head.")
        and not name.startswith("pose_head.logit_projection."),
        "pose_logits_head": lambda name: name.startswith(
            "pose_head.logit_projection."
        ),
        "observed_region_head": lambda name: name.startswith(
            "symmetry_head.point_classifier."
        ),
        "active_region_head": lambda name: name.startswith(
            "symmetry_head.active_classifier."
        ),
    }
    norms = {}
    parameters = {}
    for label, selector in selectors.items():
        norm, names = _gradient_norm(model, selector)
        norms[label] = norm
        parameters[label] = names
    required = ("observed_region_head", "active_region_head")
    missing = [name for name in required if not math.isfinite(norms[name]) or norms[name] <= 0.0]
    payload = {
        "status": "error" if missing else "ok",
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "sample_ids": list(batch["sample_id"]),
        "loss_total": float(total.detach()),
        "losses": {
            key: float(value.detach())
            for key, value in losses.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        },
        "gradient_norms": norms,
        "parameters_by_group": parameters,
        "missing_required_gradients": missing,
    }
    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = (
            Path(config["experiment"]["work_dir_root"])
            / "gradient_audits"
            / stamp
            / "gradient_audit.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output": str(output)}, indent=2))
    if missing:
        raise RuntimeError(f"required region gradients are zero/missing: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
