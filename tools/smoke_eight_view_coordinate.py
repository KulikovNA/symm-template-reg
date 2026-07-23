#!/usr/bin/env python3
"""CUDA memory smoke and one optimizer step over exactly eight views."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from coordinate_guided_audit_common import load_coordinate_audit_contexts  # noqa: E402
from symm_template_reg.config import load_config  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.overfit_trainer import _build_pose_criterion, _loss_values  # noqa: E402
from symm_template_reg.engine.single_fragment import (  # noqa: E402
    apply_trainable_prefixes, build_selective_optimizer_parameter_groups,
)
from symm_template_reg.models import register_all_modules  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, build_from_cfg  # noqa: E402


EXPECTED = (4, 5, 2, 8, 0, 1, 6, 9)


def run(args) -> dict:
    device = torch.device("cuda")
    config = load_config(args.config)
    output = Path(args.output).expanduser().resolve()
    contexts = load_coordinate_audit_contexts(
        args.checkpoint, args.manifest, output.parent / (output.stem + "_cache"), device
    )
    frames = tuple(int(context["sample"]["frame_id"]) for context in contexts)
    if frames != EXPECTED:
        raise ValueError(f"eight-view frame order mismatch: {frames}")
    batch_size = int(args.batch_size)
    if batch_size not in (4, 8) or 8 % batch_size:
        raise ValueError("batch size must be 8, or the explicit fallback 4")
    accumulation = 8 // batch_size
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    model = contexts[0]["model"]
    model.train()
    freeze = apply_trainable_prefixes(
        model, config["stage"]["trainable_module_prefixes"]
    )
    groups = build_selective_optimizer_parameter_groups(
        model, default_lr=1e-4,
        prefix_learning_rates=config["stage"]["prefix_learning_rates"],
    )
    optimizer = torch.optim.AdamW(groups, lr=1e-4, weight_decay=0.0)
    criterion = _build_pose_criterion(config)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    losses_by_batch = []
    per_sample_losses = []
    for offset in range(0, 8, batch_size):
        cpu_batch = collate([
            context["sample"] for context in contexts[offset : offset + batch_size]
        ])
        moved = move_to_device(cpu_batch, device)
        prediction = model(moved)
        total, losses = _loss_values(prediction, moved, criterion, config["loss"])
        if not bool(torch.isfinite(total)):
            raise RuntimeError("eight-view smoke produced non-finite loss")
        per_sample = losses.get("per_sample_loss_total")
        if not isinstance(per_sample, torch.Tensor) or len(per_sample) != batch_size:
            raise RuntimeError("per-sample loss contract is missing")
        per_sample_losses.extend(float(value) for value in per_sample.detach())
        losses_by_batch.append(float(total.detach()))
        (total / accumulation).backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(bool(value.isfinite().all()) for value in gradients):
        raise RuntimeError("eight-view smoke produced missing/non-finite gradients")
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
    ))
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    report = {
        "status": "ok",
        "device": "cuda",
        "frame_ids": list(frames),
        "actual_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_views_per_optimizer_step": 8,
        "optimizer_steps": 1,
        "per_sample_loss_weight": 1.0 / 8.0,
        "loss_reduction": "per_sample_mean_then_batch_mean",
        "batch_losses": losses_by_batch,
        "per_sample_losses": per_sample_losses,
        "gradient_norm": gradient_norm,
        "finite_gradients": True,
        "peak_gpu_memory_mb": peak,
        "elapsed_sec": elapsed,
        "trainable_parameter_count": freeze["trainable_parameter_count"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    register_all_modules()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
