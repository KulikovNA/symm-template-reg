#!/usr/bin/env python3
"""Run one finite fine-only optimizer step on the real four-view batch."""

from __future__ import annotations

import argparse
import json
import sys
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
from symm_template_reg.engine.overfit_trainer import (  # noqa: E402
    _build_pose_criterion, _evaluate, _loss_values,
)
from symm_template_reg.engine.single_fragment import (  # noqa: E402
    apply_trainable_prefixes, build_selective_optimizer_parameter_groups,
)
from symm_template_reg.models import register_all_modules  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, build_from_cfg  # noqa: E402


def run(args):
    device = torch.device(args.device)
    config = load_config(args.config)
    output = Path(args.output).expanduser().resolve()
    contexts = load_coordinate_audit_contexts(
        args.checkpoint, args.manifest, output.parent / (output.stem + "_cache"), device
    )
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    cpu_batch = collate([context["sample"] for context in contexts])
    batch = move_to_device(cpu_batch, device)
    model = contexts[0]["model"]
    model.train()
    report = apply_trainable_prefixes(
        model, config["stage"]["trainable_module_prefixes"]
    )
    groups = build_selective_optimizer_parameter_groups(
        model, default_lr=1e-4,
        prefix_learning_rates=config["stage"]["prefix_learning_rates"],
    )
    optimizer = torch.optim.AdamW(groups, lr=1e-4, weight_decay=0.0)
    criterion = _build_pose_criterion(config)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch)
    total, losses = _loss_values(prediction, batch, criterion, config["loss"])
    if not bool(torch.isfinite(total)):
        raise RuntimeError("four-view smoke produced non-finite loss")
    total.backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(bool(value.isfinite().all()) for value in gradients):
        raise RuntimeError("four-view smoke produced missing/non-finite gradients")
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
    ))
    optimizer.step()
    evaluation_metrics, evaluation_rows = _evaluate(
        model, [cpu_batch], device, criterion, False, torch.float16,
        config["loss"], active_path_config=config["active_coordinate_path"],
        epoch=0, max_epochs=1, show_progress=False, leave_progress=False,
    )
    required_metric = "eval/active/worst_sample_projection_score"
    if required_metric not in evaluation_metrics or len(evaluation_rows) != 4:
        raise RuntimeError("active four-view evaluation namespace smoke failed")
    result = {
        "status": "ok", "device": str(device), "optimizer_steps": 1,
        "samples": [context["sample"]["sample_id"] for context in contexts],
        "frame_ids": [int(context["sample"]["frame_id"]) for context in contexts],
        "trainable_parameter_count": report["trainable_parameter_count"],
        "loss_total": float(total.detach()), "gradient_norm": gradient_norm,
        "finite_gradients": True,
        "active_evaluation_metric": required_metric,
        "active_worst_sample_projection_score": evaluation_metrics[required_metric],
        "inactive_legacy_triangle_active": evaluation_metrics[
            "eval/inactive/legacy_triangle/active"
        ],
        "losses": {
            key: float(value.detach())
            for key, value in losses.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); register_all_modules()
    print(json.dumps(run(args), indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
