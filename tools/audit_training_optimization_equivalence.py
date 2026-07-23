#!/usr/bin/env python3
"""Strict one-step and multi-step audit of baseline versus optimized fp32."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from multifragment_overfit_common import load_multifragment_context  # noqa: E402
from symm_template_reg.config import load_config  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.overfit_trainer import (  # noqa: E402
    _build_pose_criterion, _build_scheduler, _loss_values,
)
from symm_template_reg.engine.seed import seed_everything  # noqa: E402
from symm_template_reg.engine.single_fragment import (  # noqa: E402
    build_selective_optimizer_parameter_groups,
)
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402


SCALAR_COMPONENTS = (
    "loss_total", "loss_fine_coordinate_aux_normalized",
    "loss_fine_coordinate_aux_tail_normalized", "loss_raw_aux_rotation_normalized",
    "loss_raw_aux_translation_normalized", "loss_raw_aux_alignment_normalized",
    "weighted_loss_fine_coordinate_aux", "weighted_loss_fine_coordinate_aux_tail",
    "weighted_loss_raw_aux_rotation", "weighted_loss_raw_aux_translation",
    "weighted_loss_raw_aux_alignment", "rotation_error_deg", "translation_total_mm",
    "aux_coordinate_rmse_mm", "aux_coordinate_p95_mm",
)


def _optimizer(config, model):
    groups = build_selective_optimizer_parameter_groups(
        model, default_lr=float(config["train"]["optimizer"]["lr"]),
        prefix_learning_rates=config["stage"]["prefix_learning_rates"],
    )
    return torch.optim.AdamW(
        groups, lr=float(config["train"]["optimizer"]["lr"]),
        weight_decay=float(config["train"]["optimizer"].get("weight_decay", 0.0)),
    )


def _cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _run_branch(config, initial_state, host_batch, device, steps):
    seed_everything(int(config.get("seed", 0)))
    model = build_model(config["model"]).to(device)
    model.load_state_dict(initial_state, strict=True)
    model.train()
    criterion = _build_pose_criterion(config)
    optimizer = _optimizer(config, model)
    scheduler = _build_scheduler(optimizer, config["train"]["scheduler"], steps)
    batch = move_to_device(host_batch, device)
    first = None
    curve = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        config["loss"]["joint_surface_correspondence_pose_v3"]["_runtime_epoch"] = step
        prediction = model(batch)
        total, losses = _loss_values(prediction, batch, criterion, config["loss"])
        total.backward()
        parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if first is None:
            gradient = {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            first = {
                "q_aux": prediction.correspondence_points_O.detach().cpu(),
                "pose": prediction.correspondence_pose.detach().cpu(),
                "selected": losses["selected_shared_symmetry_element"].detach().cpu(),
                "scalars": {
                    name: float(losses[name].detach()) for name in SCALAR_COMPONENTS
                    if name in losses and isinstance(losses[name], torch.Tensor)
                },
                "gradient": gradient,
                "global_gradient_norm": math.sqrt(sum(
                    float(value.double().square().sum()) for value in gradient.values()
                )),
            }
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(config["train"]["gradient_clip_norm"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        curve.append({
            "step": step, "loss": float(total.detach()),
            "gradient_norm": float(gradient_norm),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        if step == 1:
            first["updated_state"] = _cpu_state(model)
    assert first is not None
    result = {"first": first, "curve": curve, "final_state": _cpu_state(model)}
    del model, optimizer, scheduler, batch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _tensor_comparison(left, right):
    flat_left = torch.cat([value.double().flatten() for value in left.values()])
    flat_right = torch.cat([right[name].double().flatten() for name in left])
    difference = (flat_left - flat_right).abs()
    cosine = torch.nn.functional.cosine_similarity(flat_left, flat_right, dim=0)
    return {"max_abs_diff": float(difference.max()), "cosine_similarity": float(cosine)}


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("strict performance equivalence audit requires CUDA")
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline_config, _, _, _, samples, collate, initial_model = load_multifragment_context(
        args.baseline_config, args.manifest, output / "dataset_cache", "cpu"
    )
    initial_state = _cpu_state(initial_model)
    del initial_model
    optimized_config = load_config(args.optimized_config)
    register_all_modules()
    host_batch = collate(samples)
    baseline = _run_branch(
        deepcopy(baseline_config), initial_state, host_batch, device, int(args.steps)
    )
    optimized = _run_branch(
        deepcopy(optimized_config), initial_state, host_batch, device, int(args.steps)
    )
    q_diff = float((baseline["first"]["q_aux"] - optimized["first"]["q_aux"]).abs().max())
    pose_diff = float((baseline["first"]["pose"] - optimized["first"]["pose"]).abs().max())
    scalar_diff = {
        name: abs(baseline["first"]["scalars"][name] - optimized["first"]["scalars"][name])
        for name in baseline["first"]["scalars"]
    }
    selected_equal = bool(torch.equal(
        baseline["first"]["selected"], optimized["first"]["selected"]
    ))
    gradients = _tensor_comparison(
        baseline["first"]["gradient"], optimized["first"]["gradient"]
    )
    updates = _tensor_comparison(
        baseline["first"]["updated_state"], optimized["first"]["updated_state"]
    )
    final = _tensor_comparison(baseline["final_state"], optimized["final_state"])
    loss_curve_max = max(
        abs(left["loss"] - right["loss"])
        for left, right in zip(baseline["curve"], optimized["curve"])
    )
    physical_names = (
        "rotation_error_deg", "translation_total_mm",
        "aux_coordinate_rmse_mm", "aux_coordinate_p95_mm",
    )
    physical_max = max((scalar_diff.get(name, 0.0) for name in physical_names), default=0.0)
    checks = {
        "q_aux": q_diff <= 1e-6,
        "loss": max(scalar_diff.values(), default=0.0) <= 1e-6,
        "selected_symmetry": selected_equal,
        "physical_metrics": physical_max <= 1e-4,
        "gradient_max_abs": gradients["max_abs_diff"] <= 1e-5,
        "gradient_cosine": gradients["cosine_similarity"] >= 0.999999,
        "update_max_abs": updates["max_abs_diff"] <= 1e-5,
        "update_cosine": updates["cosine_similarity"] >= 0.999999,
        # The request specifies strict one-step loss/gradient/update limits and
        # asks to compare 20-step curves, but gives no curve threshold.  The
        # sequential fp32 tolerance is documented separately from the unchanged
        # 1e-5 final-parameter gate.
        "twenty_step_loss_curve": loss_curve_max <= 2e-5,
        "twenty_step_final_max_abs": final["max_abs_diff"] <= 1e-5,
        "twenty_step_final_cosine": final["cosine_similarity"] >= 0.999999,
    }
    report = {
        "audit_passed": all(checks.values()), "checks": checks,
        "steps": int(args.steps), "q_aux_max_abs_diff": q_diff,
        "procrustes_pose_max_abs_diff": pose_diff,
        "scalar_absolute_differences": scalar_diff,
        "selected_symmetry_exact_match": selected_equal,
        "active_physical_metric_max_difference_mm_or_deg": physical_max,
        "gradients": gradients,
        "global_gradient_norm_absolute_difference": abs(
            baseline["first"]["global_gradient_norm"]
            - optimized["first"]["global_gradient_norm"]
        ),
        "one_step_parameter_update": updates,
        "multi_step_loss_curve_max_abs_diff": loss_curve_max,
        "multi_step_loss_curve_documented_fp32_tolerance": 2e-5,
        "multi_step_final_parameters": final,
        "baseline_config": str(args.baseline_config),
        "optimized_config": str(args.optimized_config),
        "manifest": str(args.manifest),
    }
    (output / "fp32_equivalence_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    comparison_rows = []
    for left, right in zip(baseline["curve"], optimized["curve"]):
        comparison_rows.append({
            "step": left["step"], "baseline_loss": left["loss"],
            "optimized_loss": right["loss"],
            "loss_abs_diff": abs(left["loss"] - right["loss"]),
            "baseline_gradient_norm": left["gradient_norm"],
            "optimized_gradient_norm": right["gradient_norm"],
            "learning_rate": left["learning_rate"],
        })
    with (output / "twenty_step_learning_curves.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader(); writer.writerows(comparison_rows)
    (output / "twenty_step_comparison.json").write_text(json.dumps({
        "audit_passed": all(checks[name] for name in checks if name.startswith("twenty")),
        "steps": int(args.steps), "loss_curve_max_abs_diff": loss_curve_max,
        "final_parameters": final,
    }, indent=2, sort_keys=True) + "\n")
    (output / "shared_template_equivalence.json").write_text(json.dumps({
        "covered_by_combined_exact_fp32_audit": True,
        "q_aux_max_abs_diff": q_diff, "gradient_comparison": gradients,
        "one_step_parameter_update": updates,
    }, indent=2, sort_keys=True) + "\n")
    (output / "vectorized_loss_equivalence.json").write_text(json.dumps({
        "covered_by_combined_exact_fp32_audit": True,
        "scalar_absolute_differences": scalar_diff,
        "selected_symmetry_exact_match": selected_equal,
        "procrustes_pose_max_abs_diff": pose_diff,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["audit_passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--optimized-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
