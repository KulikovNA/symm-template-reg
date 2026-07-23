#!/usr/bin/env python3
"""Measure all exact 16-sample batch/accumulation modes and select the fastest."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from multifragment_overfit_common import load_multifragment_context  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.overfit_trainer import _build_pose_criterion, _loss_values  # noqa: E402
from symm_template_reg.engine.seed import seed_everything  # noqa: E402
from symm_template_reg.engine.single_fragment import build_selective_optimizer_parameter_groups  # noqa: E402
from symm_template_reg.models import build_model  # noqa: E402


MODES = ((16, 1), (8, 2), (4, 4), (2, 8))


def _stats(values):
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[min(len(ordered) - 1, math.ceil(.9 * len(ordered)) - 1)],
        "max_ms": max(ordered),
    }


def _optimizer(config, model):
    groups = build_selective_optimizer_parameter_groups(
        model, default_lr=float(config["train"]["optimizer"]["lr"]),
        prefix_learning_rates=config["stage"]["prefix_learning_rates"],
    )
    return torch.optim.AdamW(groups, lr=float(config["train"]["optimizer"]["lr"]), weight_decay=0.0)


def _batch_layout(samples, collate, batch_size):
    lengths = []
    for index, sample in enumerate(samples):
        padded = collate([sample])["observed"].to_padded()
        lengths.append((int(padded["valid_mask"].sum()), index))
    order = list(range(16)) if batch_size == 16 else [index for _, index in sorted(lengths)]
    groups = [order[start : start + batch_size] for start in range(0, 16, batch_size)]
    host_batches = [collate([samples[index] for index in group]) for group in groups]
    real_points = sum(length for length, _ in lengths)
    padded_points = sum(
        int(batch["observed"].to_padded()["points"].shape[0]
            * batch["observed"].to_padded()["points"].shape[1])
        for batch in host_batches
    )
    return host_batches, groups, real_points, padded_points, lengths


def _state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _one_step(config, initial_state, host_batches, device, criterion, *, timed):
    model = build_model(config["model"]).to(device)
    model.load_state_dict(initial_state, strict=True); model.train()
    optimizer = _optimizer(config, model); optimizer.zero_grad(set_to_none=True)
    forward_ms = backward_ms = 0.0
    wall_start = torch.cuda.Event(enable_timing=True); wall_end = torch.cuda.Event(enable_timing=True)
    wall_start.record()
    for host_batch in host_batches:
        batch = move_to_device(host_batch, device)
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(); prediction = model(batch)
        loss, _ = _loss_values(prediction, batch, criterion, config["loss"])
        end.record(); torch.cuda.synchronize(device); forward_ms += start.elapsed_time(end)
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(); (loss * (len(host_batch["sample_id"]) / 16.0)).backward(); end.record()
        torch.cuda.synchronize(device); backward_ms += start.elapsed_time(end)
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer_start = torch.cuda.Event(enable_timing=True); optimizer_end = torch.cuda.Event(enable_timing=True)
    optimizer_start.record(); optimizer.step(); optimizer_end.record(); wall_end.record()
    torch.cuda.synchronize(device)
    result = {
        "wall_ms": wall_start.elapsed_time(wall_end), "forward_ms": forward_ms,
        "backward_ms": backward_ms, "optimizer_ms": optimizer_start.elapsed_time(optimizer_end),
        "gradients": gradients, "updated": _state(model),
    }
    del model, optimizer
    if not timed:
        gc.collect(); torch.cuda.empty_cache()
    return result


def _comparison(left, right):
    max_abs = 0.0; dot = left_norm = right_norm = 0.0
    for name, value in left.items():
        a = value.double().flatten(); b = right[name].double().flatten()
        max_abs = max(max_abs, float((a - b).abs().max()))
        dot += float(a @ b); left_norm += float(a @ a); right_norm += float(b @ b)
    cosine = dot / max(math.sqrt(left_norm * right_norm), 1e-30)
    return {"max_abs_diff": max_abs, "cosine_similarity": cosine}


def run(args):
    device = torch.device(args.device)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    output = Path(args.output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=False)
    config, _, _, _, samples, collate, initial = load_multifragment_context(
        args.config, args.manifest, output / "dataset_cache", "cpu"
    )
    initial_state = _state(initial); del initial
    config["loss"]["joint_surface_correspondence_pose_v3"]["_runtime_epoch"] = 1
    criterion = _build_pose_criterion(config)
    layouts = {}
    reference = None; equivalence = {}; rows = []; padding_rows = []
    for batch_size, accumulation in MODES:
        host_batches, groups, real_points, padded_points, lengths = _batch_layout(
            samples, collate, batch_size
        )
        layouts[batch_size] = host_batches
        seed_everything(int(config.get("seed", 0)))
        audit = _one_step(config, initial_state, host_batches, device, criterion, timed=False)
        if reference is None:
            reference = audit
        grad_cmp = _comparison(reference["gradients"], audit["gradients"])
        update_cmp = _comparison(reference["updated"], audit["updated"])
        passed = (
            grad_cmp["max_abs_diff"] <= 1e-5
            and grad_cmp["cosine_similarity"] >= .999999
            and update_cmp["max_abs_diff"] <= 1e-5
            and update_cmp["cosine_similarity"] >= .999999
        )
        equivalence[batch_size] = {
            "passed": passed, "gradients": grad_cmp, "parameter_update": update_cmp,
        }
        seed_everything(int(config.get("seed", 0)))
        model = build_model(config["model"]).to(device); model.load_state_dict(initial_state); model.train()
        optimizer = _optimizer(config, model)
        measurements = []
        total_steps = int(args.warmup_steps) + int(args.measure_steps)
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            wall_start = torch.cuda.Event(enable_timing=True); wall_end = torch.cuda.Event(enable_timing=True)
            wall_start.record(); forward_ms = backward_ms = 0.0
            for host_batch in host_batches:
                batch = move_to_device(host_batch, device)
                start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                start.record(); prediction = model(batch)
                loss, _ = _loss_values(prediction, batch, criterion, config["loss"])
                end.record()
                backward_start = torch.cuda.Event(enable_timing=True); backward_end = torch.cuda.Event(enable_timing=True)
                backward_start.record(); (loss * (len(host_batch["sample_id"]) / 16.0)).backward(); backward_end.record()
                torch.cuda.synchronize(device)
                forward_ms += start.elapsed_time(end); backward_ms += backward_start.elapsed_time(backward_end)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt_start = torch.cuda.Event(enable_timing=True); opt_end = torch.cuda.Event(enable_timing=True)
            opt_start.record(); optimizer.step(); opt_end.record(); wall_end.record(); torch.cuda.synchronize(device)
            if step >= int(args.warmup_steps):
                measurements.append({
                    "wall_ms": wall_start.elapsed_time(wall_end), "forward_ms": forward_ms,
                    "backward_ms": backward_ms, "optimizer_ms": opt_start.elapsed_time(opt_end),
                })
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            optimizer.zero_grad(set_to_none=True)
            for host_batch in host_batches:
                batch = move_to_device(host_batch, device); prediction = model(batch)
                loss, _ = _loss_values(prediction, batch, criterion, config["loss"])
                (loss * (len(host_batch["sample_id"]) / 16.0)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        kernel_count = sum(
            event.count for event in prof.key_averages()
            if getattr(event, "device_time_total", 0.0) > 0
        )
        wall = _stats([row["wall_ms"] for row in measurements])
        row = {
            "batch_size": batch_size, "gradient_accumulation_steps": accumulation,
            "optimizer_step_median_ms": wall["median_ms"],
            "optimizer_step_p90_ms": wall["p90_ms"], "optimizer_step_max_ms": wall["max_ms"],
            "samples_per_second": 16000.0 / wall["median_ms"],
            "points_per_second": real_points * 1000.0 / wall["median_ms"],
            "forward_median_ms": _stats([item["forward_ms"] for item in measurements])["median_ms"],
            "backward_median_ms": _stats([item["backward_ms"] for item in measurements])["median_ms"],
            "optimizer_median_ms": _stats([item["optimizer_ms"] for item in measurements])["median_ms"],
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20,
            "real_point_count": real_points, "padded_point_count": padded_points,
            "padding_ratio": 1.0 - real_points / padded_points,
            "cuda_kernel_count": kernel_count, "equivalence_passed": passed,
            "gradient_max_abs_diff": grad_cmp["max_abs_diff"],
            "update_max_abs_diff": update_cmp["max_abs_diff"],
        }
        rows.append(row)
        padding_rows.append({
            "batch_size": batch_size, "micro_batch_sample_indices": groups,
            "sample_point_lengths": [length for length, _ in lengths],
            "observed_encoder_real_points": real_points,
            "observed_encoder_padded_points": padded_points,
            "fine_adapter_real_points": real_points,
            "fine_adapter_padded_points": padded_points,
            "interaction_real_tokens": sum(min(length, 256) for length, _ in lengths),
            "interaction_padded_tokens": sum(len(group) * min(max(lengths[index][0] for index in group), 256) for group in groups),
            "padding_fraction": row["padding_ratio"],
        })
        del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    passing = [row for row in rows if row["equivalence_passed"]]
    selected = min(passing, key=lambda row: row["optimizer_step_median_ms"])
    with (output / "training_batch_benchmark.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    selection = {
        "selected_batch_size": selected["batch_size"],
        "selected_gradient_accumulation_steps": selected["gradient_accumulation_steps"],
        "selection_rule": "lowest measured median among equivalence-passing modes",
        "equivalence": equivalence, "selected_metrics": selected,
        "equal_sample_weight": 1.0 / 16.0,
    }
    (output / "selected_training_batch_mode.json").write_text(json.dumps(selection, indent=2) + "\n")
    (output / "padding_instrumentation.json").write_text(json.dumps(padding_rows, indent=2) + "\n")
    lines = ["# Training batch benchmark", "", "| batch | accum | median ms | p90 ms | padding | pass |", "|---:|---:|---:|---:|---:|:---:|"]
    for row in rows:
        lines.append(f"| {row['batch_size']} | {row['gradient_accumulation_steps']} | {row['optimizer_step_median_ms']:.3f} | {row['optimizer_step_p90_ms']:.3f} | {row['padding_ratio']:.4f} | {row['equivalence_passed']} |")
    lines.extend(["", f"Selected: batch={selected['batch_size']}, accumulation={selected['gradient_accumulation_steps']}."])
    (output / "training_batch_benchmark.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(selection, indent=2)); return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    return run(parser.parse_args())


if __name__ == "__main__": raise SystemExit(main())
