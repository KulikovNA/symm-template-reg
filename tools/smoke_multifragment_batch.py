#!/usr/bin/env python3
"""Select the first CUDA mode among 16x1, 8x2, 4x4 and 2x8."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path: sys.path.insert(0, str(value))

from multifragment_overfit_common import load_multifragment_context  # noqa: E402
from symm_template_reg.engine.multifragment_overfit import WARNING_FLAGS  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.overfit_trainer import _build_pose_criterion, _evaluate, _loss_values  # noqa: E402
from symm_template_reg.engine.seed import seed_everything  # noqa: E402
from symm_template_reg.engine.single_fragment import apply_trainable_prefixes, build_selective_optimizer_parameter_groups  # noqa: E402
from symm_template_reg.models import build_model  # noqa: E402
from symm_template_reg.models.detectors.coordinate_guided_surface_registration_v3 import state_dict_sha256  # noqa: E402


def _sync(device):
    if device.type == "cuda": torch.cuda.synchronize(device)


def _loader(samples, collate, batch_size):
    return [collate(samples[offset:offset + batch_size]) for offset in range(0, 16, batch_size)]


def _csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def _permutation(model, samples, collate, device):
    direct = model(move_to_device(collate(samples[:2]), device)).correspondence_points_O
    reverse = model(move_to_device(collate(list(reversed(samples[:2]))), device)).correspondence_points_O.flip(0)
    maximum = float((direct - reverse).abs().max())
    return {"checked_samples": 2, "maximum_inverse_permutation_difference": maximum, "passed": maximum <= 1e-6}


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda": raise ValueError("memory selection must run on CUDA")
    output = Path(args.output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=False)
    config, _, _, _, samples, collate, initial = load_multifragment_context(args.config, args.manifest, output, device)
    criterion = _build_pose_criterion(config)
    loss_cfg = config["loss"]; loss_cfg["joint_surface_correspondence_pose_v3"]["_runtime_epoch"] = 0
    initial_hash = state_dict_sha256(initial)
    initial_metrics, initial_rows = _evaluate(initial, _loader(samples, collate, 2), device, criterion, False, torch.float32, loss_cfg, active_path_config=config["active_coordinate_path"], epoch=0, max_epochs=8000, show_progress=False)
    _csv(output / "scratch_initialization_per_sample.csv", initial_rows)
    permutation = _permutation(initial, samples, collate, device)
    initialization = {
        **WARNING_FLAGS,
        "training_performed": False, "initialization_mode": "scratch", "pretrained_checkpoint": None,
        "checkpoint_sources": [], "initial_state_dict_sha256": initial_hash, "sample_count": 16,
        "fragment_ids": [0, 1, 2, 3], "frame_ids": [2, 4, 5, 8], "metrics": initial_metrics,
        "input_permutation_audit": permutation,
    }
    (output / "scratch_initialization_summary.json").write_text(json.dumps(initialization, indent=2) + "\n", encoding="utf-8")
    del initial; gc.collect(); torch.cuda.empty_cache()
    failures = []; selected = None; selected_model = None
    for batch_size in (16, 8, 4, 2):
        accumulation = 16 // batch_size; seed_everything(int(config.get("seed", 0)))
        model = build_model(config["model"]).to(device); apply_trainable_prefixes(model, None)
        groups = build_selective_optimizer_parameter_groups(model, default_lr=1e-4, prefix_learning_rates=config["stage"]["prefix_learning_rates"])
        optimizer = torch.optim.AdamW(groups, lr=1e-4, weight_decay=0.0)
        model.train(); optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(device)
        timings = {"forward_loss_sec": 0.0, "backward_sec": 0.0, "optimizer_sec": 0.0}
        try:
            total_start = time.perf_counter(); last_loss = None
            for offset in range(0, 16, batch_size):
                batch = move_to_device(collate(samples[offset:offset + batch_size]), device)
                _sync(device); started = time.perf_counter(); prediction = model(batch)
                loss, _ = _loss_values(prediction, batch, criterion, loss_cfg); _sync(device)
                timings["forward_loss_sec"] += time.perf_counter() - started
                if not bool(torch.isfinite(loss)): raise RuntimeError("non-finite loss")
                started = time.perf_counter(); (loss / accumulation).backward(); _sync(device)
                timings["backward_sec"] += time.perf_counter() - started; last_loss = loss
            missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
            if missing: raise RuntimeError(f"unused trainable parameters: {missing[:5]}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            started = time.perf_counter(); optimizer.step(); _sync(device); timings["optimizer_sec"] = time.perf_counter() - started
            selected = {
                "actual_batch_size": batch_size, "gradient_accumulation_steps": accumulation,
                "effective_samples_per_optimizer_step": 16, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                "selected_batch_reason": "first_fitting_mode_in_required_order",
                "loss_reduction": "per_sample_mean_then_batch_mean", "per_sample_weight": 1 / 16,
                "per_fragment_weight": 1 / 4, "per_frame_weight": 1 / 4,
                "one_step_time_sec": time.perf_counter() - total_start,
                "final_micro_batch_loss": float(last_loss.detach()), **timings,
                "total_parameter_count": sum(p.numel() for p in model.parameters()),
                "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
            selected_model = model; break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error).lower(): raise
            failures.append({"batch_size": batch_size, "gradient_accumulation_steps": accumulation, "reason": "cuda_out_of_memory"})
            del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    if selected is None: raise RuntimeError("none of the required 16-sample batch modes fit")
    post_metrics, post_rows = _evaluate(selected_model, _loader(samples, collate, 2), device, criterion, False, torch.float32, loss_cfg, active_path_config=config["active_coordinate_path"], epoch=1, max_epochs=8000, show_progress=False)
    _csv(output / "projection_evaluation_smoke_per_sample.csv", post_rows)
    projection = {**WARNING_FLAGS, "sample_count": 16, "projection_modes": ["exact_global", "aux_guided_global_topk_k16"], "metrics": post_metrics}
    (output / "projection_evaluation_smoke.json").write_text(json.dumps(projection, indent=2) + "\n", encoding="utf-8")
    report = {**WARNING_FLAGS, "status": "ok", "selected_batch_mode": selected, "failed_larger_modes": failures, "input_permutation_audit": permutation, "initial_state_dict_sha256": initial_hash, "hard_projection_in_gradient_graph": False}
    (output / "multifragment_batch_smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda"); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); result = run(args); print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
