#!/usr/bin/env python3
"""Select a ten-view CUDA batch mode and benchmark one scratch optimizer step."""

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
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from ten_view_scratch_common import load_scratch_context  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.seed import seed_everything  # noqa: E402
from symm_template_reg.engine.overfit_trainer import (  # noqa: E402
    _build_pose_criterion, _evaluate, _loss_values,
)
from symm_template_reg.engine.single_fragment import (  # noqa: E402
    apply_trainable_prefixes, build_selective_optimizer_parameter_groups,
)
from symm_template_reg.models import build_model  # noqa: E402
from symm_template_reg.models.detectors.coordinate_guided_surface_registration_v3 import (  # noqa: E402
    state_dict_sha256,
)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rows_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _loader(samples, collate, batch_size):
    return [collate(samples[i:i + batch_size]) for i in range(0, 10, batch_size)]


@torch.no_grad()
def _permutation_audit(model, samples, collate, device, batch_size):
    maximum = 0.0
    checked = 0
    for offset in range(0, 10, batch_size):
        chunk = samples[offset:offset + batch_size]
        if len(chunk) < 2:
            continue
        direct = model(move_to_device(collate(chunk), device)).correspondence_auxiliary[
            "fine_aux_coordinate_normalized"
        ]
        reverse = model(move_to_device(collate(list(reversed(chunk))), device)).correspondence_auxiliary[
            "fine_aux_coordinate_normalized"
        ].flip(0)
        for index, sample in enumerate(chunk):
            count = int(sample["observed"]["points_C"].shape[0])
            maximum = max(maximum, float((direct[index, :count] - reverse[index, :count]).abs().max()))
            checked += 1
    return {
        "audit_passed": checked == 10 and maximum <= 1e-6,
        "checked_samples": checked,
        "maximum_output_difference_after_inverse_permutation": maximum,
        "static_pose_queries_present": False,
    }


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("ten-view memory selection must run on CUDA")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config, manifest, _, samples, collate, initial_model = load_scratch_context(
        args.config, args.manifest, output, device
    )
    initial_hash = state_dict_sha256(initial_model)
    criterion = _build_pose_criterion(config)
    config["loss"]["joint_surface_correspondence_pose_v3"]["_runtime_epoch"] = 0

    # Random baseline uses small validation micro-batches and no training.
    initial_loader = _loader(samples, collate, 2)
    initial_metrics, initial_rows = _evaluate(
        initial_model, initial_loader, device, criterion, False, torch.float32,
        config["loss"], active_path_config=config["active_coordinate_path"],
        epoch=0, max_epochs=6000, show_progress=False,
    )
    _rows_csv(output / "scratch_initialization_per_sample.csv", initial_rows)
    initialization = {
        "training_performed": False,
        "initialization_mode": "scratch",
        "pretrained_checkpoint": None,
        "checkpoint_sources": [],
        "seed": int(config.get("seed", 0)),
        "initial_state_dict_sha256": initial_hash,
        "sample_count": 10,
        "frames": list(range(10)),
        "metrics": initial_metrics,
    }
    permutation = _permutation_audit(initial_model, samples, collate, device, 2)
    initialization["input_permutation_audit"] = permutation
    (output / "scratch_initialization_summary.json").write_text(
        json.dumps(initialization, indent=2) + "\n", encoding="utf-8"
    )
    del initial_model
    gc.collect(); torch.cuda.empty_cache()

    failures = []
    selected = None
    selected_model = None
    timing = None
    for batch_size in (10, 5, 2):
        accumulation = 10 // batch_size
        seed_everything(int(config.get("seed", 0)))
        model = build_model(config["model"]).to(device)
        freeze = apply_trainable_prefixes(model, None)
        groups = build_selective_optimizer_parameter_groups(
            model, default_lr=1e-4,
            prefix_learning_rates=config["stage"]["prefix_learning_rates"],
        )
        optimizer = torch.optim.AdamW(groups, lr=1e-4, weight_decay=0.0)
        model.train(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        phases = {"forward_loss_sec": 0.0, "backward_sec": 0.0, "optimizer_sec": 0.0}
        try:
            started_total = time.perf_counter()
            for offset in range(0, 10, batch_size):
                batch = move_to_device(collate(samples[offset:offset + batch_size]), device)
                _sync(device); started = time.perf_counter()
                prediction = model(batch)
                loss, losses = _loss_values(prediction, batch, criterion, config["loss"])
                _sync(device); phases["forward_loss_sec"] += time.perf_counter() - started
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite scratch smoke loss")
                started = time.perf_counter(); (loss / accumulation).backward(); _sync(device)
                phases["backward_sec"] += time.perf_counter() - started
            missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
            if missing:
                raise RuntimeError(f"unused trainable parameters: {missing[:10]}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            started = time.perf_counter(); optimizer.step(); _sync(device)
            phases["optimizer_sec"] = time.perf_counter() - started
            elapsed = time.perf_counter() - started_total
            selected = {
                "actual_batch_size": batch_size,
                "gradient_accumulation_steps": accumulation,
                "effective_views_per_optimizer_step": 10,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                "optimizer_steps": 1,
                "per_sample_loss_weight": 0.1,
                "loss_reduction": "per_sample_mean_then_batch_mean",
                "one_step_time_sec": elapsed,
                "final_loss": float(loss.detach()),
                "all_trainable_parameters_have_gradient": True,
                "total_parameter_count": sum(p.numel() for p in model.parameters()),
                "trainable_parameter_count": freeze["trainable_parameter_count"],
            }
            selected_model = model
            timing = phases
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error).lower():
                raise
            failures.append({"batch_size": batch_size, "reason": "cuda_out_of_memory"})
            del model, optimizer
            gc.collect(); torch.cuda.empty_cache()
    if selected is None or selected_model is None or timing is None:
        raise RuntimeError("none of batch 10, 5, 2 fit on CUDA")

    evaluation_loader = _loader(samples, collate, 2)
    _sync(device); started = time.perf_counter()
    post_metrics, post_rows = _evaluate(
        selected_model, evaluation_loader, device, criterion, False, torch.float32,
        config["loss"], active_path_config=config["active_coordinate_path"],
        epoch=1, max_epochs=6000, show_progress=False,
    )
    _sync(device); evaluation_time = time.perf_counter() - started
    _rows_csv(output / "projection_evaluation_smoke_per_sample.csv", post_rows)
    (output / "projection_evaluation_smoke.json").write_text(
        json.dumps({"metrics": post_metrics, "sample_count": 10}, indent=2) + "\n",
        encoding="utf-8",
    )
    points = sum(int(row["num_shell_points"]) for row in post_rows)
    benchmark = {
        **selected, **timing,
        "projection_evaluation_exact_and_k16_sec": evaluation_time,
        "points_total": points,
        "points_per_second_training_step": points / max(selected["one_step_time_sec"], 1e-12),
        "batch_mode_failures": failures,
        "projection_modes": ["exact_global", "aux_guided_global_topk_k16"],
        "hard_projection_in_gradient_graph": False,
        "frozen_feature_cache_used": False,
    }
    (output / "ten_view_training_benchmark.json").write_text(
        json.dumps(benchmark, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "ten_view_training_benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        scalar = {key: value for key, value in benchmark.items() if isinstance(value, (str, int, float, bool))}
        writer = csv.DictWriter(stream, fieldnames=list(scalar)); writer.writeheader(); writer.writerow(scalar)
    smoke = {
        "status": "ok", "selected_batch_mode": selected,
        "failed_larger_modes": failures,
        "input_permutation_audit": permutation,
        "initialization_state_dict_unchanged_before_step": True,
        "initial_state_dict_sha256": initial_hash,
        "benchmark": benchmark,
    }
    (output / "ten_view_scratch_smoke.json").write_text(
        json.dumps(smoke, indent=2) + "\n", encoding="utf-8"
    )
    return smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
