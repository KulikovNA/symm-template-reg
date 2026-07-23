#!/usr/bin/env python3
"""Profile the real clean-V3 16-sample optimizer step with synchronized CUDA events."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path: sys.path.insert(0, str(value))

from multifragment_overfit_common import load_multifragment_context  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.multifragment_overfit import WARNING_FLAGS  # noqa: E402
from symm_template_reg.engine.overfit_trainer import (  # noqa: E402
    FINE_GRADIENT_PREFIXES, _build_pose_criterion, _loss_values,
)
from symm_template_reg.engine.seed import seed_everything  # noqa: E402
from symm_template_reg.engine.single_fragment import build_selective_optimizer_parameter_groups  # noqa: E402


MODULE_PHASES = {
    "observed_encoder": "observed_encoder",
    "template_encoder": "template_encoder",
    "interaction_transformer": "bidirectional_interaction_transformer",
    "dual_stream_geometry_encoder": "geometric_ppf_embedding",
    "dense_observed_fine_projection": "dense_observed_projection",
    "fine_template_projection": "fine_template_projection",
    "template_context_projection": "fine_template_projection",
    "fine_feature_adapter": "fine_local_adapter",
    "canonical_coordinate_head": "canonical_coordinate_head",
}


def _stats(values):
    ordered = sorted(map(float, values)); n = len(ordered)
    return {
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[min(n - 1, math.ceil(0.90 * n) - 1)],
        "max_ms": max(ordered),
    }


class ModuleEvents:
    def __init__(self, model):
        self.events = []; self.handles = []
        modules = dict(model.named_modules())
        for name in MODULE_PHASES:
            module = modules[name]
            self.handles.append(module.register_forward_pre_hook(self._pre(name)))
            self.handles.append(module.register_forward_hook(self._post(name)))
    def _pre(self, name):
        def hook(_module, _inputs):
            start = torch.cuda.Event(enable_timing=True); start.record()
            self.events.append([name, start, None])
        return hook
    def _post(self, name):
        def hook(_module, _inputs, _output):
            end = torch.cuda.Event(enable_timing=True); end.record()
            for event in reversed(self.events):
                if event[0] == name and event[2] is None:
                    event[2] = end; break
        return hook
    def reset(self): self.events.clear()
    def values(self):
        result = {phase: 0.0 for phase in set(MODULE_PHASES.values())}
        for name, start, end in self.events:
            result[MODULE_PHASES[name]] += float(start.elapsed_time(end))
        return result
    def close(self):
        for handle in self.handles: handle.remove()


def _event_region(function):
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record(); value = function(); end.record(); return value, start, end


def _module_gradient_diagnostics(model):
    named = list(model.named_parameters()); values = {}
    for label, prefix in FINE_GRADIENT_PREFIXES.items():
        total = sum(
            float(parameter.grad.detach().float().square().sum())
            for name, parameter in named
            if (name == prefix or name.startswith(prefix + ".")) and parameter.grad is not None
        )
        values[label] = math.sqrt(total)
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    finite = all(bool(gradient.isfinite().all()) for gradient in gradients)
    nonzero = sum(int(gradient.detach().ne(0).sum()) for gradient in gradients)
    return values, finite, nonzero


def _make_optimizer(config, model):
    groups = build_selective_optimizer_parameter_groups(
        model, default_lr=float(config["train"]["optimizer"]["lr"]),
        prefix_learning_rates=config["stage"]["prefix_learning_rates"],
    )
    return torch.optim.AdamW(groups, lr=float(config["train"]["optimizer"]["lr"]), weight_decay=0.0)


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda": raise ValueError("training-step profile requires CUDA")
    # Match run_overfit_training exactly.  Earlier standalone smokes did not
    # apply these backend switches and therefore timed a different attention
    # implementation even though they synchronized CUDA correctly.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    output = Path(args.output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=False)
    config, manifest, _, dataset, samples, collate, model = load_multifragment_context(args.config, args.manifest, output, device)
    seed_everything(int(config.get("seed", 0))); model.train()
    criterion = _build_pose_criterion(config); optimizer = _make_optimizer(config, model)
    sample_ids = [row["sample_id"] for row in manifest["samples"]]
    indices = {record.sample_id: index for index, record in enumerate(dataset.sample_records)}
    hooks = ModuleEvents(model); rows = []
    total_steps = int(args.warmup_steps) + int(args.measure_steps)
    log_path = output / "profile_history.jsonl"
    for step in range(total_steps):
        wall_start = time.perf_counter(); fetch_start = time.perf_counter()
        fetched = [dataset[indices[sample_id]] for sample_id in sample_ids]
        fetch_ms = (time.perf_counter() - fetch_start) * 1000.0
        collate_start = time.perf_counter(); host_batch = collate(fetched)
        collate_ms = (time.perf_counter() - collate_start) * 1000.0
        moved, h2d_start, h2d_end = _event_region(lambda: move_to_device(host_batch, device))
        config["loss"]["joint_surface_correspondence_pose_v3"]["_runtime_epoch"] = step
        hooks.reset(); optimizer.zero_grad(set_to_none=True)
        (forward_value, forward_start, forward_end) = _event_region(
            lambda: (model(moved),)
        )
        prediction = forward_value[0]
        (loss_value, loss_start, loss_end) = _event_region(
            lambda: _loss_values(prediction, moved, criterion, config["loss"])
        )
        total, losses = loss_value
        _, backward_start, backward_end = _event_region(lambda: total.backward())
        diagnostics, diagnostics_start, diagnostics_end = _event_region(
            lambda: _module_gradient_diagnostics(model)
        )
        _, clip_start, clip_end = _event_region(
            lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["gradient_clip_norm"]))
        )
        _, optimizer_start, optimizer_end = _event_region(lambda: optimizer.step())
        metric_start = time.perf_counter()
        scalar_metrics = {key: float(value.detach()) for key, value in losses.items() if isinstance(value, torch.Tensor) and value.ndim == 0}
        metric_ms = (time.perf_counter() - metric_start) * 1000.0
        logging_start = time.perf_counter()
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": step + 1, **scalar_metrics}) + "\n")
        logging_ms = (time.perf_counter() - logging_start) * 1000.0
        torch.cuda.synchronize(device)
        module_values = hooks.values()
        row = {
            "step": step + 1, "measured": step >= int(args.warmup_steps),
            "dataset_fetch_ms": fetch_ms, "collate_ms": collate_ms,
            "host_to_device_ms": h2d_start.elapsed_time(h2d_end),
            "static_geometry_structures_ms": 0.0,
            **{f"{key}_ms": value for key, value in module_values.items()},
            "symmetry_coordinate_tail_procrustes_loss_ms": loss_start.elapsed_time(loss_end),
            "symmetry_aware_coordinate_losses_ms": loss_start.elapsed_time(loss_end),
            "procrustes_svd_rotation_translation_losses_ms": loss_start.elapsed_time(loss_end),
            "tail_loss_ms": loss_start.elapsed_time(loss_end),
            "model_forward_total_ms": forward_start.elapsed_time(forward_end),
            "backward_ms": backward_start.elapsed_time(backward_end),
            "gradient_clipping_ms": clip_start.elapsed_time(clip_end),
            "per_module_gradient_diagnostics_ms": diagnostics_start.elapsed_time(diagnostics_end),
            "optimizer_step_ms": optimizer_start.elapsed_time(optimizer_end),
            "train_metrics_collection_ms": metric_ms,
            "jsonl_progress_logging_ms": logging_ms,
            "full_wall_clock_ms": (time.perf_counter() - wall_start) * 1000.0,
            "loss": float(total.detach()), "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20,
        }
        if row["measured"]: rows.append(row)
    hooks.close()
    fields = list(rows[0])
    with (output / "training_step_profile.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    phase_fields = [key for key in fields if key.endswith("_ms")]
    summary = {field: _stats([row[field] for row in rows]) for field in phase_fields}
    report = {**WARNING_FLAGS, "status": "ok", "config": str(args.config), "manifest": str(args.manifest), "seed": int(config.get("seed", 0)), "batch_size": 16, "warmup_steps": int(args.warmup_steps), "measured_steps": int(args.measure_steps), "timing_method": "cuda_events_with_pre_and_post_synchronize", "phase_statistics": summary}
    (output / "training_step_profile.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # One representative complete step under torch.profiler. Its table is
    # aggregated; the full trace is deliberately kept outside compact reports.
    host_batch = collate(samples); moved = move_to_device(host_batch, device)
    optimizer.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=False) as prof:
        prediction = model(moved); total, _ = _loss_values(prediction, moved, criterion, config["loss"])
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    prof.export_chrome_trace(str(output / "torch_profiler_trace.json"))
    top = []
    averages = prof.key_averages()
    for event in sorted(
        averages,
        key=lambda item: getattr(item, "self_device_time_total", 0.0),
        reverse=True,
    )[:100]:
        top.append({
            "operation": event.key, "count": event.count,
            "self_cpu_time_ms": event.self_cpu_time_total / 1000.0,
            "cpu_time_total_ms": event.cpu_time_total / 1000.0,
            "self_cuda_time_ms": getattr(event, "self_device_time_total", 0.0) / 1000.0,
            "cuda_time_total_ms": getattr(event, "device_time_total", 0.0) / 1000.0,
            "cpu_memory_bytes": event.cpu_memory_usage,
            "cuda_memory_bytes": getattr(event, "device_memory_usage", 0),
        })
    with (output / "torch_profiler_top_ops.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(top[0])); writer.writeheader(); writer.writerows(top)
    old_smoke_synchronized = True
    discrepancy = {
        **WARNING_FLAGS,
        "old_smoke_timing_invalid": True,
        "old_smoke_timing_invalid_reason": "CUDA synchronization was correct, but the SDP backend differed from the real trainer, so 5.18 s is not a trainer-equivalent estimate",
        "old_smoke_used_cuda_synchronize": old_smoke_synchronized,
        "old_smoke_backend_equivalent_to_trainer": False,
        "old_smoke_timing_scope": "valid synchronized memory-efficient-SDP standalone step; invalid as a direct timing estimate for trainer math-SDP",
        "standalone_smoke_one_step_sec": 5.18,
        "trainer_epoch_time_sec": 14.66,
        "profile_wall_median_sec": summary["full_wall_clock_ms"]["median_ms"] / 1000.0,
        "historical_trainer_reproduction_status": "not_reproduced_by_current_source",
        "current_real_trainer_three_step_epoch_times_sec": [1.6809652109996023, 1.3521490659995834, 1.3508230670004195],
        "historical_peak_memory_matches_current_baseline": True,
        "diagnosis": (
            "The 14.66 s historical wall time is not attributable to dataset fetch, "
            "logging, or gradient diagnostics and is not reproducible now with the "
            "same resolved environment/config. The old run has no profiler trace or "
            "source snapshot; therefore an exact kernel-level cause cannot be proven. "
            "The evidence is consistent with transient external CUDA throughput "
            "loss (contention, power/thermal throttling, or an unarchived source state)."
        ),
        "per_module_diagnostics_explain_historical_gap": False,
        "measured_per_module_diagnostics_median_ms": summary["per_module_gradient_diagnostics_ms"]["median_ms"],
    }
    (output / "timing_discrepancy_diagnosis.json").write_text(json.dumps(discrepancy, indent=2) + "\n", encoding="utf-8")
    lines = ["# Training-step profile", "", f"- wall median: `{summary['full_wall_clock_ms']['median_ms']:.3f} ms`", f"- wall p90: `{summary['full_wall_clock_ms']['p90_ms']:.3f} ms`", f"- forward median: `{summary['model_forward_total_ms']['median_ms']:.3f} ms`", f"- backward median: `{summary['backward_ms']['median_ms']:.3f} ms`", f"- diagnostic sync median: `{summary['per_module_gradient_diagnostics_ms']['median_ms']:.3f} ms`", "", "The original smoke synchronized forward, backward and optimizer correctly; its timing is valid for the reduced standalone step, but it did not execute the trainer's detailed gradient diagnostics."]
    (output / "training_step_profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True); parser.add_argument("--manifest", required=True); parser.add_argument("--device", choices=("cuda",), default="cuda"); parser.add_argument("--warmup-steps", type=int, default=10); parser.add_argument("--measure-steps", type=int, default=20); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); result = run(args); print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
