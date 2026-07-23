#!/usr/bin/env python3
"""Prove or reject frozen fine-feature cache equivalence on four views."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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
from symm_template_reg.engine.frozen_feature_cache import (  # noqa: E402
    FINE_ONLY_PREFIXES,
    FrozenFeatureCache,
    build_frozen_feature_cache_key,
    cache_eligibility,
    cached_fine_coordinate_forward,
    capture_fine_adapter_inputs,
    fine_coordinate_active_loss,
    frozen_module_state_sha256,
    point_order_sha256,
)
from symm_template_reg.engine.single_fragment import apply_trainable_prefixes  # noqa: E402
from symm_template_reg.engine.overfit_trainer import _build_pose_criterion, _loss_values  # noqa: E402
from symm_template_reg.evaluation.active_coordinate import (  # noqa: E402
    active_row,
    evaluate_active_sample,
)
from symm_template_reg.models import register_all_modules  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _coordinate_loss(predicted, target, mask, vertices):
    extent = (vertices.amax(0) - vertices.amin(0)).clamp_min(1e-8)
    normalized_target = 2.0 * (target - vertices.amin(0)) / extent - 1.0
    error = torch.nn.functional.smooth_l1_loss(
        predicted[0, mask], normalized_target[mask], reduction="none"
    ).mean(-1)
    count = max(1, math.ceil(len(error) * 0.10))
    return error.mean() + 0.5 * error.topk(count).values.mean()


def _flatten_gradients(module):
    values = [
        (
            parameter.grad.detach().flatten()
            if parameter.grad is not None
            else torch.zeros_like(parameter).flatten()
        )
        for parameter in module.parameters()
    ]
    return torch.cat(values) if values else torch.empty(0)


def _parameter_vector(*modules):
    return torch.cat(
        [parameter.detach().flatten() for module in modules for parameter in module.parameters()]
    )


def audit(args) -> dict:
    output = Path(args.output_dir).expanduser().resolve()
    device = torch.device(args.device)
    config = load_config(args.config)
    contexts = load_coordinate_audit_contexts(
        args.checkpoint, args.manifest, output, device
    )
    model = contexts[0]["model"]
    apply_trainable_prefixes(model, FINE_ONLY_PREFIXES)
    model.eval()
    eligibility = cache_eligibility(
        model,
        trainable_prefixes=config["stage"]["trainable_module_prefixes"],
        augmentations_enabled=bool(config.get("augmentations", {}).get("enabled", False)),
        deterministic_point_sampling=True,
    )
    captured_by_sample = {}
    online_coordinates = {}
    online_predictions = {}
    feature_max_abs = 0.0
    point_orders = []
    tensor_shapes = {}
    with torch.no_grad():
        for context in contexts:
            sample_id = str(context["sample"]["sample_id"])
            prediction, captured = capture_fine_adapter_inputs(model, context["batch"])
            qn = prediction.correspondence_auxiliary["fine_aux_coordinate_normalized"]
            captured_by_sample[sample_id] = captured
            online_coordinates[sample_id] = qn.detach()
            online_predictions[sample_id] = prediction
            point_orders.append(
                point_order_sha256(
                    captured["observed_points_C"], captured["observed_valid_mask"]
                )
            )
            for name, value in captured.items():
                if isinstance(value, torch.Tensor):
                    tensor_shapes[f"{sample_id}/{name}"] = tuple(value.shape)
    manifest_payload = contexts[0]["manifest"]
    key, key_payload = build_frozen_feature_cache_key(
        frozen_module_state_sha256_value=frozen_module_state_sha256(model),
        initialization_checkpoint=args.checkpoint,
        manifest=args.manifest,
        template_sha256=manifest_payload["template_sha256"],
        sidecar_sha256=manifest_payload["symmetry_sidecar_sha256"],
        point_selection_policy=config["data"]["observed_filter"],
        model_config={
            "model": config["model"],
            "active_coordinate_path": config.get("active_coordinate_path", {}),
        },
        dtype="float32",
        tensor_shapes=tensor_shapes,
        point_order_sha256_value="|".join(point_orders),
    )
    cache = FrozenFeatureCache(output / "cache", key)
    cache_path = cache.store(captured_by_sample, key_payload)
    loaded = cache.load(device)["payload"]
    cached_coordinates = {}
    physical_differences = []
    loss_differences = []
    criterion = _build_pose_criterion(config)
    for context in contexts:
        sample_id = str(context["sample"]["sample_id"])
        payload = loaded[sample_id]
        for name, online in captured_by_sample[sample_id].items():
            if isinstance(online, torch.Tensor):
                difference = (
                    online.detach().ne(payload[name]).to(torch.float32)
                    if online.dtype == torch.bool
                    else (online.detach() - payload[name]).abs()
                )
                feature_max_abs = max(
                    feature_max_abs,
                    float(difference.max()),
                )
        with torch.no_grad():
            _, cached_qn = cached_fine_coordinate_forward(
                model.correspondence_head.fine_feature_adapter,
                model.correspondence_head.fine_coordinate_auxiliary_head,
                payload,
            )
        cached_coordinates[sample_id] = cached_qn
        online_loss, _ = _loss_values(
            online_predictions[sample_id], context["batch"], criterion, config["loss"]
        )
        cached_loss, _ = fine_coordinate_active_loss(
            cached_qn, context["batch"], context["mask"][None], config["loss"]
        )
        loss_differences.append(float((online_loss - cached_loss).abs()))
        extent = (context["vertices"].amax(0) - context["vertices"].amin(0)).clamp_min(1e-8)
        cached_q = 0.5 * (cached_qn[0] + 1.0) * extent + context["vertices"].amin(0)
        online_q = 0.5 * (online_coordinates[sample_id][0] + 1.0) * extent + context["vertices"].amin(0)
        active_rows = []
        for q in (online_q, cached_q):
            result = evaluate_active_sample(
                q_aux_O=q, valid_mask=context["mask"], target_O=context["target"],
                observed_C=context["observed"], vertices_O=context["vertices"],
                faces=context["faces"], equivalent_pose=context["equivalent_pose"],
                procrustes=model.weighted_procrustes, candidate_k=16,
            )
            active_rows.append(
                active_row(
                    result, sample_id=sample_id,
                    frame_id=int(context["sample"]["frame_id"]),
                    T_W_from_C=context["T_W_from_C"],
                )
            )
        scalar_keys = [
            key for key, value in active_rows[0].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and not key.endswith("runtime_ms")
        ]
        physical_differences.append(
            max(abs(float(active_rows[0][key]) - float(active_rows[1][key])) for key in scalar_keys)
        )

    # Gradient and one-step update equivalence are checked independently on
    # every view, always starting from the same audited trainable state.
    adapter_a = model.correspondence_head.fine_feature_adapter
    head_a = model.correspondence_head.fine_coordinate_auxiliary_head
    adapter_state = copy.deepcopy(adapter_a.state_dict())
    head_state = copy.deepcopy(head_a.state_dict())
    gradient_max_abs = 0.0
    update_max_abs = 0.0
    gradient_differences_by_sample = {}
    update_differences_by_sample = {}
    for context in contexts:
        sample_id = str(context["sample"]["sample_id"])
        adapter_a.load_state_dict(adapter_state)
        head_a.load_state_dict(head_state)
        adapter_b = copy.deepcopy(adapter_a).to(device)
        head_b = copy.deepcopy(head_a).to(device)
        payload = loaded[sample_id]
        model.zero_grad(set_to_none=True)
        online_prediction = model(context["batch"])
        _, q_b = cached_fine_coordinate_forward(adapter_b, head_b, payload)
        loss_a, _ = _loss_values(
            online_prediction, context["batch"], criterion, config["loss"]
        )
        loss_b, _ = fine_coordinate_active_loss(
            q_b, context["batch"], context["mask"][None], config["loss"]
        )
        loss_a.backward(); loss_b.backward()
        grad_a = torch.cat((_flatten_gradients(adapter_a), _flatten_gradients(head_a)))
        grad_b = torch.cat((_flatten_gradients(adapter_b), _flatten_gradients(head_b)))
        gradient_difference = float((grad_a - grad_b).abs().max())
        gradient_differences_by_sample[sample_id] = gradient_difference
        gradient_max_abs = max(gradient_max_abs, gradient_difference)
        optimizer_a = torch.optim.AdamW(
            [*adapter_a.parameters(), *head_a.parameters()], lr=1e-4, weight_decay=0.0
        )
        optimizer_b = torch.optim.AdamW(
            [*adapter_b.parameters(), *head_b.parameters()], lr=1e-4, weight_decay=0.0
        )
        optimizer_a.step(); optimizer_b.step()
        update_difference = float(
            (_parameter_vector(adapter_a, head_a) - _parameter_vector(adapter_b, head_b)).abs().max()
        )
        update_differences_by_sample[sample_id] = update_difference
        update_max_abs = max(update_max_abs, update_difference)
    adapter_a.load_state_dict(adapter_state)
    head_a.load_state_dict(head_state)

    def measure(mode: str, repeats: int = 2):
        times = []
        peak = 0.0
        for iteration in range(repeats + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            with torch.no_grad():
                for context in contexts:
                    sid = str(context["sample"]["sample_id"])
                    if mode == "online":
                        model(context["batch"])
                    else:
                        cached_fine_coordinate_forward(
                            model.correspondence_head.fine_feature_adapter,
                            model.correspondence_head.fine_coordinate_auxiliary_head,
                            loaded[sid],
                        )
            _sync(device)
            elapsed = time.perf_counter() - started
            if iteration:
                times.append(elapsed)
                if device.type == "cuda":
                    peak = max(peak, torch.cuda.max_memory_allocated(device) / 1024 ** 2)
        return sum(times) / len(times), peak

    online_time, online_memory = measure("online")
    cached_time, cached_memory = measure("cached")
    benchmark_rows = [
        {"mode": "online", "epoch_time_sec": online_time, "peak_memory_mb": online_memory},
        {"mode": "cached", "epoch_time_sec": cached_time, "peak_memory_mb": cached_memory},
    ]
    speedup = online_time / max(cached_time, 1e-12)
    checks = {
        "policy": bool(eligibility["cache_allowed_by_policy"]),
        "maximum_absolute_feature_difference": feature_max_abs <= 1e-6,
        "loss_difference": max(loss_differences) <= 1e-6,
        "trainable_gradient_difference": gradient_max_abs <= 1e-6,
        "one_step_parameter_update_difference": update_max_abs <= 1e-6,
        "physical_metrics_difference": max(physical_differences) <= 1e-6,
    }
    report = {
        "audit_completed": True,
        "cache_allowed": all(checks.values()),
        "checks": checks,
        "eligibility": eligibility,
        "cache_key": key,
        "cache_key_payload": key_payload,
        "cache_path": str(cache_path),
        "maximum_absolute_feature_difference": feature_max_abs,
        "maximum_loss_difference": max(loss_differences),
        "maximum_trainable_gradient_difference": gradient_max_abs,
        "gradient_differences_by_sample": gradient_differences_by_sample,
        "maximum_one_step_parameter_update_difference": update_max_abs,
        "update_differences_by_sample": update_differences_by_sample,
        "maximum_physical_metric_difference": max(physical_differences),
        "documented_gradient_tolerance": 1e-6,
        "documented_update_tolerance": 1e-6,
        "online_epoch_time_sec": online_time,
        "cached_epoch_time_sec": cached_time,
        "speedup": speedup,
        "online_peak_memory_mb": online_memory,
        "cached_peak_memory_mb": cached_memory,
        "fallback": "cached" if all(checks.values()) else "online",
    }
    (output / "frozen_feature_cache_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "frozen_feature_cache_benchmark.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(benchmark_rows[0]))
        writer.writeheader(); writer.writerows(benchmark_rows)
    (output / "frozen_feature_cache_report.md").write_text(
        "\n".join((
            "# Frozen feature cache audit", "",
            f"- cache allowed: `{report['cache_allowed']}`",
            f"- maximum feature difference: `{feature_max_abs:.9g}`",
            f"- maximum loss difference: `{max(loss_differences):.9g}`",
            f"- maximum gradient difference: `{gradient_max_abs:.9g}`",
            f"- maximum update difference: `{update_max_abs:.9g}`",
            f"- online epoch: `{online_time:.6f} s`",
            f"- cached epoch: `{cached_time:.6f} s`",
            f"- measured speed-up: `{speedup:.3f}x`",
        )) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    register_all_modules()
    result = audit(args)
    print(json.dumps({"output_dir": str(output), **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
