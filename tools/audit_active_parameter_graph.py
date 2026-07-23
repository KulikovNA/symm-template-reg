#!/usr/bin/env python3
"""Audit the real autograd graph of the clean ten-view scratch model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from ten_view_scratch_common import load_scratch_context  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.overfit_trainer import _build_pose_criterion, _loss_values  # noqa: E402
from symm_template_reg.models.detectors.coordinate_guided_surface_registration_v3 import (  # noqa: E402
    LEGACY_MODULE_TOKENS,
)


def audit(config_path, manifest_path, output: Path, device, batch_size=2, allow_unused=()):
    output.mkdir(parents=True, exist_ok=False)
    config, _, _, samples, collate, model = load_scratch_context(
        config_path, manifest_path, output, device
    )
    model.train()
    called = set()
    hooks = [
        module.register_forward_hook(lambda _m, _i, _o, name=name: called.add(name))
        for name, module in model.named_modules()
    ]
    criterion = _build_pose_criterion(config)
    loss_cfg = config["loss"]
    loss_cfg["joint_surface_correspondence_pose_v3"]["_runtime_epoch"] = 0
    model.zero_grad(set_to_none=True)
    batches = 0
    total_loss = 0.0
    for offset in range(0, 10, int(batch_size)):
        batch = move_to_device(collate(samples[offset:offset + int(batch_size)]), device)
        prediction = model(batch)
        loss, _ = _loss_values(prediction, batch, criterion, loss_cfg)
        (loss / (10 / int(batch_size))).backward()
        total_loss += float(loss.detach())
        batches += 1
    for hook in hooks:
        hook.remove()
    allow = tuple(map(str, allow_unused))
    rows = []
    trainable_used, trainable_no_gradient, frozen_used = [], [], []
    for name, parameter in model.named_parameters():
        module_name = name.rsplit(".", 1)[0]
        used = parameter.grad is not None
        status = (
            "trainable_and_used" if parameter.requires_grad and used
            else "trainable_but_no_gradient" if parameter.requires_grad
            else "frozen_but_used" if used else "frozen_and_unused"
        )
        rows.append({
            "parameter": name, "module": module_name,
            "parameter_count": parameter.numel(), "trainable": parameter.requires_grad,
            "gradient_present": used,
            "gradient_norm": 0.0 if parameter.grad is None else float(parameter.grad.detach().float().norm()),
            "status": status,
        })
        if status == "trainable_and_used": trainable_used.append(name)
        elif status == "trainable_but_no_gradient": trainable_no_gradient.append(name)
        elif status == "frozen_but_used": frozen_used.append(name)
    unused_disallowed = [
        name for name in trainable_no_gradient
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in allow)
    ]
    instantiated_unused = []
    for name, module in model.named_modules():
        direct = list(module.parameters(recurse=False))
        if direct and all(parameter.grad is None for parameter in direct):
            instantiated_unused.append(name)
    instantiated_unused.sort()
    module_counts = {}
    for row in rows:
        top = row["parameter"].split(".", 1)[0]
        module_counts[top] = module_counts.get(top, 0) + int(row["parameter_count"])
    legacy_modules = sorted(
        name for name, _ in model.named_modules()
        if any(token in name.lower() for token in LEGACY_MODULE_TOKENS)
    )
    legacy_parameters = sorted(
        name for name, _ in model.named_parameters()
        if any(token in name.lower() for token in LEGACY_MODULE_TOKENS)
    )
    report = {
        "audit_passed": not unused_disallowed and not legacy_modules and not legacy_parameters,
        "real_forward_sample_count": 10,
        "backward_sample_count": 10,
        "micro_batch_size": int(batch_size),
        "accumulation_steps": batches,
        "mean_micro_batch_loss": total_loss / batches,
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_and_used": trainable_used,
        "trainable_but_no_gradient": trainable_no_gradient,
        "frozen_but_used": frozen_used,
        "instantiated_parameter_modules_but_unused": instantiated_unused,
        "unused_trainable_allowlist": list(allow),
        "disallowed_unused_trainable_parameters": unused_disallowed,
        "legacy_modules": legacy_modules,
        "legacy_parameters": legacy_parameters,
        "parameter_count_by_top_level_module": module_counts,
        "active_graph_source": "real_forward_loss_backward",
    }
    with (output / "active_parameter_graph.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "active_parameter_graph.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Clean V3 active parameter graph", "",
        f"- passed: `{report['audit_passed']}`",
        f"- total/trainable parameters: `{report['total_parameter_count']}` / `{report['trainable_parameter_count']}`",
        f"- trainable without gradient: `{len(trainable_no_gradient)}`",
        f"- legacy parameters: `{len(legacy_parameters)}`", "",
        "| top-level module | parameters |", "|---|---:|",
        *(f"| {name} | {count} |" for name, count in sorted(module_counts.items())),
    ]
    (output / "active_parameter_graph_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if not report["audit_passed"]:
        raise RuntimeError(f"active parameter graph audit failed: {unused_disallowed[:10]}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--allow-unused", nargs="*", default=())
    args = parser.parse_args()
    result = audit(
        args.config, args.manifest, Path(args.output_dir).expanduser().resolve(),
        torch.device(args.device), args.micro_batch_size, args.allow_unused,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
