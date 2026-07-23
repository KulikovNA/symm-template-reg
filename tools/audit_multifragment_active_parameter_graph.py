#!/usr/bin/env python3
"""Audit the real 16-sample autograd graph of clean V3."""

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

from multifragment_overfit_common import load_multifragment_context  # noqa: E402
from symm_template_reg.engine.multifragment_overfit import WARNING_FLAGS  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.overfit_trainer import _build_pose_criterion, _loss_values  # noqa: E402
from symm_template_reg.models.detectors.coordinate_guided_surface_registration_v3 import LEGACY_MODULE_TOKENS  # noqa: E402


def run(args):
    output = Path(args.output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=False)
    config, _, _, _, samples, collate, model = load_multifragment_context(
        args.config, args.manifest, output, args.device
    )
    model.train(); criterion = _build_pose_criterion(config)
    config["loss"]["joint_surface_correspondence_pose_v3"]["_runtime_epoch"] = 0
    called = set()
    hooks = [module.register_forward_hook(lambda _m, _i, _o, name=name: called.add(name)) for name, module in model.named_modules()]
    model.zero_grad(set_to_none=True)
    micro = int(args.micro_batch_size)
    for offset in range(0, 16, micro):
        batch = move_to_device(collate(samples[offset:offset + micro]), torch.device(args.device))
        prediction = model(batch); loss, _ = _loss_values(prediction, batch, criterion, config["loss"])
        (loss / (16 // micro)).backward()
    for hook in hooks: hook.remove()
    rows = []
    for name, parameter in model.named_parameters():
        rows.append({
            "parameter": name, "parameter_count": parameter.numel(),
            "trainable": parameter.requires_grad, "gradient_present": parameter.grad is not None,
            "gradient_norm": 0.0 if parameter.grad is None else float(parameter.grad.detach().float().norm()),
        })
    missing = [row["parameter"] for row in rows if row["trainable"] and not row["gradient_present"]]
    unused_modules = sorted(
        name for name, module in model.named_modules()
        if (direct := list(module.parameters(recurse=False)))
        and all(parameter.grad is None for parameter in direct)
    )
    legacy = sorted(name for name, _ in model.named_parameters() if any(token in name.lower() for token in LEGACY_MODULE_TOKENS))
    module_norms = {}
    for row in rows:
        top = row["parameter"].split(".", 1)[0]
        module_norms[top] = module_norms.get(top, 0.0) + row["gradient_norm"] ** 2
    module_norms = {key: value ** 0.5 for key, value in module_norms.items()}
    report = {
        **WARNING_FLAGS,
        "audit_passed": not missing and not legacy and not unused_modules,
        "real_forward_sample_count": 16, "micro_batch_size": micro,
        "gradient_accumulation_steps": 16 // micro,
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "module_gradient_norms": module_norms,
        "trainable_parameters_without_gradients": missing,
        "instantiated_but_unused_modules": unused_modules,
        "unused_trainable_allowlist": [], "legacy_parameter_count": sum(row["parameter_count"] for row in rows if row["parameter"] in legacy),
        "legacy_parameters": legacy, "legacy_checkpoint_keys": [],
    }
    with (output / "active_parameter_graph.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "active_parameter_graph.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "active_parameter_graph_report.md").write_text(
        f"# 4x4 active graph\n\n- passed: `{report['audit_passed']}`\n- total/trainable: `{report['total_parameter_count']}` / `{report['trainable_parameter_count']}`\n- missing gradients: `{len(missing)}`\n- legacy parameters: `{len(legacy)}`\n",
        encoding="utf-8",
    )
    if not report["audit_passed"]: raise RuntimeError(f"active graph failed: missing={missing[:5]}, unused={unused_modules[:5]}, legacy={legacy[:5]}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--micro-batch-size", type=int, default=2); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); result = run(args); print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
