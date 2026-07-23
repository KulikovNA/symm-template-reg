#!/usr/bin/env python3
"""Overfit frozen Stage-A fine features with small diagnostic heads."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from audit_local_triangle_target_contract import _padded_target, _sample  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.models.geometry.patch_targets import (  # noqa: E402
    multi_positive_softmax_loss,
)
from symm_template_reg.registry import COLLATE_FUNCTIONS, build_from_cfg  # noqa: E402


class LinearTriangleDiagnostic(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.query = nn.Linear(dimension, dimension, bias=False)

    def forward(self, point: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nd,nld->nl", self.query(point), candidate) / point.shape[-1] ** 0.5


class MLPTriangleDiagnostic(nn.Module):
    def __init__(self, dimension: int, hidden: int = 64):
        super().__init__()
        self.point = nn.Linear(dimension, hidden)
        self.candidate = nn.Linear(dimension, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, point: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        value = torch.relu(self.point(point)[:, None] + self.candidate(candidate))
        return self.output(value).squeeze(-1)


def _train_classifier(model, point, candidate, valid, mask, steps, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for step in range(steps + 1):
        logits = model(point, candidate).masked_fill(~mask, float("-inf"))
        loss = multi_positive_softmax_loss(logits, valid)
        top1 = valid.gather(-1, logits.argmax(-1)[:, None]).float().mean()
        top4_ids = logits.topk(min(4, logits.shape[-1]), -1).indices
        top4 = valid.gather(-1, top4_ids).any(-1).float().mean()
        if step in {0, steps}:
            history.append(
                {"step": step, "loss": float(loss.detach()), "valid_top1": float(top1), "valid_top4": float(top4)}
            )
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return history[-1], history


def _train_coordinate(point, target, steps, learning_rate):
    model = nn.Sequential(
        nn.Linear(point.shape[-1], 128), nn.ReLU(), nn.Linear(128, 3)
    ).to(point.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for step in range(steps + 1):
        prediction = model(point)
        distance = torch.linalg.vector_norm(prediction - target, dim=-1)
        loss = distance.square().mean()
        if step in {0, steps}:
            history.append(
                {
                    "step": step,
                    "rmse_mm": float(loss.sqrt().detach() * 1000.0),
                    "p95_mm": float(torch.quantile(distance.detach(), 0.95) * 1000.0),
                }
            )
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return history[-1], history


def run_capacity(args) -> dict:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    run = checkpoint.parents[1]
    output = Path(args.output_dir).expanduser().resolve()
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    config = deepcopy(config)
    head_cfg = config["model"]["correspondence_head"]
    head_cfg.update(
        teacher_forcing_initial_probability=1.0,
        teacher_forcing_final_probability=1.0,
        teacher_forcing_select_shared_symmetry_element=True,
        deduplicate_local_candidates=True,
        inject_all_valid_triangles=True,
        triangle_target_tolerance_m=args.triangle_target_tolerance_m,
        candidate_geometry_weight=1.0,
        max_local_candidate_total=32,
        sort_owned_faces_by_distance=True,
    )
    sample, _ = _sample(
        config, args.manifest, output / "cache" / "fragment_mesh_metadata"
    )
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    batch = move_to_device(collate([sample]), torch.device(args.device))
    model = build_model(config["model"]).to(args.device).train()
    payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.correspondence_head.teacher_forcing_probability = 1.0
    with torch.no_grad():
        prediction = model(batch)
    auxiliary = prediction.correspondence_auxiliary
    assert auxiliary is not None
    mask = prediction.observed_valid_mask[0]
    point = auxiliary["fine_observed_query_features"][0, mask].detach()
    candidate_ids = auxiliary["candidate_triangle_ids"][0, mask]
    candidate_mask = auxiliary["candidate_triangle_mask"][0, mask]
    global_features = auxiliary["global_triangle_features"][0]
    candidate = global_features[candidate_ids.clamp_min(0)].detach()
    valid = auxiliary["valid_triangle_local_mask"][0, mask] & candidate_mask
    target = _padded_target(batch)[0, mask].detach()
    selected_s = int(auxiliary["teacher_forcing_selected_symmetry_element"][0])
    # The head already constructed candidates with its selected shared S.  Use
    # its exact teacher target points from the loss convention by applying the
    # same transform through the public audit helper.
    from audit_local_triangle_target_contract import shared_symmetry_target
    target = shared_symmetry_target(
        _padded_target(batch)[0],
        batch["template_symmetry_metadata"][0],
        batch["gt"]["effective_symmetry_group"][0],
        selected_s,
    )[mask].detach()
    count = min(len(point), int(args.max_points))
    ids = torch.linspace(0, len(point) - 1, count, device=point.device).long()
    point, candidate, valid, candidate_mask, target = (
        value[ids] for value in (point, candidate, valid, candidate_mask, target)
    )
    torch.manual_seed(0)
    linear = LinearTriangleDiagnostic(point.shape[-1]).to(point.device)
    linear_final, linear_history = _train_classifier(
        linear, point, candidate, valid, candidate_mask, args.steps, args.learning_rate
    )
    torch.manual_seed(0)
    mlp = MLPTriangleDiagnostic(point.shape[-1]).to(point.device)
    mlp_final, mlp_history = _train_classifier(
        mlp, point, candidate, valid, candidate_mask, args.steps, args.learning_rate
    )
    torch.manual_seed(0)
    coordinate_final, coordinate_history = _train_coordinate(
        point, target, args.steps, args.learning_rate
    )
    coordinate_passed = coordinate_final["p95_mm"] < 1.0
    linear_passed = linear_final["valid_top1"] >= 0.95
    mlp_passed = mlp_final["valid_top1"] >= 0.95
    preliminary_diagnosis = (
        "all_diagnostic_heads_succeed"
        if coordinate_passed and linear_passed and mlp_passed
        else "fine_head_architecture_problem"
        if coordinate_passed and mlp_passed
        else "local_candidate_or_triangle_target_problem"
        if coordinate_passed
        else "frozen_stage_a_features_lack_fine_spatial_information"
    )
    summary = {
        "audit_passed": True,
        "smoke_only": args.steps < 50,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "sample_id": sample.get("sample_id"),
        "selected_shared_symmetry_element": selected_s,
        "point_count": count,
        "steps": args.steps,
        "linear_triangle_classifier": linear_final,
        "mlp_triangle_classifier": mlp_final,
        "coordinate_regression": coordinate_final,
        "diagnosis": (
            "smoke_completed_not_interpretable"
            if args.steps < 50 else preliminary_diagnosis
        ),
        "preliminary_diagnosis": preliminary_diagnosis,
    }
    (output / "fine_feature_capacity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    triangle_rows = [
        {"head": head, **row}
        for head, history in (("linear", linear_history), ("mlp", mlp_history))
        for row in history
    ]
    with (output / "triangle_classifier_capacity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(triangle_rows[0]))
        writer.writeheader(); writer.writerows(triangle_rows)
    with (output / "coordinate_regression_capacity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coordinate_history[0]))
        writer.writeheader(); writer.writerows(coordinate_history)
    (output / "fine_feature_capacity_report.md").write_text(
        "# Frozen fine-feature capacity\n\n"
        f"- smoke only: `{summary['smoke_only']}`\n"
        f"- linear valid top-1: `{linear_final['valid_top1']:.6f}`\n"
        f"- MLP valid top-1: `{mlp_final['valid_top1']:.6f}`\n"
        f"- coordinate p95: `{coordinate_final['p95_mm']:.6f} mm`\n"
        f"- diagnosis: `{summary['diagnosis']}`\n"
        f"- preliminary diagnosis: `{preliminary_diagnosis}`\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--triangle-target-tolerance-m", type=float, default=0.00015)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    register_all_modules()
    summary = run_capacity(args)
    print(json.dumps({"output_dir": str(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
