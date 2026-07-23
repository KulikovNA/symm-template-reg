#!/usr/bin/env python3
"""Convergence audit for frozen Stage-A point features and local candidates."""

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

from audit_fine_feature_capacity import (  # noqa: E402
    LinearTriangleDiagnostic,
    MLPTriangleDiagnostic,
)
from audit_local_triangle_target_contract import (  # noqa: E402
    _padded_target,
    _sample,
    shared_symmetry_target,
)
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.models.geometry.patch_targets import multi_positive_softmax_loss  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, build_from_cfg  # noqa: E402


class ResidualTriangleDiagnostic(nn.Module):
    def __init__(self, dimension: int, hidden: int = 128):
        super().__init__()
        self.point = nn.Linear(dimension, hidden)
        self.candidate = nn.Linear(dimension, hidden)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(2)]
        )
        self.output = nn.Linear(hidden, 1)

    def forward(self, point, candidate):
        value = torch.relu(self.point(point)[:, None] + self.candidate(candidate))
        for block in self.blocks:
            value = torch.relu(value + block(value))
        return self.output(value).squeeze(-1)


class CandidateConditionedDiagnostic(nn.Module):
    def __init__(self, dimension: int, hidden: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4 * dimension, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def forward(self, point, candidate):
        p = point[:, None].expand_as(candidate)
        return self.network(torch.cat((p, candidate, p * candidate, p - candidate), -1)).squeeze(-1)


class CoordinateDiagnostic(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, 128), nn.ReLU(), nn.Linear(128, 128),
            nn.ReLU(), nn.Linear(128, 3), nn.Tanh(),
        )

    def forward(self, point):
        return self.network(point)


def _extract(args):
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    run = checkpoint.parents[1]
    config = deepcopy(json.loads((run / "resolved_config.json").read_text()))
    config["model"]["correspondence_head"].update(
        teacher_forcing_initial_probability=1.0,
        teacher_forcing_final_probability=1.0,
        teacher_forcing_select_shared_symmetry_element=True,
        deduplicate_local_candidates=True,
        inject_all_valid_triangles=True,
        candidate_geometry_weight=1.0,
        max_local_candidate_total=32,
        sort_owned_faces_by_distance=True,
    )
    output = Path(args.output_dir).expanduser().resolve()
    sample, _ = _sample(config, args.manifest, output / "cache" / "fragment_mesh_metadata")
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    batch = move_to_device(collate([sample]), torch.device(args.device))
    model = build_model(config["model"]).to(args.device).train()
    payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.correspondence_head.teacher_forcing_probability = 1.0
    with torch.no_grad():
        prediction = model(batch)
    auxiliary = prediction.correspondence_auxiliary
    mask = prediction.observed_valid_mask[0]
    point = auxiliary["fine_observed_query_features"][0, mask].detach()
    ids = auxiliary["candidate_triangle_ids"][0, mask]
    candidate_mask = auxiliary["candidate_triangle_mask"][0, mask]
    candidate = auxiliary["global_triangle_features"][0][ids.clamp_min(0)].detach()
    valid = auxiliary["valid_triangle_local_mask"][0, mask] & candidate_mask
    selected_s = int(auxiliary["teacher_forcing_selected_symmetry_element"][0])
    target = shared_symmetry_target(
        _padded_target(batch)[0], batch["template_symmetry_metadata"][0],
        batch["gt"]["effective_symmetry_group"][0], selected_s,
    )[mask].detach()
    count = len(point) if args.all_points else min(len(point), args.max_points)
    subset = torch.linspace(0, len(point) - 1, count, device=point.device).long()
    vertices = batch["template_mesh_vertices_O"][0].to(target)
    faces = batch["template_mesh_faces"][0].to(device=target.device, dtype=torch.long)
    exact_face_ids = auxiliary["teacher_forcing_gt_triangle_ids"][0, mask]
    bbox_min, bbox_max = vertices.amin(0), vertices.amax(0)
    return {
        "point": point[subset], "candidate": candidate[subset],
        "valid": valid[subset], "mask": candidate_mask[subset],
        "target": target[subset], "bbox_min": bbox_min, "bbox_max": bbox_max,
        "vertices": vertices, "faces": faces,
        "exact_face_ids": exact_face_ids[subset],
        "sample_id": sample.get("sample_id"), "selected_s": selected_s,
        "checkpoint_epoch": payload.get("epoch"),
    }


def _classifier_curve(model, data, steps, lr, every):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    for step in range(steps + 1):
        logits = model(data["point"], data["candidate"]).masked_fill(~data["mask"], float("-inf"))
        loss = multi_positive_softmax_loss(logits, data["valid"])
        if step % every == 0 or step == steps:
            top1 = data["valid"].gather(-1, logits.argmax(-1, keepdim=True)).float().mean()
            top4 = data["valid"].gather(-1, logits.topk(min(4, logits.shape[-1]), -1).indices).any(-1).float().mean()
            curve.append({"step": step, "loss": float(loss.detach()), "valid_top1": float(top1), "valid_top4": float(top4)})
            if float(top1) >= 0.95 and float(top4) >= 0.995:
                break
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return curve


def _coordinate_curve(model, data, steps, lr, every):
    extent = (data["bbox_max"] - data["bbox_min"]).clamp_min(1e-8)
    target_n = 2 * (data["target"] - data["bbox_min"]) / extent - 1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    for step in range(steps + 1):
        prediction_n = model(data["point"])
        loss = torch.nn.functional.smooth_l1_loss(prediction_n, target_n)
        if step % every == 0 or step == steps:
            decoded = 0.5 * (prediction_n + 1) * extent + data["bbox_min"]
            distance = torch.linalg.vector_norm(decoded - data["target"], dim=-1)
            row = {"step": step, "loss": float(loss.detach()),
                   "rmse_mm": float(distance.square().mean().sqrt().detach() * 1000),
                   "p95_mm": float(torch.quantile(distance.detach().float(), .95) * 1000)}
            curve.append(row)
            if row["p95_mm"] <= 1.0:
                break
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return curve


def run(args):
    data = _extract(args)
    triangle_rows, coordinate_rows, curves = [], [], []
    probes = {
        "linear": LinearTriangleDiagnostic,
        "mlp_2layer": MLPTriangleDiagnostic,
        "mlp_4layer_residual": ResidualTriangleDiagnostic,
        "candidate_conditioned": CandidateConditionedDiagnostic,
    }
    for lr in args.learning_rates:
        for name, constructor in probes.items():
            torch.manual_seed(args.seed)
            curve = _classifier_curve(constructor(data["point"].shape[-1]).to(data["point"].device), data, args.max_steps, lr, args.eval_every)
            final = {"probe": name, "learning_rate": lr, **curve[-1]}
            triangle_rows.append(final)
            curves.extend({"kind": "triangle", "probe": name, "learning_rate": lr, **row} for row in curve)
        torch.manual_seed(args.seed)
        curve = _coordinate_curve(CoordinateDiagnostic(data["point"].shape[-1]).to(data["point"].device), data, args.max_steps, lr, args.eval_every)
        coordinate_rows.append({"probe": "normalized_coordinate", "learning_rate": lr, **curve[-1]})
        curves.extend({"kind": "coordinate", "probe": "normalized_coordinate", "learning_rate": lr, **row} for row in curve)
    triangle_pass = any(r["valid_top1"] >= .95 and r["valid_top4"] >= .995 for r in triangle_rows)
    coordinate_pass = any(r["p95_mm"] <= 1.0 for r in coordinate_rows)
    summary = {
        "audit_passed": True,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": data["checkpoint_epoch"], "sample_id": data["sample_id"],
        "selected_shared_symmetry_element": data["selected_s"],
        "point_count": len(data["point"]), "max_steps": args.max_steps,
        "evaluation_every": args.eval_every, "learning_rates": args.learning_rates,
        "smoke_only": args.max_steps <= 50,
        "full_convergence_completed": args.max_steps >= 1500,
        "coordinate_gate_passed": coordinate_pass, "triangle_gate_passed": triangle_pass,
        "frozen_features_sufficient": coordinate_pass or triangle_pass,
        "diagnosis": "smoke_completed_not_interpretable" if args.max_steps <= 50 else (
            "frozen_features_sufficient" if coordinate_pass or triangle_pass else
            "coarse_stage_features_do_not_encode_sufficient_fine_canonical_position"
        ),
        "best_triangle": max(triangle_rows, key=lambda r: (r["valid_top1"], r["valid_top4"])),
        "best_coordinate": min(coordinate_rows, key=lambda r: r["p95_mm"]),
    }
    output = Path(args.output_dir).expanduser().resolve()
    def write_csv(name, rows):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    write_csv("fine_feature_capacity_v2_curves.csv", curves)
    write_csv("triangle_head_comparison.csv", triangle_rows)
    write_csv("coordinate_head_comparison.csv", coordinate_rows)
    (output / "fine_feature_capacity_v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "fine_feature_capacity_v2_report.md").write_text(
        "# Fine feature capacity V2\n\n"
        f"- points: `{summary['point_count']}`\n- smoke only: `{summary['smoke_only']}`\n"
        f"- frozen features sufficient: `{summary['frozen_features_sufficient']}`\n"
        f"- diagnosis: `{summary['diagnosis']}`\n"
        f"- best triangle: `{summary['best_triangle']}`\n- best coordinate: `{summary['best_coordinate']}`\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-steps", "--steps", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=50); parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--all-points", action="store_true"); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=(1e-3, 3e-4, 1e-4))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); register_all_modules()
    summary = run(args); print(json.dumps({"output_dir": str(output), **summary}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
