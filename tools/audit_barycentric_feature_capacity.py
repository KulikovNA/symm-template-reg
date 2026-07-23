#!/usr/bin/env python3
"""Measure barycentric capacity when the exact GT triangle is supplied."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from audit_fine_feature_capacity_v2 import _extract  # noqa: E402
from symm_template_reg.models import register_all_modules  # noqa: E402
from symm_template_reg.models.geometry.triangle_targets import closest_barycentric_on_triangles  # noqa: E402


def _triangle_descriptor(triangles):
    centroid = triangles.mean(1)
    centered = (triangles - centroid[:, None]).reshape(len(triangles), 9)
    cross = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1)
    norm = torch.linalg.vector_norm(cross, dim=-1, keepdim=True)
    normal = cross / norm.clamp_min(1e-12)
    area = 0.5 * norm
    edges = torch.stack(
        [torch.linalg.vector_norm(triangles[:, (i + 1) % 3] - triangles[:, i], dim=-1) for i in range(3)], -1
    )
    return torch.cat((centered, normal, area, edges), -1)


class BarycentricProbe(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3)
        )

    def forward(self, value):
        return torch.softmax(self.network(value), -1)


def run(args):
    data = _extract(args)
    triangles = data["vertices"][data["faces"][data["exact_face_ids"]]]
    target_surface = closest_barycentric_on_triangles(data["target"], triangles)
    target_bary = target_surface["barycentric"].detach()
    target_q = target_surface["points"].detach()
    inputs = torch.cat((data["point"], _triangle_descriptor(triangles)), -1)
    torch.manual_seed(args.seed)
    model = BarycentricProbe(inputs.shape[-1]).to(inputs.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    curve = []
    for step in range(args.max_steps + 1):
        bary = model(inputs)
        reconstructed = (bary[..., None] * triangles).sum(1)
        loss = torch.nn.functional.smooth_l1_loss(bary, target_bary)
        if step % args.eval_every == 0 or step == args.max_steps:
            bary_distance = torch.linalg.vector_norm(reconstructed - target_q, dim=-1)
            canonical_distance = torch.linalg.vector_norm(reconstructed - data["target"], dim=-1)
            surface_membership = (bary.sum(-1) - 1).abs()
            row = {
                "step": step, "loss": float(loss.detach()),
                "barycentric_reconstruction_p50_mm": float(torch.quantile(bary_distance.detach().float(), .50) * 1000),
                "barycentric_reconstruction_p95_mm": float(torch.quantile(bary_distance.detach().float(), .95) * 1000),
                "canonical_coordinate_p50_mm": float(torch.quantile(canonical_distance.detach().float(), .50) * 1000),
                "canonical_coordinate_p95_mm": float(torch.quantile(canonical_distance.detach().float(), .95) * 1000),
                "surface_membership_error": float(surface_membership.max().detach()),
            }
            curve.append(row)
            if row["canonical_coordinate_p95_mm"] <= .5:
                break
        if step == args.max_steps:
            break
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    final = curve[-1]
    summary = {
        "audit_passed": True,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "sample_id": data["sample_id"], "point_count": len(inputs),
        "exact_gt_triangle_supplied": True, "max_steps": args.max_steps,
        "smoke_only": args.max_steps <= 50, **final,
        "full_convergence_completed": args.max_steps >= 1500,
        "barycentric_capacity_passed": final["canonical_coordinate_p95_mm"] <= .5,
        "interpretation": "smoke_completed_not_interpretable" if args.max_steps <= 50 else (
            "intra_triangle_information_available" if final["canonical_coordinate_p95_mm"] <= .5
            else "fine_per_point_representation_lacks_local_canonical_information"
        ),
    }
    output = Path(args.output_dir).expanduser().resolve()
    (output / "barycentric_feature_capacity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "barycentric_feature_capacity_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0])); writer.writeheader(); writer.writerows(curve)
    (output / "barycentric_feature_capacity_report.md").write_text(
        "# Barycentric feature capacity\n\n"
        f"- exact GT triangle supplied: `True`\n- points: `{len(inputs)}`\n"
        f"- canonical p95: `{final['canonical_coordinate_p95_mm']:.6f} mm`\n"
        f"- gate passed: `{summary['barycentric_capacity_passed']}`\n"
        f"- interpretation: `{summary['interpretation']}`\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-steps", "--steps", type=int, default=1500); parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--max-points", type=int, default=1024); parser.add_argument("--all-points", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); register_all_modules()
    result = run(args); print(json.dumps({"output_dir": str(output), **result}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
