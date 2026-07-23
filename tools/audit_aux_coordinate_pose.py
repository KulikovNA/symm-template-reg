#!/usr/bin/env python3
"""Measure pose obtained directly from F1 auxiliary canonical coordinates."""

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

from coordinate_guided_audit_common import (  # noqa: E402
    load_f1_audit_context, pose_and_alignment_metrics, quantile_metrics_mm,
)
from symm_template_reg.models import register_all_modules  # noqa: E402
from symm_template_reg.models.pose.pose_representation import transform_points  # noqa: E402


def run(args) -> dict:
    output = Path(args.output_dir).expanduser().resolve()
    context = load_f1_audit_context(
        args.checkpoint, args.manifest, output, torch.device(args.device)
    )
    mask = context["mask"]
    q = context["q_aux"]
    target = context["target"]
    coordinate_distance = torch.linalg.vector_norm(q - target, dim=-1)[mask]
    coordinate_metrics = quantile_metrics_mm(coordinate_distance, "aux_coordinate")
    pose_metrics, diagnostics = pose_and_alignment_metrics(
        q, context["observed"], mask, context["equivalent_pose"],
        context["metadata"], context["model"].weighted_procrustes,
        prefix="aux_pose",
    )
    # Required public names use aux_alignment rather than aux_pose_alignment.
    pose_metrics["aux_alignment_rmse_mm"] = pose_metrics.pop("aux_pose_alignment_rmse_mm")
    pose_metrics["aux_alignment_p95_mm"] = pose_metrics.pop("aux_pose_alignment_p95_mm")
    pose_metrics["aux_correspondence_rank"] = pose_metrics.pop("aux_pose_correspondence_rank")
    pose_metrics["aux_procrustes_rank_valid"] = pose_metrics.pop("aux_pose_procrustes_rank_valid")
    pose_metrics["aux_covariance_eigenvalues"] = pose_metrics.pop("aux_pose_covariance_eigenvalues")
    reconstructed = transform_points(
        diagnostics["pose"].unsqueeze(0), q.unsqueeze(0)
    )[0]
    rows = []
    valid_ids = torch.nonzero(mask, as_tuple=False).flatten()
    for point_id in valid_ids.tolist():
        rows.append({
            "point_id": point_id,
            **{f"q_aux_{axis}_O": float(q[point_id, i]) for i, axis in enumerate("xyz")},
            **{f"q_target_{axis}_O": float(target[point_id, i]) for i, axis in enumerate("xyz")},
            **{f"p_observed_{axis}_C": float(context["observed"][point_id, i]) for i, axis in enumerate("xyz")},
            "coordinate_error_mm": float(torch.linalg.vector_norm(q[point_id] - target[point_id]) * 1000),
            "alignment_error_mm": float(torch.linalg.vector_norm(reconstructed[point_id] - context["observed"][point_id]) * 1000),
        })
    summary = {
        "audit_passed": True,
        "checkpoint": str(context["checkpoint"]),
        "checkpoint_epoch": context["checkpoint_payload"].get("epoch"),
        "sample_id": context["sample"].get("sample_id"),
        "point_count": len(rows),
        "selected_shared_symmetry_element": context["selected_symmetry_element"],
        "pose_source": "q_aux_uniform_weighted_procrustes",
        "main_triangle_barycentric_pose_not_used": True,
        **coordinate_metrics, **pose_metrics,
    }
    (output / "aux_coordinate_pose_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "aux_coordinate_pose_per_point.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "aux_coordinate_pose_report.md").write_text(
        "# Auxiliary-coordinate pose audit\n\n"
        f"- points: `{len(rows)}`\n"
        f"- shared symmetry element: `{context['selected_symmetry_element']}`\n"
        f"- coordinate p95: `{summary['aux_coordinate_p95_mm']:.6f} mm`\n"
        f"- pose rotation: `{summary['aux_pose_rotation_error_deg']:.6f} deg`\n"
        f"- pose translation: `{summary['aux_pose_translation_total_mm']:.6f} mm`\n"
        f"- alignment p95: `{summary['aux_alignment_p95_mm']:.6f} mm`\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); register_all_modules()
    result = run(args); print(json.dumps({"output_dir": str(output), **result}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
