#!/usr/bin/env python3
"""Audit exact surface projection guided by the F1 auxiliary coordinate."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from symm_template_reg.geometry import nearest_triangles_on_mesh  # noqa: E402
from symm_template_reg.models import register_all_modules  # noqa: E402
from symm_template_reg.models.geometry.triangle_targets import (  # noqa: E402
    closest_barycentric_on_triangles, triangle_target_sets,
)
from symm_template_reg.models.heads.coordinate_guided_surface_projection import (  # noqa: E402
    CoordinateGuidedSurfaceProjectionHead,
)


def _padded_ids_from_mask(face_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, face_count = face_mask.shape
    width = int(face_mask.sum(-1).max())
    values = torch.arange(face_count, device=face_mask.device)[None].expand(n, -1)
    sorted_ids = values.masked_fill(~face_mask, face_count).sort(-1).values[:, :width]
    valid = sorted_ids.lt(face_count)
    return sorted_ids.masked_fill(~valid, -1), valid


def _variant(
    name: str,
    q_aux: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    valid_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    valid_triangle_global: torch.Tensor,
    context: dict,
    projector: CoordinateGuidedSurfaceProjectionHead,
) -> tuple[dict, list[dict]]:
    result = projector(
        q_aux.unsqueeze(0), candidate_ids.unsqueeze(0), [vertices], [faces],
        valid_mask.unsqueeze(0), candidate_mask.unsqueeze(0),
    )
    projected = result["surface_correspondence_points_O"][0]
    selected = result["selected_triangle_ids"][0]
    local_top4 = result["candidate_distances"][0].topk(
        min(4, candidate_ids.shape[-1]), largest=False
    ).indices
    top4_global = candidate_ids.gather(-1, local_top4).clamp_min(0)
    valid_selected = valid_triangle_global.gather(
        -1, selected.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1) & selected.ge(0)
    valid_top4 = valid_triangle_global.gather(-1, top4_global).any(-1)
    candidate_global = candidate_ids.clamp_min(0)
    candidate_recall_rows = (
        valid_triangle_global.gather(-1, candidate_global) & candidate_mask
    ).any(-1)
    error = torch.linalg.vector_norm(projected - target, dim=-1)[valid_mask]
    selected_triangles = vertices[faces[selected[valid_mask]]]
    membership = closest_barycentric_on_triangles(
        projected[valid_mask], selected_triangles
    )["distances"]
    metrics = {
        "mode": name,
        **quantile_metrics_mm(error, "projected_correspondence"),
        "valid_triangle_set_top1": float(valid_selected[valid_mask].float().mean()),
        "valid_triangle_set_top4": float(valid_top4[valid_mask].float().mean()),
        "selected_triangle_valid_fraction": float(valid_selected[valid_mask].float().mean()),
        "triangle_candidate_set_recall": float(candidate_recall_rows[valid_mask].float().mean()),
        "surface_membership_p95_mm": float(torch.quantile(membership.float(), .95) * 1000),
        "mean_candidate_count": float(candidate_mask[valid_mask].sum(-1).float().mean()),
        "min_candidate_count": int(candidate_mask[valid_mask].sum(-1).min()),
        "max_candidate_count": int(candidate_mask[valid_mask].sum(-1).max()),
        "padded_candidate_slot_fraction": float(
            (~candidate_mask[valid_mask]).float().mean()
        ),
        "actual_invalid_triangle_id_fraction": float(
            ((candidate_ids[valid_mask] < 0) & candidate_mask[valid_mask]).float().mean()
        ),
    }
    pose_metrics, _ = pose_and_alignment_metrics(
        projected, observed, valid_mask, context["equivalent_pose"],
        context["metadata"], context["model"].weighted_procrustes,
        prefix="projection_pose",
    )
    metrics.update(pose_metrics)
    metrics["projection_alignment_p95_mm"] = metrics.pop(
        "projection_pose_alignment_p95_mm"
    )
    metrics["projection_alignment_rmse_mm"] = metrics.pop(
        "projection_pose_alignment_rmse_mm"
    )
    metrics["projection_correspondence_rank"] = metrics.pop(
        "projection_pose_correspondence_rank"
    )
    rows = []
    for point_id in torch.nonzero(valid_mask, as_tuple=False).flatten().tolist():
        rows.append({
            "mode": name, "point_id": point_id,
            "selected_triangle_id": int(selected[point_id]),
            "selected_triangle_valid": bool(valid_selected[point_id]),
            "candidate_contains_valid_triangle": bool(candidate_recall_rows[point_id]),
            "candidate_count": int(candidate_mask[point_id].sum()),
            "q_aux_to_surface_mm": float(result["distance_to_selected_triangle"][0, point_id] * 1000),
            "projected_correspondence_error_mm": float(torch.linalg.vector_norm(projected[point_id] - target[point_id]) * 1000),
            "surface_membership_mm": float(closest_barycentric_on_triangles(
                projected[point_id:point_id + 1], vertices[faces[selected[point_id:point_id + 1]]]
            )["distances"][0] * 1000),
            **{f"barycentric_{i}": float(result["analytic_barycentric_coordinates"][0, point_id, i]) for i in range(3)},
        })
    return metrics, rows


def projection_gate(predicted: dict, *, leakage: bool, nonfinite: bool) -> dict:
    thresholds = {
        "projected_correspondence_p95_mm": {"operator": "<=", "value": 1.0},
        "projection_pose_rotation_error_deg": {"operator": "<=", "value": 1.0},
        "projection_pose_translation_total_mm": {"operator": "<=", "value": 1.0},
        "projection_alignment_p95_mm": {"operator": "<=", "value": 1.0},
        "projection_correspondence_rank": {"operator": "==", "value": 3},
        "post_truncation_triangle_recall": {"operator": ">=", "value": .995},
        "surface_membership_p95_mm": {"operator": "<=", "value": .1},
        "target_leakage_detected": {"operator": "==", "value": False},
        "nonfinite_detected": {"operator": "==", "value": False},
    }
    checks = {
        "projected_correspondence_p95_mm": predicted["projected_correspondence_p95_mm"] <= 1.0 + 1e-6,
        "projection_pose_rotation_error_deg": predicted["projection_pose_rotation_error_deg"] <= 1.0 + 1e-6,
        "projection_pose_translation_total_mm": predicted["projection_pose_translation_total_mm"] <= 1.0 + 1e-6,
        "projection_alignment_p95_mm": predicted["projection_alignment_p95_mm"] <= 1.0 + 1e-6,
        "projection_correspondence_rank": predicted["projection_correspondence_rank"] == 3,
        "post_truncation_triangle_recall": predicted["post_truncation_triangle_recall"] >= .995 - 1e-6,
        "surface_membership_p95_mm": predicted["surface_membership_p95_mm"] <= .1 + 1e-6,
        "target_leakage_detected": not leakage,
        "nonfinite_detected": not nonfinite,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "stage_passed": not failures, "projection_ready": not failures,
        "next_surface_stage_allowed": not failures,
        "thresholds": thresholds, "checks": checks, "failures": failures,
        "target_leakage_detected": leakage, "nonfinite_detected": nonfinite,
        "predicted_candidate_metrics": predicted,
    }


def run(args) -> dict:
    output = Path(args.output_dir).expanduser().resolve()
    context = load_f1_audit_context(
        args.checkpoint, args.manifest, output, torch.device(args.device)
    )
    mask, vertices, faces = context["mask"], context["vertices"], context["faces"]
    q_aux, target = context["q_aux"], context["target"]
    target_sets = triangle_target_sets(
        target[mask], vertices, faces,
        tolerance_m=float(context["config"]["loss"]["joint_surface_correspondence_pose_v3"].get("triangle_target_tolerance_m", .00015)),
        point_chunk_size=256,
    )
    valid_global_full = torch.zeros((len(mask), len(faces)), dtype=torch.bool, device=mask.device)
    valid_global_full[mask] = target_sets["valid_triangle_mask"]
    global_nearest = nearest_triangles_on_mesh(q_aux[mask], vertices, faces, 4, point_chunk_size=256)["face_ids"]
    global_ids = torch.full((len(mask), 4), -1, dtype=torch.long, device=mask.device)
    global_ids[mask] = global_nearest
    global_mask = global_ids.ge(0)
    teacher_aux = context["teacher_aux"]
    valid_patch = teacher_aux["teacher_forcing_valid_patch_mask"][0]
    face_owner = teacher_aux["face_owner_patch_ids"][0]
    gt_patch_face_mask = valid_patch[:, face_owner]
    gt_patch_ids, gt_patch_mask = _padded_ids_from_mask(gt_patch_face_mask)
    predicted_aux = context["predicted_aux"]
    if predicted_aux is None:
        raise ValueError("predicted candidate forward has no correspondence auxiliary")
    predicted_ids = predicted_aux["candidate_triangle_ids"][0]
    predicted_mask = predicted_aux["candidate_triangle_mask"][0]
    projector = CoordinateGuidedSurfaceProjectionHead().to(mask.device)
    variants, all_rows = {}, []
    for name, ids, candidate_mask in (
        ("global_mesh_projection", global_ids, global_mask),
        ("gt_patch_candidate_projection", gt_patch_ids, gt_patch_mask),
        ("predicted_candidate_projection", predicted_ids, predicted_mask),
    ):
        metrics, rows = _variant(
            name, q_aux, target, context["observed"], mask, ids, candidate_mask,
            vertices, faces, valid_global_full, context, projector,
        )
        variants[name] = metrics; all_rows.extend(rows)
    # The global search considered every template face before retaining its
    # nearest four for top-k diagnostics.  Its candidate-set recall is exactly
    # one by construction; top-4 recall remains reported separately.
    variants["global_mesh_projection"]["triangle_candidate_set_recall"] = 1.0
    variants["global_mesh_projection"]["mean_candidate_count"] = float(len(faces))
    variants["global_mesh_projection"]["min_candidate_count"] = int(len(faces))
    variants["global_mesh_projection"]["max_candidate_count"] = int(len(faces))
    variants["global_mesh_projection"]["padded_candidate_slot_fraction"] = 0.0
    variants["global_mesh_projection"]["actual_invalid_triangle_id_fraction"] = 0.0
    predicted_patch_valid = teacher_aux["teacher_forcing_valid_patch_mask"][0]
    predicted_topk = predicted_aux["selected_topk_patch_ids"][0]
    predicted_patch_top4 = predicted_patch_valid.gather(-1, predicted_topk).any(-1)
    variants["predicted_candidate_projection"]["predicted_patch_valid_set_top4_recall"] = float(
        predicted_patch_top4[mask].float().mean()
    )
    variants["predicted_candidate_projection"]["post_truncation_triangle_recall"] = variants[
        "predicted_candidate_projection"
    ]["triangle_candidate_set_recall"]
    leakage_path = context["config"].get("target_leakage_policy", {}).get("audit_path")
    leakage = True
    if leakage_path and Path(str(leakage_path)).is_file():
        leakage = bool(json.loads(Path(str(leakage_path)).read_text())["target_leakage_detected"])
    numeric_values = [v for metrics in variants.values() for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    nonfinite = any(not math.isfinite(float(value)) for value in numeric_values)
    gate = projection_gate(variants["predicted_candidate_projection"], leakage=leakage, nonfinite=nonfinite)
    global_good = variants["global_mesh_projection"]["projected_correspondence_p95_mm"] <= 1.0
    gt_good = variants["gt_patch_candidate_projection"]["projected_correspondence_p95_mm"] <= 1.0
    predicted_good = variants["predicted_candidate_projection"]["projected_correspondence_p95_mm"] <= 1.0
    interpretation = (
        "predicted_candidate_projection_good_learned_triangle_and_barycentric_heads_unnecessary"
        if predicted_good else
        "coarse_candidate_problem" if gt_good and not predicted_good else
        "candidate_construction_or_local_ambiguity_problem" if global_good else
        "q_aux_accuracy_problem"
    )
    summary = {
        "audit_passed": True, "checkpoint": str(context["checkpoint"]),
        "checkpoint_epoch": context["checkpoint_payload"].get("epoch"),
        "sample_id": context["sample"].get("sample_id"), "point_count": int(mask.sum()),
        "selected_shared_symmetry_element": context["selected_symmetry_element"],
        "learned_barycentric_head_used": False,
        "predicted_mode_is_inference_valid": True,
        "variants": variants, "interpretation": interpretation,
        "projection_gate_passed": gate["stage_passed"],
    }
    (output / "coordinate_projection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "coordinate_projection_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    with (output / "coordinate_projection_per_point.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    lines = ["# Coordinate-guided surface projection", "", f"- interpretation: `{interpretation}`", f"- projection gate passed: `{gate['stage_passed']}`", ""]
    for name, metrics in variants.items():
        lines.extend([f"## {name}", "", f"- correspondence p95: `{metrics['projected_correspondence_p95_mm']:.6f} mm`", f"- triangle candidate-set recall: `{metrics['triangle_candidate_set_recall']:.6f}`", f"- pose: `{metrics['projection_pose_rotation_error_deg']:.6f} deg`, `{metrics['projection_pose_translation_total_mm']:.6f} mm`", ""])
    (output / "coordinate_projection_report.md").write_text("\n".join(lines) + "\n")
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
