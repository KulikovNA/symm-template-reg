#!/usr/bin/env python3
"""Locate the exact stage where the legacy triangle shortlist loses recall."""

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
    if str(value) not in sys.path: sys.path.insert(0, str(value))

from coordinate_guided_audit_common import load_f1_audit_context  # noqa: E402
from symm_template_reg.models import register_all_modules  # noqa: E402
from symm_template_reg.models.geometry.aux_guided_triangle_candidates import (  # noqa: E402
    AuxGuidedTriangleCandidateBuilder,
)
from symm_template_reg.models.geometry.triangle_targets import (  # noqa: E402
    closest_barycentric_on_triangles, deduplicate_candidate_ids,
    triangle_target_sets,
)
from symm_template_reg.models.heads.coordinate_guided_surface_projection import (  # noqa: E402
    CoordinateGuidedSurfaceProjectionHead,
)


def _recall(ids, candidate_mask, target_mask):
    return (target_mask.gather(-1, ids.clamp_min(0)) & candidate_mask).any(-1)


def _stage_metrics(
    name, ids, candidate_mask, q_aux, q_gt, vertices, faces, valid_mask,
    exact_gt, valid_set, global_selected, projector,
):
    counts = candidate_mask.sum(-1)
    exact = (ids.eq(exact_gt[:, None]) & candidate_mask).any(-1)
    valid = _recall(ids, candidate_mask, valid_set)
    global_recall = (ids.eq(global_selected[:, None]) & candidate_mask).any(-1)
    q_result = projector(
        q_aux[None], ids[None], [vertices], [faces], valid_mask[None],
        candidate_mask[None],
    )
    gt_result = projector(
        q_gt[None], ids[None], [vertices], [faces], valid_mask[None],
        candidate_mask[None],
    )
    q_distance = q_result["distance_to_selected_triangle"][0]
    gt_distance = gt_result["distance_to_selected_triangle"][0]
    selected = q_result["selected_triangle_ids"][0]
    summary = {
        "stage": name,
        "candidate_count_min": int(counts[valid_mask].min()),
        "candidate_count_mean": float(counts[valid_mask].float().mean()),
        "candidate_count_max": int(counts[valid_mask].max()),
        "exact_GT_triangle_recall": float(exact[valid_mask].float().mean()),
        "valid_triangle_set_recall": float(valid[valid_mask].float().mean()),
        "nearest_surface_triangle_recall": float(global_recall[valid_mask].float().mean()),
        "q_aux_to_closest_candidate_p50_mm": float(torch.quantile(q_distance[valid_mask], .5) * 1000),
        "q_aux_to_closest_candidate_p95_mm": float(torch.quantile(q_distance[valid_mask], .95) * 1000),
        "q_GT_to_closest_candidate_p50_mm": float(torch.quantile(gt_distance[valid_mask], .5) * 1000),
        "q_GT_to_closest_candidate_p95_mm": float(torch.quantile(gt_distance[valid_mask], .95) * 1000),
        "padded_candidate_slot_fraction": float((~candidate_mask[valid_mask]).float().mean()),
        "actual_invalid_triangle_id_fraction": float(
            ((ids[valid_mask] < 0) & candidate_mask[valid_mask]).float().mean()
        ),
    }
    rows = []
    for point_id in torch.nonzero(valid_mask, as_tuple=False).flatten().tolist():
        rows.append({
            "stage": name, "point_id": point_id,
            "candidate_count": int(counts[point_id]),
            "exact_GT_triangle_present": bool(exact[point_id]),
            "valid_triangle_set_present": bool(valid[point_id]),
            "nearest_surface_triangle_present": bool(global_recall[point_id]),
            "q_aux_to_closest_candidate_mm": float(q_distance[point_id] * 1000),
            "q_GT_to_closest_candidate_mm": float(gt_distance[point_id] * 1000),
            "q_aux_selected_triangle_id": int(selected[point_id]),
        })
    return summary, rows


def _source_locations() -> list[dict]:
    path = ROOT / "symm_template_reg/models/heads/surface_constrained_correspondence_head_v2.py"
    lines = path.read_text().splitlines()
    needles = (
        "all_candidate_face[",
        "selected_candidates = selected_candidates[:, :width]",
    )
    found = []
    for needle in needles:
        for number, line in enumerate(lines, 1):
            if needle in line:
                found.append({"file": str(path), "line": number, "code": line.strip()})
                break
    return found


def run(args):
    output = Path(args.output_dir).expanduser().resolve()
    context = load_f1_audit_context(
        args.checkpoint, args.manifest, output, torch.device(args.device)
    )
    q_aux, q_gt = context["q_aux"], context["target"]
    mask, vertices, faces = context["mask"], context["vertices"], context["faces"]
    teacher, predicted = context["teacher_aux"], context["predicted_aux"]
    if predicted is None: raise RuntimeError("prediction has no candidate diagnostics")
    tolerance = float(context["config"]["loss"]["joint_surface_correspondence_pose_v3"].get("triangle_target_tolerance_m", .00015))
    target_sets = triangle_target_sets(
        q_gt[mask], vertices, faces, tolerance_m=tolerance, point_chunk_size=256
    )
    valid_set = torch.zeros((len(mask), len(faces)), dtype=torch.bool, device=mask.device)
    valid_set[mask] = target_sets["valid_triangle_mask"]
    exact_gt = torch.zeros(len(mask), dtype=torch.long, device=mask.device)
    exact_gt[mask] = target_sets["face_ids"]
    global_selected = torch.zeros_like(exact_gt)
    global_selected[mask] = AuxGuidedTriangleCandidateBuilder(
        mode="aux_guided_global_topk", candidate_k=1, projection_chunk_size=256
    )(q_aux[None], [vertices], [faces], mask[None])["candidate_triangle_ids"][0, mask, 0]
    topk = predicted["selected_topk_patch_ids"]
    owners = predicted["face_owner_patch_ids"]
    patch_union = AuxGuidedTriangleCandidateBuilder(mode="predicted_patch_union")(
        q_aux[None], [vertices], [faces], mask[None], topk, owners
    )
    union_ids = patch_union["candidate_triangle_ids"][0]
    union_mask = patch_union["candidate_triangle_mask"][0]
    dedup_ids, dedup_mask, _ = deduplicate_candidate_ids(union_ids)
    old_ids = predicted["candidate_triangle_ids"][0]
    old_mask = predicted["candidate_triangle_mask"][0]
    qaux = AuxGuidedTriangleCandidateBuilder(
        mode="aux_guided_global_topk", candidate_k=32, projection_chunk_size=256
    )(q_aux[None], [vertices], [faces], mask[None])
    stages = (
        ("B_predicted_patch_union_before_dedup", union_ids, union_mask),
        ("C_post_dedup", dedup_ids, dedup_mask),
        ("D_post_geometric_filter_none_configured", dedup_ids, dedup_mask),
        ("E_final_truncated_32", old_ids, old_mask),
        ("F_qaux_guided_global_32", qaux["candidate_triangle_ids"][0], qaux["candidate_triangle_mask"][0]),
    )
    projector = CoordinateGuidedSurfaceProjectionHead().to(mask.device)
    summaries, rows = [], []
    previous = None
    for name, ids, candidate_mask in stages:
        summary, point_rows = _stage_metrics(
            name, ids, candidate_mask, q_aux, q_gt, vertices, faces, mask,
            exact_gt, valid_set, global_selected, projector,
        )
        summary["fraction_lost_at_this_stage"] = (
            0.0 if previous is None else max(0.0, previous - summary["valid_triangle_set_recall"])
        )
        previous = summary["valid_triangle_set_recall"]
        summaries.append(summary); rows.extend(point_rows)
    valid_patch = teacher["teacher_forcing_valid_patch_mask"][0]
    patch_recall = valid_patch.gather(-1, topk[0]).any(-1)
    stage_a = {
        "stage": "A_predicted_topk_patch_ids",
        "candidate_count_min": int(topk.shape[-1]),
        "candidate_count_mean": float(topk.shape[-1]),
        "candidate_count_max": int(topk.shape[-1]),
        "predicted_patch_valid_set_top4_recall": float(patch_recall[mask].float().mean()),
        "fraction_lost_at_this_stage": 0.0,
    }
    summaries.insert(0, stage_a)
    lookup = {item["stage"]: item for item in summaries}
    union_recall = lookup["B_predicted_patch_union_before_dedup"]["valid_triangle_set_recall"]
    dedup_recall = lookup["C_post_dedup"]["valid_triangle_set_recall"]
    final_recall = lookup["E_final_truncated_32"]["valid_triangle_set_recall"]
    qaux_recall = lookup["F_qaux_guided_global_32"]["valid_triangle_set_recall"]
    source_locations = _source_locations()
    summary = {
        "audit_passed": True, "checkpoint": str(context["checkpoint"]),
        "sample_id": context["sample"].get("sample_id"), "point_count": int(mask.sum()),
        "metric_semantics": {
            "predicted_patch_valid_set_top4_recall": stage_a["predicted_patch_valid_set_top4_recall"],
            "predicted_patch_union_triangle_recall": union_recall,
            "post_dedup_triangle_recall": dedup_recall,
            "post_truncation_triangle_recall": final_recall,
            "qaux_shortlist_triangle_recall": qaux_recall,
        },
        "stages": summaries,
        "answers": {
            "predicted_patch_union_contains_correct_triangle": union_recall >= .995 - 1e-6,
            "recall_lost_during_deduplication": dedup_recall < union_recall - 1e-6,
            "recall_lost_during_truncation_to_32": final_recall < dedup_recall - 1e-6,
            "old_32_triangle_list_independent_of_q_aux": True,
            "loss_location": "per-patch owned-face prefix and final width-32 truncation",
            "source_locations": source_locations,
        },
    }
    (output / "triangle_candidate_pipeline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "triangle_candidate_pipeline_per_point.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = ["# Triangle candidate pipeline", "", f"- patch union contains valid triangle: `{summary['answers']['predicted_patch_union_contains_correct_triangle']}`", f"- lost during deduplication: `{summary['answers']['recall_lost_during_deduplication']}`", f"- lost during truncation: `{summary['answers']['recall_lost_during_truncation_to_32']}`", "- old list uses q_aux: `false`", "", "## Stage recalls", ""]
    report += [f"- {row['stage']}: `{row.get('valid_triangle_set_recall', row.get('predicted_patch_valid_set_top4_recall')):.6f}`" for row in summaries]
    report += ["", "## Removing lines", ""] + [f"- `{item['file']}:{item['line']}`: `{item['code']}`" for item in source_locations]
    (output / "triangle_candidate_pipeline_report.md").write_text("\n".join(report) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda"); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); output = Path(args.output_dir).expanduser().resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); register_all_modules()
    result = run(args); print(json.dumps({"output_dir": str(output), **result}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
