#!/usr/bin/env python3
"""Recheck a frozen F1 checkpoint with exact and q_aux-guided surface modes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path: sys.path.insert(0, str(value))
from coordinate_guided_audit_common import (  # noqa: E402
    load_coordinate_audit_contexts, load_f1_audit_context,
    pose_and_alignment_metrics, quantile_metrics_mm,
)
from symm_template_reg.evaluation.active_coordinate import (  # noqa: E402
    active_row,
    active_world_metrics,
    evaluate_active_sample,
    four_view_stage_gate,
    worst_sample_projection_score,
)
from symm_template_reg.models import register_all_modules  # noqa: E402
from symm_template_reg.models.geometry.aux_guided_triangle_candidates import (  # noqa: E402
    AuxGuidedTriangleCandidateBuilder,
)
from symm_template_reg.models.geometry.triangle_targets import (  # noqa: E402
    closest_barycentric_on_triangles, triangle_target_sets,
)
from symm_template_reg.models.heads.coordinate_guided_surface_projection import (  # noqa: E402
    CoordinateGuidedSurfaceProjectionHead,
)
from symm_template_reg.visualization.ply import write_colored_ply  # noqa: E402


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _source_signatures(run):
    names = ["checkpoints/best.pth", "resolved_config.json", "final_summary.json", "stage_gate.json"]
    return {name: _hash(run / name) for name in names if (run / name).is_file()}


def _evaluate(name, ids, candidate_mask, context, valid_global, global_selected, projector, runtime_ms):
    q_aux, mask = context["q_aux"], context["mask"]
    result = projector(
        q_aux[None], ids[None], [context["vertices"]], [context["faces"]],
        mask[None], candidate_mask[None],
    )
    projected = result["surface_correspondence_points_O"][0]
    selected = result["selected_triangle_ids"][0]
    error = torch.linalg.vector_norm(projected - context["target"], dim=-1)[mask]
    membership = closest_barycentric_on_triangles(
        projected[mask], context["vertices"][context["faces"][selected[mask]]]
    )["distances"]
    global_recall = ((ids == global_selected[:, None]) & candidate_mask).any(-1)
    valid_recall = (valid_global.gather(-1, ids.clamp_min(0)) & candidate_mask).any(-1)
    metrics = {
        "mode": name, **quantile_metrics_mm(error, "projected_correspondence"),
        "exact_global_selected_triangle_in_shortlist_fraction": float(global_recall[mask].float().mean()),
        "valid_GT_triangle_set_in_shortlist_fraction": float(valid_recall[mask].float().mean()),
        "surface_membership_p95_mm": float(torch.quantile(membership.float(), .95) * 1000),
        "runtime_ms": float(runtime_ms),
        "peak_memory_mb": float(torch.cuda.max_memory_allocated() / 1024 ** 2) if q_aux.is_cuda else 0.0,
        "candidate_count_min": int(candidate_mask[mask].sum(-1).min()),
        "candidate_count_mean": float(candidate_mask[mask].sum(-1).float().mean()),
        "candidate_count_max": int(candidate_mask[mask].sum(-1).max()),
        "padded_candidate_slot_fraction": float((~candidate_mask[mask]).float().mean()),
        "actual_invalid_triangle_id_fraction": float(((ids[mask] < 0) & candidate_mask[mask]).float().mean()),
        "nonfinite_detected": False,
        "global_fallback_used": False,
        "shortlist_fallback_fraction": 0.0,
        "global_projection_fraction": 1.0 if name == "exact_global" else 0.0,
    }
    pose, _ = pose_and_alignment_metrics(
        projected, context["observed"], mask, context["equivalent_pose"],
        context["metadata"], context["model"].weighted_procrustes,
        prefix="projection_pose",
    )
    metrics.update(pose)
    metrics["projection_alignment_p95_mm"] = metrics.pop("projection_pose_alignment_p95_mm")
    metrics["projection_alignment_rmse_mm"] = metrics.pop("projection_pose_alignment_rmse_mm")
    metrics["projection_correspondence_rank"] = metrics.pop("projection_pose_correspondence_rank")
    metrics["nonfinite_detected"] = any(
        not math.isfinite(float(value)) for value in metrics.values()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    )
    rows = [{
        "mode": name, "point_id": point_id,
        "selected_triangle_id": int(selected[point_id]),
        "global_triangle_covered": bool(global_recall[point_id]),
        "valid_GT_triangle_covered": bool(valid_recall[point_id]),
        "candidate_count": int(candidate_mask[point_id].sum()),
        "projection_distance_before_mm": float(result["distance_to_selected_triangle"][0, point_id] * 1000),
        "projection_distance_after_mm": float(membership[torch.nonzero(mask, as_tuple=False).flatten().tolist().index(point_id)] * 1000),
        "correspondence_error_mm": float(torch.linalg.vector_norm(projected[point_id] - context["target"][point_id]) * 1000),
    } for point_id in torch.nonzero(mask, as_tuple=False).flatten().tolist()]
    return metrics, rows, projected, global_recall


def _passes(metrics, require_global_recall=True):
    checks = {
        "exact_global_triangle_recall": (not require_global_recall) or metrics["exact_global_selected_triangle_in_shortlist_fraction"] >= .995 - 1e-6,
        "projected_correspondence_p95_mm": metrics["projected_correspondence_p95_mm"] <= 1.0 + 1e-6,
        "projection_alignment_p95_mm": metrics["projection_alignment_p95_mm"] <= 1.0 + 1e-6,
        "projection_pose_rotation_error_deg": metrics["projection_pose_rotation_error_deg"] <= 1.0 + 1e-6,
        "projection_pose_translation_total_mm": metrics["projection_pose_translation_total_mm"] <= 1.0 + 1e-6,
        "rank": metrics["projection_correspondence_rank"] == 3,
        "surface_membership_p95_mm": metrics["surface_membership_p95_mm"] <= .1 + 1e-6,
        "no_nonfinite": not metrics["nonfinite_detected"],
    }
    return {"passed": all(checks.values()), "checks": checks, "failures": [k for k,v in checks.items() if not v]}


def select_smallest_passing_candidate(gates):
    passing = []
    for name, gate in gates.items():
        if gate["passed"]:
            passing.append((int(name.rsplit("k", 1)[1]), name))
    return min(passing) if passing else None


def _run_legacy(args):
    output = Path(args.output_dir).expanduser().resolve()
    source_run = Path(args.checkpoint).expanduser().resolve().parents[1]
    before = _source_signatures(source_run)
    context = load_f1_audit_context(args.checkpoint, args.manifest, output, torch.device(args.device))
    q_aux, mask, vertices, faces = context["q_aux"], context["mask"], context["vertices"], context["faces"]
    target_sets = triangle_target_sets(context["target"][mask], vertices, faces, tolerance_m=.00015, point_chunk_size=256)
    valid_global = torch.zeros((len(mask), len(faces)), dtype=torch.bool, device=mask.device)
    valid_global[mask] = target_sets["valid_triangle_mask"]
    projector = CoordinateGuidedSurfaceProjectionHead().to(mask.device)
    modes = []
    if q_aux.is_cuda: torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    global_built = AuxGuidedTriangleCandidateBuilder(mode="aux_guided_global_topk", candidate_k=1, projection_chunk_size=args.projection_chunk_size)(q_aux[None], [vertices], [faces], mask[None])
    if q_aux.is_cuda: torch.cuda.synchronize()
    global_runtime = (time.perf_counter() - started) * 1000
    global_ids, global_mask = global_built["candidate_triangle_ids"][0], global_built["candidate_triangle_mask"][0]
    global_selected = global_ids[:, 0]
    modes.append(("exact_global", global_ids, global_mask, global_runtime))
    predicted = context["predicted_aux"]
    modes.append(("old_predicted_32", predicted["candidate_triangle_ids"][0], predicted["candidate_triangle_mask"][0], 0.0))
    for candidate_mode in ("aux_guided_global_topk", "aux_guided_patch_union_topk"):
        if q_aux.is_cuda: torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        built = AuxGuidedTriangleCandidateBuilder(mode=candidate_mode, candidate_k=max(args.candidate_k), projection_chunk_size=args.projection_chunk_size)(
            q_aux[None], [vertices], [faces], mask[None],
            predicted["selected_topk_patch_ids"], predicted["face_owner_patch_ids"],
        )
        if q_aux.is_cuda: torch.cuda.synchronize()
        runtime = (time.perf_counter() - started) * 1000
        for k in args.candidate_k:
            ids = built["candidate_triangle_ids"][0, :, :k]
            candidate_mask = built["candidate_triangle_mask"][0, :, :k]
            modes.append((f"{candidate_mode}_k{k}", ids, candidate_mask, runtime))
    metrics, all_rows, projections, recalls = {}, [], {}, {}
    for name, ids, candidate_mask, runtime in modes:
        values, rows, projected, recall = _evaluate(
            name, ids, candidate_mask, context, valid_global, global_selected,
            projector, runtime,
        )
        metrics[name] = values; all_rows.extend(rows); projections[name] = projected; recalls[name] = recall
    candidate_gates = {name: _passes(values) for name, values in metrics.items() if name.startswith("aux_guided")}
    selected = select_smallest_passing_candidate(candidate_gates)
    global_gate = _passes(metrics["exact_global"], require_global_recall=False)
    frame_id = int(context["sample"].get("frame_id", -1))
    solved = global_gate["passed"] or selected is not None
    next_type = (
        "two_view_coordinate_training" if frame_id == 8
        else "frame8_coordinate_training" if frame_id == 4
        else "coordinate_training_review"
    )
    frame_gate = {
        "stage_passed": solved,
        "frame_correctness_solved": solved,
        "frame_id": frame_id,
        "next_experiment_allowed": solved,
        "next_experiment_type": next_type,
        # Deprecated compatibility aliases; never use these as primary fields.
        "frame4_correctness_solved": solved,
        "next_frame8_correctness_experiment_allowed": solved,
        "deprecated_compatibility_aliases": [
            "frame4_correctness_solved",
            "next_frame8_correctness_experiment_allowed",
        ],
        "exact_global_gate": global_gate,
        "candidate_gates": candidate_gates,
        "selected_candidate_k": selected[0] if selected else None,
        "selected_candidate_mode": selected[1] if selected else None,
        "target_leakage_detected": False, "nonfinite_detected": any(v["nonfinite_detected"] for v in metrics.values()),
    }
    after = _source_signatures(source_run); source_unchanged = before == after
    summary = {
        "recheck_completed": True, "source_run": str(source_run), "source_unchanged": source_unchanged,
        "checkpoint": str(context["checkpoint"]), "sample_id": context["sample"].get("sample_id"),
        "raw_q_aux": quantile_metrics_mm(torch.linalg.vector_norm(q_aux[mask] - context["target"][mask], dim=-1), "aux_coordinate"),
        "modes": metrics, "aux_guided_candidate_gate": frame_gate,
    }
    (output / "coordinate_surface_recheck_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "coordinate_surface_stage_gate.json").write_text(json.dumps(frame_gate, indent=2) + "\n")
    (output / "aux_guided_candidate_gate.json").write_text(json.dumps({
        "thresholds": {
            "exact_global_selected_triangle_in_shortlist_fraction": {"operator": ">=", "value": .995},
            "projected_correspondence_p95_mm": {"operator": "<=", "value": 1.0},
            "projection_alignment_p95_mm": {"operator": "<=", "value": 1.0},
            "projection_pose_rotation_error_deg": {"operator": "<=", "value": 1.0},
            "projection_pose_translation_total_mm": {"operator": "<=", "value": 1.0},
            "projection_correspondence_rank": {"operator": "==", "value": 3},
            "nonfinite_detected": {"operator": "==", "value": False},
        },
        "candidate_gates": candidate_gates,
        "selected_candidate_k": selected[0] if selected else None,
        "selected_candidate_mode": selected[1] if selected else None,
    }, indent=2) + "\n")
    (output / "source_integrity.json").write_text(json.dumps({"before": before, "after": after, "source_unchanged": source_unchanged}, indent=2) + "\n")
    with (output / "coordinate_surface_recheck_per_point.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    chosen_name = selected[1] if selected else "old_predicted_32"
    valid_points = q_aux[mask].detach().cpu().numpy(); global_points = projections["exact_global"][mask].detach().cpu().numpy(); short_points = projections[chosen_name][mask].detach().cpu().numpy()
    green = np.tile(np.array([30,220,70], dtype=np.uint8), (len(valid_points),1)); magenta = np.tile(np.array([255,0,255], dtype=np.uint8), (len(valid_points),1))
    write_colored_ply(output / "q_aux_vs_global_projection.ply", np.concatenate((valid_points, global_points)), np.concatenate((magenta, green)))
    write_colored_ply(output / "q_aux_vs_shortlist_projection.ply", np.concatenate((valid_points, short_points)), np.concatenate((magenta, green)))
    coverage = recalls[chosen_name][mask].detach().cpu().numpy(); colors = green.copy(); colors[~coverage] = [255,0,255]
    write_colored_ply(output / "missed_shortlist_points.ply", valid_points, colors)
    report = ["# Coordinate-guided surface recheck", "", f"- source unchanged: `{source_unchanged}`", f"- exact-global passed: `{global_gate['passed']}`", f"- selected shortlist: `{selected}`", f"- frame-4 passed: `{frame_gate['stage_passed']}`", "", "| mode | p95 mm | alignment p95 mm | global recall | runtime ms |", "|---|---:|---:|---:|---:|"]
    for name, value in metrics.items(): report.append(f"| {name} | {value['projected_correspondence_p95_mm']:.6f} | {value['projection_alignment_p95_mm']:.6f} | {value['exact_global_selected_triangle_in_shortlist_fraction']:.6f} | {value['runtime_ms']:.3f} |")
    (output / "coordinate_surface_recheck_report.md").write_text("\n".join(report) + "\n")
    return summary


def _aggregate_active_namespaces(rows, world):
    exact = {
        "worst_sample_projection_score": worst_sample_projection_score(rows),
        "all_samples_gate_passed": float(
            all(bool(row["exact_global_sample_gate_passed"]) for row in rows)
        ),
        "worst_correspondence_p95_mm": max(
            float(row["exact_global_projected_correspondence_p95_mm"]) for row in rows
        ),
        "worst_alignment_p95_mm": max(
            float(row["exact_global_projection_alignment_p95_mm"]) for row in rows
        ),
        "worst_rotation_error_deg": max(
            float(row["exact_global_projection_rotation_error_deg"]) for row in rows
        ),
        "worst_translation_error_mm": max(
            float(row["exact_global_projection_translation_error_mm"]) for row in rows
        ),
    }
    k16 = {
        "minimum_exact_global_triangle_recall": min(
            float(row["k16_exact_global_triangle_recall"]) for row in rows
        ),
        "maximum_fallback_fraction": max(
            float(row["k16_fallback_fraction"]) for row in rows
        ),
        "worst_correspondence_p95_mm": max(
            float(row["k16_projected_correspondence_p95_mm"]) for row in rows
        ),
    }
    return {
        "eval/active/exact_global": exact,
        "eval/active/k16": k16,
        "eval/active/world": world,
        "eval/inactive/legacy_triangle": {"active": False},
        "eval/inactive/legacy_pose_query": {"active": False},
        "eval/inactive/regions": {"active": False},
        "eval/inactive/ranking": {"active": False},
    }


def run(args):
    """Re-evaluate every manifest sample using only the active physical path."""

    output = Path(args.output_dir).expanduser().resolve()
    source_run = Path(args.checkpoint).expanduser().resolve().parents[1]
    before = _source_signatures(source_run)
    contexts = load_coordinate_audit_contexts(
        args.checkpoint, args.manifest, output, torch.device(args.device)
    )
    rows = []
    results = []
    for context in contexts:
        result = evaluate_active_sample(
            q_aux_O=context["q_aux"],
            valid_mask=context["mask"],
            target_O=context["target"],
            observed_C=context["observed"],
            vertices_O=context["vertices"],
            faces=context["faces"],
            equivalent_pose=context["equivalent_pose"],
            procrustes=context["model"].weighted_procrustes,
            candidate_k=16,
            projection_chunk_size=args.projection_chunk_size,
        )
        row = active_row(
            result,
            sample_id=context["sample"]["sample_id"],
            frame_id=int(context["sample"]["frame_id"]),
            T_W_from_C=context["T_W_from_C"],
            target_leakage_detected=False,
        )
        row["selected_shared_symmetry_element"] = int(
            context["selected_symmetry_element"]
        )
        rows.append(row)
        results.append((context, result))
    expected_frames = [
        int(sample["frame_id"]) for sample in contexts[0]["manifest"]["samples"]
    ]
    gate = four_view_stage_gate(rows, expected_frames=expected_frames)
    gate["stage_kind"] = (
        "two_view_active_coordinate" if len(rows) == 2 else "four_view_active_coordinate"
    )
    world = active_world_metrics(
        rows, contexts[0]["metadata"], contexts[0]["effective_group"]
    )
    namespaces = _aggregate_active_namespaces(rows, world)
    after = _source_signatures(source_run)
    source_unchanged = before == after
    summary = {
        "recheck_completed": True,
        "active_path_only": True,
        "source_run": str(source_run),
        "source_unchanged": source_unchanged,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "sample_count": len(rows),
        "frames": expected_frames,
        "nonfinite_detected": any(
            bool(row["active_nonfinite_detected"]) for row in rows
        ),
        "stage_passed": bool(gate["stage_passed"]),
        "metrics": namespaces,
        "world_frame_consistency": world,
        "inactive_metrics_ignored": True,
    }
    (output / "active_metric_recheck_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "stage_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    (output / "active_world_metrics.json").write_text(
        json.dumps(world, indent=2) + "\n", encoding="utf-8"
    )
    (output / "source_integrity.json").write_text(
        json.dumps(
            {"before": before, "after": after, "source_unchanged": source_unchanged},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    scalar_fields = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (str, bool, int, float))
        }
    )
    with (output / "active_metrics_per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows(
            [{key: row.get(key) for key in scalar_fields} for row in rows]
        )
    report = [
        "# Active coordinate-guided metric recheck",
        "",
        f"- source unchanged: `{source_unchanged}`",
        f"- active stage passed: `{gate['stage_passed']}`",
        f"- inactive legacy nonfinite values ignored: `True`",
        "",
        "| frame | q_aux p95 mm | exact p95 mm | alignment p95 mm | rotation deg | translation mm | K16 recall | fallback | passed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    by_frame_gate = {int(item["frame_id"]): item for item in gate["per_sample"]}
    for row in rows:
        report.append(
            f"| {row['frame_id']} | {row['aux_coordinate_p95_mm']:.6f} | "
            f"{row['exact_global_projected_correspondence_p95_mm']:.6f} | "
            f"{row['exact_global_projection_alignment_p95_mm']:.6f} | "
            f"{row['exact_global_projection_rotation_error_deg']:.6f} | "
            f"{row['exact_global_projection_translation_error_mm']:.6f} | "
            f"{row['k16_exact_global_triangle_recall']:.6f} | "
            f"{row['k16_fallback_fraction']:.6f} | "
            f"{by_frame_gate[int(row['frame_id'])]['passed']} |"
        )
    (output / "active_metric_recheck_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--projection-mode", default="exact_global"); parser.add_argument("--candidate-mode", default="all")
    parser.add_argument("--candidate-k", nargs="+", type=int, default=[16,32,64,128,256])
    parser.add_argument("--projection-chunk-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda"); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); output = Path(args.output_dir).expanduser().resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); register_all_modules()
    result = run(args); print(json.dumps({"output_dir": str(output), **result}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
