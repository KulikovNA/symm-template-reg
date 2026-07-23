#!/usr/bin/env python3
"""Recompute correspondence metrics from raw best-checkpoint tensors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.manifest import load_and_validate_manifest  # noqa: E402
from symm_template_reg.geometry import closest_points_on_triangle_mesh  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.models.losses import SymmetryAwareCorrespondenceLoss  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg  # noqa: E402


def _stats_mm(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu() * 1000.0
    return {
        "p50_mm": float(torch.quantile(values, 0.50)),
        "p95_mm": float(torch.quantile(values, 0.95)),
        "max_mm": float(values.max()),
    }


def _dataset_and_sample(config: dict, run: Path):
    dataset_cfg = deepcopy(config["dataset"])
    data_cfg = config["data"]
    dataset_cfg["fragment_mesh_filter"] = deepcopy(data_cfg["fragment_mesh_filter"])
    dataset_cfg["observed_filter"] = deepcopy(data_cfg["observed_filter"])
    dataset_cfg["symmetry_region_activity"] = deepcopy(
        data_cfg.get("symmetry_region_activity", {})
    )
    dataset_cfg["fragment_mesh_cache_dir"] = str(
        Path("/tmp/correspondence_metric_audit_cache") / run.name
    )
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    manifest_path = Path(data_cfg["train_manifest"])
    manifest, _ = load_and_validate_manifest(manifest_path, config, dataset)
    by_id = {record.sample_id: index for index, record in enumerate(dataset.sample_records)}
    sample_id = str(manifest["samples"][0]["sample_id"])
    return dataset, dataset[by_id[sample_id]]


@torch.no_grad()
def audit_run(run: Path, device: torch.device, tolerance_m: float) -> tuple[dict, list[dict]]:
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    dataset, sample = _dataset_and_sample(config, run)
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    batch = move_to_device(collate([sample]), device)
    model = build_model(config["model"]).to(device).eval()
    checkpoint = torch.load(run / "checkpoints" / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    prediction = model(batch)
    valid = prediction.observed_valid_mask[0]
    q_pred = prediction.correspondence_points_O[0, valid]
    target_payload = batch["gt"]["points_O_corresponding"]
    q_gt_all = target_payload.to_padded()["points"] if hasattr(target_payload, "to_padded") else target_payload
    match = SymmetryAwareCorrespondenceLoss().forward_with_diagnostics(
        prediction.correspondence_points_O,
        q_gt_all,
        prediction.observed_valid_mask,
        batch["template_symmetry_metadata"],
        batch["gt"]["effective_symmetry_group"],
        prediction.correspondence_confidence,
    )
    q_gt = match["matched_target_points_O"][0, valid]
    vertices = batch["template_mesh_vertices_O"][0]
    faces = batch["template_mesh_faces"][0]
    row = torch.linalg.vector_norm(q_pred - q_gt, dim=-1)
    pred_surface = closest_points_on_triangle_mesh(q_pred, vertices, faces)["distances"]
    gt_surface = closest_points_on_triangle_mesh(q_gt, vertices, faces)["distances"]
    reverse = torch.cdist(q_gt[None].float(), q_pred[None].float())[0].amin(-1)
    forward_set = torch.cdist(q_pred[None].float(), q_gt[None].float())[0].amin(-1)
    inequality_rhs = row + gt_surface + float(tolerance_m)
    violations = pred_surface > inequality_rhs
    point_rows = [
        {
            "point_index": int(index),
            "row_error_mm": float(row[index] * 1000.0),
            "predicted_to_template_surface_mm": float(pred_surface[index] * 1000.0),
            "gt_to_template_surface_mm": float(gt_surface[index] * 1000.0),
            "inequality_margin_mm": float((inequality_rhs[index] - pred_surface[index]) * 1000.0),
            "inequality_passed": not bool(violations[index]),
        }
        for index in torch.nonzero(violations, as_tuple=False).flatten().tolist()
    ]
    summary = {
        "run_dir": str(run),
        "run_id": run.name,
        "checkpoint": str(run / "checkpoints" / "best.pth"),
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "sample_id": sample["sample_id"],
        "frame_id": int(sample["frame_id"]),
        "valid_point_count": int(valid.sum()),
        "selected_shared_symmetry_element": int(match["selected_shared_symmetry_element"][0]),
        "row_aligned_error": _stats_mm(row),
        "predicted_to_template_triangle_surface": _stats_mm(pred_surface),
        "gt_to_template_triangle_surface": _stats_mm(gt_surface),
        "template_visible_patch_to_predicted": _stats_mm(reverse),
        "predicted_to_template_visible_patch": _stats_mm(forward_set),
        "symmetric_chamfer_p95_mm": float(max(torch.quantile(reverse, .95), torch.quantile(forward_set, .95)) * 1000.0),
        "pointwise_inequality_tolerance_mm": float(tolerance_m * 1000.0),
        "pointwise_inequality_violation_count": int(violations.sum()),
        "pointwise_inequality_max_excess_mm": float(
            (pred_surface - inequality_rhs).clamp_min(0).max() * 1000.0
        ),
        "metric_consistency_passed": not bool(violations.any()),
        "metric_bug_diagnosis": (
            "no_pointwise_metric_contradiction"
            if not bool(violations.any())
            else "evaluator_point_set_frame_or_unit_mismatch"
        ),
    }
    return summary, point_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--numerical-tolerance-m", type=float, default=1e-6)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    register_all_modules()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    summaries, violations = [], []
    for value in args.run_dir:
        summary, point_rows = audit_run(Path(value).expanduser().resolve(), torch.device(args.device), args.numerical_tolerance_m)
        summaries.append(summary)
        violations.extend({"run_id": summary["run_id"], **row} for row in point_rows)
    passed = all(item["metric_consistency_passed"] for item in summaries)
    result = {
        "audit_passed": passed,
        "same_valid_indices_required": True,
        "distance_definition": "exact one-way closest point on template triangles",
        "runs": summaries,
    }
    (output / "correspondence_metric_consistency_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    flat = []
    for item in summaries:
        flat.append({
            "run_id": item["run_id"], "frame_id": item["frame_id"],
            "valid_point_count": item["valid_point_count"],
            "row_aligned_p95_mm": item["row_aligned_error"]["p95_mm"],
            "predicted_to_template_surface_p95_mm": item["predicted_to_template_triangle_surface"]["p95_mm"],
            "gt_to_template_surface_p95_mm": item["gt_to_template_triangle_surface"]["p95_mm"],
            "template_visible_patch_to_predicted_p95_mm": item["template_visible_patch_to_predicted"]["p95_mm"],
            "symmetric_chamfer_p95_mm": item["symmetric_chamfer_p95_mm"],
            "inequality_violation_count": item["pointwise_inequality_violation_count"],
            "metric_consistency_passed": item["metric_consistency_passed"],
        })
    with (output / "correspondence_metric_consistency.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    with (output / "pointwise_inequality_violations.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(violations[0]) if violations else ["run_id", "point_index", "row_error_mm", "predicted_to_template_surface_mm", "gt_to_template_surface_mm", "inequality_margin_mm", "inequality_passed"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(violations)
    (output / "correspondence_metric_consistency_report.md").write_text(
        "# Correspondence metric consistency\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), **result}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
