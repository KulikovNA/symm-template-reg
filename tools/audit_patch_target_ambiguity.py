#!/usr/bin/env python3
"""Audit single-owner versus set-valued patch targets for one checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.manifest import load_and_validate_manifest  # noqa: E402
from symm_template_reg.geometry import closest_points_on_triangle_mesh  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.models.geometry.patch_targets import (  # noqa: E402
    single_owner_patch_ids,
    valid_patch_mask,
    valid_set_topk_hits,
)
from symm_template_reg.models.losses.symmetry_aware_correspondence_loss import (  # noqa: E402
    SymmetryAwareCorrespondenceLoss,
)
from symm_template_reg.registry import (  # noqa: E402
    COLLATE_FUNCTIONS,
    DATASETS,
    build_from_cfg,
)


SUMMARY_NAME = "patch_target_ambiguity_summary.json"
CSV_NAME = "patch_target_ambiguity_per_point.csv"
REPORT_NAME = "patch_target_ambiguity_report.md"


def _load_sample(config: dict[str, Any], cache: Path) -> tuple[dict[str, Any], Any]:
    dataset_cfg = deepcopy(config["dataset"])
    data = config["data"]
    dataset_cfg["fragment_mesh_filter"] = deepcopy(data["fragment_mesh_filter"])
    dataset_cfg["observed_filter"] = deepcopy(data["observed_filter"])
    dataset_cfg["symmetry_region_activity"] = deepcopy(
        data.get("symmetry_region_activity", {})
    )
    dataset_cfg["fragment_mesh_cache_dir"] = str(cache)
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    manifest, _ = load_and_validate_manifest(
        data["train_manifest"], config, dataset
    )
    records = {record.sample_id: index for index, record in enumerate(dataset.sample_records)}
    sample_id = str(manifest["samples"][0]["sample_id"])
    return dataset[records[sample_id]], manifest


def _target_points(batch: dict[str, Any]) -> torch.Tensor:
    payload = batch["gt"]["points_O_corresponding"]
    return payload.to_padded()["points"] if hasattr(payload, "to_padded") else payload


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float(), q)) if values.numel() else math.nan


@torch.no_grad()
def audit_checkpoint(
    run_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the exact-triangle ambiguity audit and write its three artifacts."""

    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    sample, manifest = _load_sample(config, output_dir / "cache" / "fragment_mesh_metadata")
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    batch = move_to_device(collate([sample]), device)
    model = build_model(config["model"]).to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    prediction = model(batch)
    auxiliary = prediction.correspondence_auxiliary
    if auxiliary is None or "all_candidate_triangle_ids" not in auxiliary:
        raise ValueError("checkpoint does not expose SurfaceV2 patch candidates")

    mask = prediction.observed_valid_mask[0]
    raw_target = _target_points(batch)
    symmetry_match = SymmetryAwareCorrespondenceLoss().forward_with_diagnostics(
        prediction.correspondence_points_O,
        raw_target,
        prediction.observed_valid_mask,
        batch["template_symmetry_metadata"],
        batch["gt"]["effective_symmetry_group"],
        prediction.correspondence_confidence,
    )
    selected_symmetry = int(
        symmetry_match["selected_shared_symmetry_element"][0]
    )
    target = symmetry_match["matched_target_points_O"][0, mask]
    vertices = batch["template_mesh_vertices_O"][0]
    faces = batch["template_mesh_faces"][0].long()
    nearest = closest_points_on_triangle_mesh(
        target, vertices, faces, point_chunk_size=256
    )
    gt_triangle = nearest["face_ids"]
    all_candidates = auxiliary["all_candidate_triangle_ids"][0]
    valid_targets = valid_patch_mask(gt_triangle, all_candidates)
    owners = auxiliary["face_owner_patch_ids"][0]
    single_owner = single_owner_patch_ids(gt_triangle, owners)
    logits = auxiliary["coarse_patch_logits"][0, mask]
    top = logits.topk(min(4, logits.shape[-1]), dim=-1).indices
    top1 = top[:, 0]
    single_hit = top1.eq(single_owner)
    valid_top1 = valid_targets.gather(-1, top1[:, None]).squeeze(-1)
    valid_top4 = valid_set_topk_hits(top, valid_targets, 4, already_topk=True)
    valid_count = valid_targets.sum(-1)

    centroids = vertices[faces].mean(1)
    predicted_region_faces = all_candidates[top1]
    region_distance = torch.linalg.vector_norm(
        centroids[predicted_region_faces] - centroids[gt_triangle, None], dim=-1
    ).amin(-1)
    candidate_face_vertices = faces[predicted_region_faces]
    gt_face_vertices = faces[gt_triangle]
    shared_vertices = candidate_face_vertices[..., :, None].eq(
        gt_face_vertices[:, None, None, :]
    ).any(-1).sum(-1)
    wrong_adjacent = (~single_hit) & shared_vertices.ge(2).any(-1)
    corrected_single_error = (~single_hit) & valid_top1

    rows: list[dict[str, Any]] = []
    target_cpu = target.detach().cpu()
    for index in range(len(target)):
        valid_ids = torch.nonzero(
            valid_targets[index], as_tuple=False
        ).flatten().detach().cpu().tolist()
        rows.append(
            {
                "point_index": index,
                "target_x_m": float(target_cpu[index, 0]),
                "target_y_m": float(target_cpu[index, 1]),
                "target_z_m": float(target_cpu[index, 2]),
                "gt_nearest_template_triangle": int(gt_triangle[index]),
                "single_owner_gt_patch": int(single_owner[index]),
                "valid_patch_ids": " ".join(map(str, valid_ids)),
                "valid_patch_count": len(valid_ids),
                "predicted_top1_patch": int(top1[index]),
                "predicted_top4_patches": " ".join(
                    map(str, top[index].detach().cpu().tolist())
                ),
                "single_owner_top1_correct": bool(single_hit[index]),
                "valid_patch_set_top1_correct": bool(valid_top1[index]),
                "valid_patch_set_top4_hit": bool(valid_top4[index]),
                "wrong_top1_but_same_triangle_available": bool(
                    corrected_single_error[index]
                ),
                "wrong_top1_adjacent_patch": bool(wrong_adjacent[index]),
                "predicted_patch_region_to_gt_triangle_centroid_min_distance_mm": float(
                    region_distance[index] * 1000.0
                ),
            }
        )

    unique_patch, frequencies = top1.unique(return_counts=True)
    distance_mm = region_distance * 1000.0
    summary = {
        "audit_passed": True,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "sample_id": sample.get("sample_id"),
        "frame_id": int(sample["frame_id"]),
        "manifest": str(config["data"]["train_manifest"]),
        "manifest_sample_count": len(manifest["samples"]),
        "observed_point_count": int(mask.sum()),
        "selected_shared_symmetry_element": selected_symmetry,
        "patch_count": int(logits.shape[-1]),
        "single_owner_top1_accuracy": float(single_hit.float().mean()),
        "valid_patch_set_top1_accuracy": float(valid_top1.float().mean()),
        "valid_patch_set_top4_recall": float(valid_top4.float().mean()),
        "valid_patch_set_in_candidate_set_fraction": float(
            valid_top4.float().mean()
        ),
        "mean_valid_patch_count": float(valid_count.float().mean()),
        "max_valid_patch_count": int(valid_count.max()),
        "fraction_with_multiple_valid_patches": float(
            valid_count.gt(1).float().mean()
        ),
        "wrong_top1_but_same_triangle_available_fraction": float(
            corrected_single_error.float().mean()
        ),
        "wrong_top1_adjacent_patch_fraction": float(wrong_adjacent.float().mean()),
        "predicted_patch_region_to_gt_triangle_distance_definition": (
            "minimum Euclidean distance from the GT-triangle centroid to any "
            "triangle centroid in the predicted patch region"
        ),
        "predicted_patch_region_to_gt_triangle_distance_mean_mm": float(
            distance_mm.mean()
        ),
        "predicted_patch_region_to_gt_triangle_distance_p50_mm": _quantile(
            distance_mm, 0.50
        ),
        "predicted_patch_region_to_gt_triangle_distance_p95_mm": _quantile(
            distance_mm, 0.95
        ),
        "predicted_patch_region_to_gt_triangle_distance_max_mm": float(
            distance_mm.max()
        ),
        "unique_predicted_patches": int(unique_patch.numel()),
        "most_popular_patch_fraction": float(frequencies.max() / len(top1)),
        "nonfinite_detected": not bool(torch.isfinite(logits).all()),
        "set_valued_interpretation": (
            "predicted top-1 inside valid_patch_ids is geometrically correct even "
            "when it differs from single_owner_gt_patch"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    with (output_dir / CSV_NAME).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = (
        "# Patch target ambiguity audit\n\n"
        f"- sample: `{summary['sample_id']}`\n"
        f"- points: {summary['observed_point_count']}\n"
        f"- single-owner top-1: {summary['single_owner_top1_accuracy']:.8f}\n"
        f"- valid-set top-1: {summary['valid_patch_set_top1_accuracy']:.8f}\n"
        f"- valid-set top-4: {summary['valid_patch_set_top4_recall']:.8f}\n"
        f"- points with multiple valid patches: "
        f"{summary['fraction_with_multiple_valid_patches']:.8f}\n\n"
        "A top-1 patch that contains the exact GT nearest triangle is counted as "
        "geometrically correct, even if it is not that triangle's single FPS owner.\n\n"
        "The region-distance statistic is centroid based and is reported in millimetres.\n"
    )
    (output_dir / REPORT_NAME).write_text(report, encoding="utf-8")
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    register_all_modules()
    summary, _ = audit_checkpoint(
        Path(args.run_dir).expanduser().resolve(),
        Path(args.checkpoint).expanduser().resolve(),
        output,
        torch.device(args.device),
    )
    print(json.dumps({"output_dir": str(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
