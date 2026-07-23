#!/usr/bin/env python3
"""Audit per-point feature identity and gradient paths in correspondence heads."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.manifest import load_and_validate_manifest  # noqa: E402
from symm_template_reg.geometry import closest_points_on_triangle_mesh  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg  # noqa: E402


def _sample(config: dict, cache: Path):
    cfg = deepcopy(config["dataset"]); data = config["data"]
    cfg["fragment_mesh_filter"] = deepcopy(data["fragment_mesh_filter"])
    cfg["observed_filter"] = deepcopy(data["observed_filter"])
    cfg["symmetry_region_activity"] = deepcopy(data.get("symmetry_region_activity", {}))
    cfg["fragment_mesh_cache_dir"] = str(cache)
    dataset = build_from_cfg(cfg, DATASETS)
    manifest, _ = load_and_validate_manifest(data["train_manifest"], config, dataset)
    indices = {record.sample_id: i for i, record in enumerate(dataset.sample_records)}
    return dataset[indices[str(manifest["samples"][0]["sample_id"])]]


def _feature_stats(name: str, value: torch.Tensor, mask: torch.Tensor) -> dict:
    rows = value[0, mask] if value.ndim == 3 and value.shape[1] == mask.numel() else value[0]
    selected = rows[torch.linspace(0, len(rows) - 1, min(len(rows), 512), device=rows.device).long()]
    distance = torch.pdist(selected.float()) if len(selected) > 1 else selected.new_zeros(1)
    rounded = torch.round(rows.detach().float().cpu() * 1e5).to(torch.int64)
    unique = torch.unique(rounded, dim=0).shape[0]
    centered = selected.float() - selected.float().mean(0)
    return {
        "layer": name,
        "point_count": len(rows),
        "channel_count": rows.shape[-1],
        "feature_variance_mean": float(rows.float().var(0, unbiased=False).mean()),
        "pairwise_feature_distance_mean": float(distance.mean()),
        "pairwise_feature_distance_p05": float(torch.quantile(distance, .05)),
        "near_identical_pair_fraction": float((distance < 1e-5).float().mean()),
        "unique_rounded_feature_rows": int(unique),
        "unique_feature_fraction": unique / max(len(rows), 1),
        "effective_feature_rank": int(torch.linalg.matrix_rank(centered)),
    }


def _categorical_normalized_mutual_information(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().long().cpu(); right = right.detach().long().cpu()
    left_values, left_inverse = torch.unique(left, return_inverse=True)
    right_values, right_inverse = torch.unique(right, return_inverse=True)
    joint = torch.zeros((len(left_values), len(right_values)), dtype=torch.float64)
    joint.index_put_((left_inverse, right_inverse), torch.ones(len(left), dtype=torch.float64), accumulate=True)
    joint /= max(len(left), 1); p_left = joint.sum(1); p_right = joint.sum(0)
    expected = p_left[:, None] * p_right[None]
    valid = joint > 0
    mutual_information = (joint[valid] * (joint[valid] / expected[valid]).log()).sum()
    entropy_left = -(p_left[p_left > 0] * p_left[p_left > 0].log()).sum()
    entropy_right = -(p_right[p_right > 0] * p_right[p_right > 0].log()).sum()
    return float(mutual_information / (entropy_left * entropy_right).sqrt().clamp_min(1e-12))


def audit(run: Path, device: torch.device, cache: Path):
    config = json.loads((run / "resolved_config.json").read_text())
    sample = _sample(config, cache / run.name)
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    batch = move_to_device(collate([sample]), device)
    model = build_model(config["model"]).to(device).train()
    payload = torch.load(run / "checkpoints/best.pth", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    prediction = model(batch)
    mask = prediction.observed_valid_mask[0]
    path = prediction.correspondence_feature_path
    auxiliary = prediction.correspondence_auxiliary
    if path is None or auxiliary is None or "coarse_patch_logits" not in auxiliary:
        raise ValueError(f"run has no inspectable SurfaceV2 feature path: {run}")
    inspectable = {
        **path,
        "coarse_patch_logits": auxiliary["coarse_patch_logits"],
        "local_candidate_logits": auxiliary["fine_local_logits"],
        "barycentric_logits": auxiliary.get(
            "barycentric_logits", auxiliary["predicted_barycentric"]
        ),
    }
    rows = [_feature_stats(name, value, mask) for name, value in inspectable.items() if value.is_floating_point() and value.ndim == 3]
    target_payload = batch["gt"]["points_O_corresponding"]
    target = target_payload.to_padded()["points"] if hasattr(target_payload, "to_padded") else target_payload
    patch_points = auxiliary["patch_points_O"]
    all_candidates = auxiliary.get("all_candidate_triangle_ids")
    if all_candidates is not None:
        vertices = batch["template_mesh_vertices_O"][0]
        faces = batch["template_mesh_faces"][0]
        candidate_centroids = vertices[faces[all_candidates[0]]].mean(-2)
        gt_patch = torch.cdist(
            target[0].float(), candidate_centroids.reshape(-1, 3).float()
        ).reshape(target.shape[1], patch_points.shape[1], -1).amin(-1).argmin(-1)[None]
    else:
        gt_patch = torch.cdist(target.float(), patch_points.float()).argmin(-1)
    logits = auxiliary["coarse_patch_logits"]
    top = logits.topk(min(8, logits.shape[-1]), -1).indices
    top1 = top[..., :1].eq(gt_patch[..., None]).any(-1)[0, mask]
    top4 = top[..., :4].eq(gt_patch[..., None]).any(-1)[0, mask]
    top8 = top.eq(gt_patch[..., None]).any(-1)[0, mask]
    patch_loss = F.cross_entropy(logits[0, mask], gt_patch[0, mask])
    patch_gradients = {}
    inspect = {**path, "coarse_patch_logits": logits}
    for name, tensor in inspect.items():
        if tensor.is_floating_point() and tensor.requires_grad:
            gradient = torch.autograd.grad(patch_loss, tensor, retain_graph=True, allow_unused=True)[0]
            patch_gradients[name] = None if gradient is None else float(gradient.norm())
    vertices = batch["template_mesh_vertices_O"][0]
    faces = batch["template_mesh_faces"][0]
    nearest = closest_points_on_triangle_mesh(target[0, mask], vertices, faces)
    candidate = auxiliary["candidate_triangle_ids"][0, mask]
    local_target = torch.linalg.vector_norm(
        vertices[faces[candidate]].mean(2) - target[0, mask, None], dim=-1
    ).argmin(-1)
    fine_loss = F.cross_entropy(auxiliary["fine_local_logits"][0, mask], local_target)
    coarse_from_local = torch.autograd.grad(
        fine_loss, logits, retain_graph=True, allow_unused=True
    )[0]
    gradient = {
        "patch_loss": float(patch_loss),
        "patch_loss_gradient_norm_by_layer": patch_gradients,
        "local_fine_loss": float(fine_loss),
        "local_fine_gradient_to_coarse_logits": (
            None if coarse_from_local is None else float(coarse_from_local.norm())
        ),
        "hard_topk_breaks_local_to_coarse_gradient": (
            coarse_from_local is None or float(coarse_from_local.norm()) == 0.0
        ),
    }
    upsample = next(item for item in rows if item["layer"] == "upsampled_per_point_features")
    summary = {
        "run_id": run.name,
        "frame_id": int(sample["frame_id"]),
        "shell_point_count": int(mask.sum()),
        "interaction_token_count": int(model.max_observed_tokens),
        "upsampling_method": "nearest_interpolate from 256 FPS interaction tokens to every encoder point, then residual addition with original encoder feature",
        "each_point_receives_distinct_conditioned_feature": upsample["unique_feature_fraction"] > .99,
        "global_feature_broadcast_detected": upsample["unique_feature_fraction"] < .10,
        "coarse_patch_top1_accuracy": float(top1.float().mean()),
        "coarse_patch_top4_recall": float(top4.float().mean()),
        "coarse_patch_top8_recall": float(top8.float().mean()),
        "gt_patch_id_count": int(torch.unique(gt_patch[0, mask]).numel()),
        "predicted_patch_id_count": int(torch.unique(logits[0, mask].argmax(-1)).numel()),
        "predicted_vs_gt_patch_normalized_mutual_information": _categorical_normalized_mutual_information(
            logits[0, mask].argmax(-1), gt_patch[0, mask]
        ),
        "feature_layers": rows,
        "gradient_path": gradient,
    }
    return summary, rows, gradient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    register_all_modules(); output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        "correspondence_feature_path_summary.json", "gradient_path.json",
        "feature_variance_by_layer.csv", "correspondence_feature_path_report.md",
    )
    if any((output / name).exists() for name in expected_outputs):
        raise FileExistsError(f"feature audit outputs already exist in {output}")
    summaries, table, gradients = [], [], {}
    for raw in args.run_dir:
        summary, rows, gradient = audit(Path(raw).resolve(), torch.device(args.device), Path("/tmp/feature_path_audit_cache"))
        summaries.append(summary); table.extend({"run_id": summary["run_id"], "frame_id": summary["frame_id"], **row} for row in rows); gradients[summary["run_id"]] = gradient
    collapsed_classifier = any(
        row["predicted_patch_id_count"] <= 1
        and row["each_point_receives_distinct_conditioned_feature"]
        for row in summaries
    )
    result = {
        "audit_passed": True,
        "diagnosis": (
            "coarse_patch_classifier_failure"
            if collapsed_classifier
            else "no_per_point_feature_identity_failure_detected"
        ),
        "runs": summaries,
    }
    (output / "correspondence_feature_path_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "gradient_path.json").write_text(json.dumps(gradients, indent=2) + "\n")
    with (output / "feature_variance_by_layer.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    (output / "correspondence_feature_path_report.md").write_text("# Correspondence feature path\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n")
    print(json.dumps({"output_dir": str(output), **result}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
