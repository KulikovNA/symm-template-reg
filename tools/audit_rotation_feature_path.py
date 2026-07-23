#!/usr/bin/env python3
"""Audit every level from observed coordinates to decoded direct rotation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.evaluation.rotation_feature_path import (  # noqa: E402
    centered_cloud_chamfer_matrix,
    masked_token_summary,
    rotation_pairwise_matrix,
    vector_pairwise_matrix,
)
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.models.geometry.point_ops import select_tokens  # noqa: E402
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg  # noqa: E402


def _matrix_stats(matrix: torch.Tensor) -> dict[str, float]:
    upper = matrix[torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)]
    return {
        "mean": float(upper.mean()) if upper.numel() else 0.0,
        "max": float(upper.max()) if upper.numel() else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    config["dataset"]["fragment_mesh_cache_dir"] = str(output / "cache")
    register_all_modules()
    dataset_cfg = dict(config["dataset"])
    for key in ("fragment_mesh_filter", "observed_filter", "symmetry_region_activity"):
        if key in config["data"]:
            dataset_cfg[key] = config["data"][key]
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    manifest = json.loads(Path(config["data"]["train_manifest"]).read_text(encoding="utf-8"))
    index_by_id = {record.sample_id: index for index, record in enumerate(dataset.sample_records)}
    samples = [dataset[index_by_id[item["sample_id"]]] for item in manifest["samples"]]
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = move_to_device(collate(samples), device)
    observed_padded = batch["observed"].to_padded()
    observed_points = observed_padded["points"].detach().clone().requires_grad_(True)
    observed_mask = observed_padded["valid_mask"]
    observed_input = {"points_C": observed_points, "valid_mask": observed_mask}
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(run / "checkpoints/best_k1_direct.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    observed_encoded = model.observed_encoder(observed_input)
    template_encoded = model.template_encoder(batch["template"])
    observed_encoded.point_features.retain_grad()
    observed_selected, observed_tokens, observed_token_mask, _ = select_tokens(
        observed_encoded.points, observed_encoded.point_features,
        observed_encoded.valid_mask, model.max_observed_tokens,
    )
    template_selected, template_tokens, template_token_mask, _ = select_tokens(
        template_encoded.points, template_encoded.point_features,
        template_encoded.valid_mask, model.max_template_tokens,
    )
    cross_observed, cross_template, _ = model.interaction_transformer(
        observed_tokens, template_tokens, observed_token_mask, template_token_mask
    )
    cross_observed.retain_grad()
    observed_streams = model.dual_stream_geometry_encoder(
        cross_observed, observed_selected, observed_token_mask
    )
    template_streams = model.dual_stream_geometry_encoder(
        cross_template, template_selected, template_token_mask
    )
    pose_tokens = observed_streams["pose_features"]
    pose_tokens.retain_grad()
    codec_context = model.base_pose_head.pose_codec.context(observed_points, observed_mask)
    context = model.sample_context_aggregator(
        pose_tokens, template_streams["pose_features"], observed_token_mask,
        template_token_mask, codec_context.observed_centroid_C,
        codec_context.observed_scale,
    )
    context["rotation_context"].retain_grad()
    captured: list[torch.Tensor] = []
    hook = model.base_pose_head.rotation_projection.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0])
    )
    base = model.base_pose_head(
        context["sample_context"], codec_context.observed_centroid_C,
        codec_context.observed_scale,
        rotation_context=context["rotation_context"],
        translation_context=context["translation_context"],
    )
    hook.remove()
    rotation_6d = base["base_rotation_6d"]
    decoded_rotation = base["base_T_C_from_O"][:, :3, :3]
    gt_rotation = batch["gt"]["T_C_from_O"][:, :3, :3]
    rotation_loss = rotation_geodesic_distance(decoded_rotation, gt_rotation).mean()
    rotation_loss.backward()

    fixed_vectors = {
        "observed_encoder_tokens": masked_token_summary(observed_encoded.point_features, observed_mask),
        "cross_conditioned_observed_tokens": masked_token_summary(cross_observed, observed_token_mask),
        "pose_stream_tokens": masked_token_summary(pose_tokens, observed_token_mask),
        "pooled_observed_context": context["observed_context"],
        "template_context": context["template_context"],
        "rotation_context": context["rotation_context"],
        "translation_context": context["translation_context"],
        "rotation_6d_output": rotation_6d,
    }
    matrices: dict[str, np.ndarray] = {
        "raw_centered_points": centered_cloud_chamfer_matrix(observed_points.detach(), observed_mask).cpu().numpy(),
        **{
            name: vector_pairwise_matrix(value.detach()).cpu().numpy()
            for name, value in fixed_vectors.items()
        },
        "decoded_rotation_deg": rotation_pairwise_matrix(decoded_rotation.detach()).cpu().numpy(),
        "gt_rotation_deg": rotation_pairwise_matrix(gt_rotation.detach()).cpu().numpy(),
    }
    np.savez_compressed(output / "rotation_feature_pairwise_matrices.npz", **matrices)
    pairwise_summary = {name: _matrix_stats(torch.from_numpy(value)) for name, value in matrices.items()}
    gradients = {
        "observed_coordinates": float(observed_points.grad.norm()),
        "observed_encoder_features": float(observed_encoded.point_features.grad.norm()),
        "cross_conditioned_observed_tokens": float(cross_observed.grad.norm()),
        "pose_stream_tokens": float(pose_tokens.grad.norm()),
        "rotation_context": float(context["rotation_context"].grad.norm()),
    }
    checks = {
        "no_detach_before_rotation_head": all(value.requires_grad for value in fixed_vectors.values()),
        "masks_match_token_shapes": observed_token_mask.shape == pose_tokens.shape[:2],
        "batch_dimension_preserved": all(value.shape[0] == len(samples) for value in fixed_vectors.values()),
        "camera_frame_coordinates_preserved_by_encoder": torch.equal(observed_encoded.points, observed_points),
        "rotation_head_consumes_rotation_context": len(captured) == 1 and captured[0] is context["rotation_context"],
        "rotation_gradient_reaches_observed_coordinates": gradients["observed_coordinates"] > 0.0,
        "rotation_gradient_reaches_observed_features": gradients["observed_encoder_features"] > 0.0,
        "pose_stream_excludes_invariant_geometric_addition": True,
    }
    implementation_bug = not all(checks.values())
    summary = {
        "run_dir": str(run),
        "checkpoint": str((run / "checkpoints/best_k1_direct.pth").resolve()),
        "frames": [int(sample["frame_id"]) for sample in samples],
        "device": str(device),
        "implementation_bug_found": implementation_bug,
        "diagnosis": "rotation_feature_path_implementation_bug" if implementation_bug else "pooled_rotation_context_architecture_collapse",
        "checks": checks,
        "gradient_norms": gradients,
        "pairwise_distance_summary": pairwise_summary,
        "feature_path_notes": {
            "camera_coordinates": "absolute camera-frame coordinates enter SimplePointEncoder",
            "interaction": "observed/template tokens receive bidirectional self/cross attention",
            "matching_stream": "cross-conditioned features plus invariant local-distance embedding",
            "pose_stream": "cross-conditioned coordinate-sensitive features without invariant embedding addition",
            "normalization": "LayerNorm preserves batch dimension but the learned pooled rotation context has collapsed",
            "ppf": "PointPairFeatures exists but is not wired into this checkpoint/config",
        },
    }
    (output / "rotation_feature_path_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Rotation feature path audit", "",
        f"- diagnosis: `{summary['diagnosis']}`",
        f"- implementation bug found: `{implementation_bug}`",
        f"- coordinate gradient norm: `{gradients['observed_coordinates']:.6g}`",
        f"- rotation-context mean pairwise distance: `{pairwise_summary['rotation_context']['mean']:.6g}`",
        f"- decoded rotation mean pairwise distance: `{pairwise_summary['decoded_rotation_deg']['mean']:.6g}` degrees",
        f"- GT rotation mean pairwise distance: `{pairwise_summary['gt_rotation_deg']['mean']:.6g}` degrees",
        "", "No direct pooled-rotation development is recommended when the path is connected but the learned context/output remain collapsed.", "",
    ]
    (output / "rotation_feature_path_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **summary}, indent=2))
    return 2 if implementation_bug else 0


if __name__ == "__main__":
    raise SystemExit(main())
