#!/usr/bin/env python3
"""Counterfactual audit of the conditioned model's actual forward contract."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import load_config  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg  # noqa: E402


def _prediction_tensors(prediction: Any) -> dict[str, torch.Tensor]:
    names = (
        "base_pose", "pose_hypotheses", "pose_logits",
        "correspondence_points_O", "correspondence_confidence",
        "correspondence_logits",
    )
    return {
        name: getattr(prediction, name).detach().clone()
        for name in names
        if isinstance(getattr(prediction, name, None), torch.Tensor)
    }


def _max_difference(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    values = []
    for key in left:
        finite = torch.isfinite(left[key]) & torch.isfinite(right[key])
        if bool(finite.any()):
            values.append(float((left[key][finite] - right[key][finite]).abs().max()))
    return max(values, default=0.0)


def _set_packed_points(value: Any, points: torch.Tensor) -> Any:
    result = copy.deepcopy(value)
    if hasattr(result, "points"):
        result.points = points
        result.validate()
        return result
    if isinstance(result, dict):
        key = "points_C" if "points_C" in result else "points_O" if "points_O" in result else "points"
        result[key] = points
        if "points" in result:
            result["points"] = points
        return result
    raise TypeError("unsupported point batch")


def _padded(value: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(value, "to_padded"):
        dense = value.to_padded()
        return dense["points"], dense["valid_mask"]
    points = value.get("points_C", value.get("points_O", value.get("points")))
    return points, value["valid_mask"]


def _correspondence_rmse(prediction: Any, batch: dict[str, Any]) -> float:
    target, mask = _padded(batch["gt"]["points_O_corresponding"])
    distance = torch.linalg.vector_norm(prediction.correspondence_points_O - target, dim=-1)
    return float(torch.sqrt((distance[mask] ** 2).mean()) * 1000.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-samples", type=int, default=4)
    args = parser.parse_args()
    checkpoint_path: Path | None = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            parser.error(
                "--checkpoint does not exist: "
                f"{checkpoint_path}. Train correspondence-only first and pass "
                "the best_checkpoint path from per_run_summary.csv; "
                "'/absolute/path/to/...' in the runbook is not a literal path."
            )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    register_all_modules()
    dataset_cfg = copy.deepcopy(config["dataset"])
    dataset_cfg["fragment_mesh_filter"] = copy.deepcopy(config["data"]["fragment_mesh_filter"])
    dataset_cfg["observed_filter"] = copy.deepcopy(config["data"]["observed_filter"])
    dataset_cfg["symmetry_region_activity"] = copy.deepcopy(
        config["data"].get("symmetry_region_activity", {})
    )
    dataset_cfg["fragment_mesh_cache_dir"] = str(output / "cache")
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    manifest = json.loads(Path(args.manifest).expanduser().read_text(encoding="utf-8"))
    sample_ids = [str(item["sample_id"]) for item in manifest["samples"][: args.max_samples]]
    index_by_id = {record.sample_id: index for index, record in enumerate(dataset.sample_records)}
    samples = [dataset[index_by_id[sample_id]] for sample_id in sample_ids]
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    batch = move_to_device(collate(samples), device)
    model = build_model(config["model"]).to(device).eval()
    if checkpoint_path is not None:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=True)

    def run(value: dict[str, Any]) -> tuple[Any, dict[str, torch.Tensor]]:
        with torch.no_grad():
            prediction = model(value)
        return prediction, _prediction_tensors(prediction)

    original_prediction, original = run(batch)
    changed_points = copy.deepcopy(batch)
    target = changed_points["gt"]["points_O_corresponding"]
    if hasattr(target, "points"):
        target.points.add_(123.0)
    else:
        target.add_(123.0)
    _, points_output = run(changed_points)
    changed_pose = copy.deepcopy(batch)
    changed_pose["gt"]["T_C_from_O"] = changed_pose["gt"]["T_C_from_O"].clone()
    changed_pose["gt"]["T_C_from_O"][:, :3, 3] += 10.0
    _, pose_output = run(changed_pose)
    changed_fragment_mesh = copy.deepcopy(batch)
    for item in changed_fragment_mesh.get("meta", []):
        item["fragment_mesh"] = {"counterfactual": "must_not_be_a_model_input"}
    _, fragment_mesh_output = run(changed_fragment_mesh)

    observed_points, observed_mask = _padded(batch["observed"])
    permutation = torch.arange(len(sample_ids) - 1, -1, -1, device=device)
    # Re-collation is the authoritative way to permute variable-length packed
    # inputs, including every point-aligned feature and metadata entry.
    permuted = move_to_device(collate(list(reversed(samples))), device)
    _, permuted_output = run(permuted)
    correspondence_equivariance_difference = float(
        (
            permuted_output["correspondence_points_O"]
            - original["correspondence_points_O"][permutation]
        ).abs().max()
    )
    pose_equivariance_difference = float(
        (permuted_output["base_pose"] - original["base_pose"][permutation])
        .abs()
        .max()
    )

    zero_observed = copy.deepcopy(batch)
    zero_observed["observed"] = _set_packed_points(
        zero_observed["observed"], torch.zeros_like(zero_observed["observed"].points)
    )
    zero_observed_prediction, zero_observed_output = run(zero_observed)
    zero_template = copy.deepcopy(batch)
    zero_template["template"] = _set_packed_points(
        zero_template["template"], torch.zeros_like(zero_template["template"].points)
    )
    zero_template_prediction, zero_template_output = run(zero_template)
    baseline_rmse = _correspondence_rmse(original_prediction, batch)
    observed_zero_rmse = _correspondence_rmse(zero_observed_prediction, batch)
    template_zero_rmse = _correspondence_rmse(zero_template_prediction, batch)
    trained_quality_check = args.checkpoint is not None
    checks = {
        "gt_points_do_not_change_outputs": _max_difference(original, points_output) <= 1e-8,
        "gt_pose_does_not_change_outputs": _max_difference(original, pose_output) <= 1e-8,
        "fragment_mesh_does_not_change_outputs": _max_difference(
            original, fragment_mesh_output
        ) <= 1e-8,
        "observed_permutation_is_equivariant": correspondence_equivariance_difference <= 1e-5,
        "zero_observed_changes_correspondences": _max_difference(original, zero_observed_output) > 1e-7,
        "zero_template_changes_correspondences": _max_difference(original, zero_template_output) > 1e-7,
    }
    if trained_quality_check:
        checks["zero_observed_worsens_correspondence_quality"] = observed_zero_rmse > baseline_rmse
        checks["zero_template_worsens_correspondence_quality"] = template_zero_rmse > baseline_rmse
    structural_pass = all(checks.values())
    report = {
        "config": str(Path(args.config).expanduser().resolve()),
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "checkpoint": args.checkpoint,
        "quality_worsening_enforced": trained_quality_check,
        "checks": checks,
        "max_output_change_after_gt_points": _max_difference(original, points_output),
        "max_output_change_after_gt_pose": _max_difference(original, pose_output),
        "max_output_change_after_fragment_mesh": _max_difference(
            original, fragment_mesh_output
        ),
        "observed_permutation_equivariance_max_abs": correspondence_equivariance_difference,
        "correspondence_input_permutation_error": correspondence_equivariance_difference,
        "base_pose_input_permutation_error": pose_equivariance_difference,
        "correspondence_observed_zeroing_degradation_mm": observed_zero_rmse - baseline_rmse,
        "correspondence_template_zeroing_degradation_mm": template_zero_rmse - baseline_rmse,
        "correspondence_context_pairwise_distance": float(
            torch.pdist(
                original["correspondence_points_O"].flatten(1)
            ).mean()
        ) if len(sample_ids) > 1 else 0.0,
        "correspondence_rmse_mm": {
            "original": baseline_rmse,
            "zero_observed": observed_zero_rmse,
            "zero_template": template_zero_rmse,
        },
        "matching_feature_contract": {
            "observed_template_self_cross_interaction": hasattr(
                model, "interaction_transformer"
            ),
            "geotransformer_style_geometric_embedding": getattr(
                model.dual_stream_geometry_encoder,
                "matching_geometric_embedding",
                None,
            ) is not None,
            "ppf_embedding": getattr(
                model.dual_stream_geometry_encoder,
                "matching_ppf_embedding",
                None,
            ) is not None,
            "coordinate_independent_matching_only": bool(
                getattr(
                    model.dual_stream_geometry_encoder,
                    "matching_geometric_only",
                    False,
                )
            ),
            "valid_masks_passed_to_matching": True,
        },
        "target_leakage_detected": not (
            checks["gt_points_do_not_change_outputs"]
            and checks["gt_pose_does_not_change_outputs"]
            and checks["fragment_mesh_does_not_change_outputs"]
        ),
        "diagnosis": (
            "correspondence_context_collapse"
            if trained_quality_check
            and len(sample_ids) > 1
            and float(torch.pdist(original["correspondence_points_O"].flatten(1)).mean()) <= 1e-5
            else "correspondence_observed_dependency_failure"
            if trained_quality_check and observed_zero_rmse <= baseline_rmse
            else "correspondence_template_dependency_failure"
            if trained_quality_check and template_zero_rmse <= baseline_rmse
            else "correspondence_responds_to_input_geometry"
        ),
        "audit_passed": structural_pass,
    }
    (output / "target_leakage_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "target_leakage_report.md").write_text(
        "# Target leakage audit\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), **report}, indent=2))
    return 0 if structural_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
