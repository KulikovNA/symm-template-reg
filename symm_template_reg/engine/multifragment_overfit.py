"""Contracts and metrics for the four-fragment/four-view overfit experiment."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


WARNING_FLAGS = {
    "debug_training_on_test_split": True,
    "train_and_validation_use_same_samples": True,
    "results_are_not_final_evaluation": True,
}
EXPECTED_FRAGMENTS = (0, 1, 2, 3)
EXPECTED_FRAMES = (2, 4, 5, 8)
EXPECTED_SAMPLE_COUNT = 16


def canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    value = dict(payload)
    value.pop("manifest_sha256", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_content_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(payload)).hexdigest()


def validate_multifragment_manifest_payload(
    payload: Mapping[str, Any], *, min_num_faces: int = 840
) -> dict[str, Any]:
    """Validate the exact 4 physical fragments x 4 shared views contract."""

    for key, expected in WARNING_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"multifragment manifest requires {key}=true")
    if payload.get("manifest_type") != "four_fragments_four_views_overfit":
        raise ValueError("unexpected multifragment manifest_type")
    if payload.get("experiment_type") != "four_fragments_four_views_overfit":
        raise ValueError("unexpected experiment_type")
    if payload.get("initialization_mode") != "scratch":
        raise ValueError("main multifragment manifest requires scratch initialization")
    if payload.get("pretrained_checkpoint") is not None:
        raise ValueError("main multifragment manifest forbids a pretrained checkpoint")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("multifragment manifest must contain exactly 16 samples")
    ids = [str(row.get("sample_id")) for row in samples]
    if len(set(ids)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("multifragment sample IDs must be unique")
    if payload.get("train_sample_ids") != ids or payload.get("validation_sample_ids") != ids:
        raise ValueError("train and validation must contain the same ordered 16 samples")
    scenes = {str(row.get("scene_id")) for row in samples}
    if scenes != {"scene_000000"}:
        raise ValueError(f"expected only scene_000000, got {sorted(scenes)}")
    fragment_counts = Counter(int(row.get("fragment_id", -1)) for row in samples)
    frame_counts = Counter(int(row.get("frame_id", -1)) for row in samples)
    if fragment_counts != Counter({value: 4 for value in EXPECTED_FRAGMENTS}):
        raise ValueError(f"each fragment must occur four times: {dict(fragment_counts)}")
    if frame_counts != Counter({value: 4 for value in EXPECTED_FRAMES}):
        raise ValueError(f"each frame must occur four times: {dict(frame_counts)}")
    meshes: dict[int, set[str]] = defaultdict(set)
    for row in samples:
        fragment = int(row["fragment_id"])
        meshes[fragment].add(str(row.get("fragment_mesh_sha256")))
        if int(row.get("fragment_num_faces", -1)) < int(min_num_faces):
            raise ValueError(f"fragment {fragment} has fewer than {min_num_faces} faces")
        if int(row.get("shell_point_count", -1)) < 128:
            raise ValueError(f"sample {row['sample_id']} has fewer than 128 shell points")
        if int(row.get("fracture_point_count", -1)) != 0:
            raise ValueError(f"sample {row['sample_id']} passes fracture points to the model")
        if row.get("registration_point_selection") != "shell_only":
            raise ValueError("multifragment manifest requires shell_only")
        if not bool(row.get("T_W_from_C_available")):
            raise ValueError(f"sample {row['sample_id']} has no T_W_from_C")
        if row.get("data_contract_errors"):
            raise ValueError(f"sample {row['sample_id']} has data contract errors")
    if any(len(values) != 1 or "None" in values for values in meshes.values()):
        raise ValueError("physical mesh SHA must stay constant over four views")
    expected_hash = manifest_content_sha256(payload)
    if payload.get("manifest_sha256") != expected_hash:
        raise ValueError("multifragment manifest internal SHA256 mismatch")
    return {
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "scene_id": "scene_000000",
        "fragment_counts": dict(sorted(fragment_counts.items())),
        "frame_counts": dict(sorted(frame_counts.items())),
        "fragment_mesh_sha256": {key: next(iter(value)) for key, value in meshes.items()},
    }


def validate_multifragment_config(config: Mapping[str, Any]) -> None:
    data = config.get("data", {})
    if not bool(data.get("multifragment_contract", False)):
        return
    for key, expected in WARNING_FLAGS.items():
        if config.get(key) is not expected:
            raise ValueError(f"multifragment config requires {key}=true")
    experiment_type = config.get("experiment_type")
    if experiment_type not in {
        "four_fragments_four_views_overfit",
        "four_fragments_four_views_warmstart_control",
    }:
        raise ValueError("multifragment config has wrong experiment_type")
    if experiment_type == "four_fragments_four_views_overfit" and (
        config.get("initialization_mode") != "scratch"
        or config.get("pretrained_checkpoint") is not None
    ):
        raise ValueError("main multifragment config must initialize from scratch")
    if str(data.get("validation_manifest")) != "same_as_train":
        raise ValueError("multifragment validation_manifest must be same_as_train")
    if int(data.get("expected_selected_samples", -1)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("multifragment config expects exactly 16 samples")
    train = config.get("train", {})
    effective = int(data.get("train_batch_size", -1)) * int(
        train.get("gradient_accumulation_steps", -1)
    )
    if effective != EXPECTED_SAMPLE_COUNT:
        raise ValueError("one optimizer step must cover exactly 16 samples")
    loss = config.get("loss", {}).get("joint_surface_correspondence_pose_v3", {})
    if loss.get("loss_reduction") != "per_sample_mean_then_batch_mean":
        raise ValueError("multifragment loss must use equal per-sample reduction")
    cache = config.get("static_geometry_cache", {})
    cache_enabled = bool(cache.get("enabled", False)) if isinstance(cache, Mapping) else bool(cache)
    if cache_enabled and bool(config.get("augmentations", {}).get("enabled", False)):
        raise ValueError("static_geometry_cache requires augmentations.enabled=false")
    if cache_enabled and bool(config.get("frozen_feature_cache", {}).get("enabled", False)):
        raise ValueError("optimized scratch run forbids learned feature caching")


MULTIFRAGMENT_THRESHOLDS = {
    "strict_surface_gate": {
        "correspondence_p95_mm": 1.0, "alignment_p95_mm": 1.0,
        "rotation_error_deg": 0.25, "translation_error_mm": 0.10,
    },
    "practical_surface_gate": {
        "correspondence_p95_mm": 2.5, "alignment_p95_mm": 2.5,
        "rotation_error_deg": 1.0, "translation_error_mm": 0.50,
    },
    "pose_placement_gate": {
        "rotation_error_deg": 1.0, "translation_error_mm": 0.50,
        "surface_membership_p95_mm": 0.1,
    },
}


def multifragment_sample_score(row: Mapping[str, Any]) -> float:
    return (
        float(row["exact_global_projected_correspondence_p95_mm"]) / 2.5
        + float(row["exact_global_projection_alignment_p95_mm"]) / 2.5
        + float(row["exact_global_projection_rotation_error_deg"]) / 1.0
        + float(row["exact_global_projection_translation_error_mm"]) / 0.5
    )


def worst_multifragment_sample_score(rows: Sequence[Mapping[str, Any]]) -> float:
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("worst multifragment score requires all 16 samples")
    return max(map(multifragment_sample_score, rows))


def _sample_gate(row: Mapping[str, Any], name: str) -> dict[str, Any]:
    threshold = MULTIFRAGMENT_THRESHOLDS[name]
    checks = {
        "rank_three": int(float(row["exact_global_projection_rank"])) == 3,
        "k16_exact_global_recall": float(row["k16_exact_global_triangle_recall"]) >= 0.995 - 1e-6,
        "k16_zero_fallback": float(row["k16_fallback_fraction"]) <= 1e-6,
        "rotation_error_deg": float(row["exact_global_projection_rotation_error_deg"]) <= threshold["rotation_error_deg"] + 1e-6,
        "translation_error_mm": float(row["exact_global_projection_translation_error_mm"]) <= threshold["translation_error_mm"] + 1e-6,
    }
    if "correspondence_p95_mm" in threshold:
        checks["correspondence_p95_mm"] = float(row["exact_global_projected_correspondence_p95_mm"]) <= threshold["correspondence_p95_mm"] + 1e-6
        checks["alignment_p95_mm"] = float(row["exact_global_projection_alignment_p95_mm"]) <= threshold["alignment_p95_mm"] + 1e-6
    if "surface_membership_p95_mm" in threshold:
        checks["surface_membership_p95_mm"] = float(row["exact_global_surface_membership_p95_mm"]) <= threshold["surface_membership_p95_mm"] + 1e-6
    return {
        "sample_id": str(row["sample_id"]), "fragment_id": int(float(row["fragment_id"])),
        "frame_id": int(float(row["frame_id"])), "checks": checks,
        "passed": all(checks.values()),
        "failures": [key for key, value in checks.items() if not value],
    }


def multifragment_stage_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("multifragment gates require exactly 16 samples")
    identities = {(int(float(row["fragment_id"])), int(float(row["frame_id"]))) for row in rows}
    expected = {(fragment, frame) for fragment in EXPECTED_FRAGMENTS for frame in EXPECTED_FRAMES}
    if identities != expected:
        raise ValueError("gate rows do not form the required 4x4 Cartesian product")
    result: dict[str, Any] = {}
    for name in MULTIFRAGMENT_THRESHOLDS:
        per_sample = [_sample_gate(row, name) for row in rows]
        passed_fragments = [fragment for fragment in EXPECTED_FRAGMENTS if all(
            item["passed"] for item in per_sample if item["fragment_id"] == fragment
        )]
        passed_frames = [frame for frame in EXPECTED_FRAMES if all(
            item["passed"] for item in per_sample if item["frame_id"] == frame
        )]
        worst = max(rows, key=multifragment_sample_score)
        fragment_scores = {fragment: max(multifragment_sample_score(row) for row in rows if int(float(row["fragment_id"])) == fragment) for fragment in EXPECTED_FRAGMENTS}
        frame_scores = {frame: max(multifragment_sample_score(row) for row in rows if int(float(row["frame_id"])) == frame) for frame in EXPECTED_FRAMES}
        stage_passed = all(item["passed"] for item in per_sample)
        result[name] = {
            "gate_name": name, "thresholds": MULTIFRAGMENT_THRESHOLDS[name],
            "stage_passed": stage_passed, "next_stage_allowed": stage_passed,
            "passed_sample_count": sum(item["passed"] for item in per_sample),
            "passed_fragment_count": len(passed_fragments), "passed_frame_count": len(passed_frames),
            "passed_fragments": passed_fragments, "passed_frames": passed_frames,
            "worst_sample": str(worst["sample_id"]),
            "worst_fragment": max(fragment_scores, key=fragment_scores.get),
            "worst_frame": max(frame_scores, key=frame_scores.get),
            "per_sample": per_sample,
            "failures": [item["sample_id"] for item in per_sample if not item["passed"]],
        }
    practical = result["practical_surface_gate"]
    return {
        **result, "stage_passed": practical["stage_passed"],
        "next_stage_allowed": practical["stage_passed"],
        "transition_policy": "practical_surface_gate",
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(map(float, values))
    def quantile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)
    return {"mean": sum(ordered) / len(ordered), "p50": quantile(0.5), "p95": quantile(0.95), "max": max(ordered)}


def aggregate_metrics(rows: Sequence[Mapping[str, Any]], group_key: str) -> list[dict[str, Any]]:
    """Aggregate physical scalar metrics by fragment or frame."""
    if group_key not in {"fragment_id", "frame_id"}:
        raise ValueError("group_key must be fragment_id or frame_id")
    metric_fields = (
        "exact_global_projected_correspondence_p95_mm",
        "exact_global_projection_alignment_p95_mm",
        "exact_global_projection_rotation_error_deg",
        "exact_global_projection_translation_error_mm",
    )
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(float(row[group_key]))].append(row)
    output = []
    for group, selected in sorted(groups.items()):
        item: dict[str, Any] = {group_key: group, "sample_count": len(selected)}
        for field in metric_fields:
            item.update({f"{field}_{suffix}": value for suffix, value in _numeric_summary([float(row[field]) for row in selected]).items()})
        item["pose_success_rate"] = sum(_sample_gate(row, "pose_placement_gate")["passed"] for row in selected) / len(selected)
        item["surface_success_rate"] = sum(_sample_gate(row, "practical_surface_gate")["passed"] for row in selected) / len(selected)
        item["passed_other_axis_count"] = sum(_sample_gate(row, "practical_surface_gate")["passed"] for row in selected)
        output.append(item)
    return output


def equal_sample_weights(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(samples) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("equal 4x4 weighting requires 16 samples")
    weights = {str(row["sample_id"]): 1.0 / EXPECTED_SAMPLE_COUNT for row in samples}
    fragment = Counter()
    frame = Counter()
    for row in samples:
        fragment[int(row["fragment_id"])] += weights[str(row["sample_id"])]
        frame[int(row["frame_id"])] += weights[str(row["sample_id"])]
    return {"per_sample": weights, "per_fragment": dict(fragment), "per_frame": dict(frame)}


def diagnose_multifragment_failure(rows: Sequence[Mapping[str, Any]], *, best_is_final=False, improving=False) -> dict[str, Any]:
    gates = multifragment_stage_gates(rows)
    if gates["practical_surface_gate"]["stage_passed"]:
        return {"diagnosis": "practical_gate_passed"}
    passed = {str(row["sample_id"]): _sample_gate(row, "practical_surface_gate")["passed"] for row in rows}
    failures_by_fragment = Counter(int(float(row["fragment_id"])) for row in rows if not passed[str(row["sample_id"])])
    failures_by_frame = Counter(int(float(row["frame_id"])) for row in rows if not passed[str(row["sample_id"])])
    if best_is_final and improving:
        diagnosis = "undertrained_but_improving"
    elif failures_by_fragment.get(2, 0) == 0 and sum(failures_by_fragment.values()) > 0:
        diagnosis = "single_fragment_representation_bias"
    elif sum(count == 4 for count in failures_by_fragment.values()) == 1:
        diagnosis = "fragment_geometry_specific_failure"
    elif sum(count >= 3 for count in failures_by_frame.values()) == 1:
        diagnosis = "viewpoint_specific_failure"
    elif sum(failures_by_fragment.values()) == 1:
        diagnosis = "sample_or_annotation_failure"
    elif all(_sample_gate(row, "pose_placement_gate")["passed"] for row in rows):
        diagnosis = "pose_good_surface_tail_failure"
    else:
        diagnosis = "shared_model_capacity_or_optimization_limit"
    return {"diagnosis": diagnosis, "failed_samples_by_fragment": dict(failures_by_fragment), "failed_samples_by_frame": dict(failures_by_frame)}


__all__ = [
    "EXPECTED_FRAGMENTS", "EXPECTED_FRAMES", "EXPECTED_SAMPLE_COUNT",
    "MULTIFRAGMENT_THRESHOLDS", "aggregate_metrics", "diagnose_multifragment_failure", "equal_sample_weights",
    "manifest_content_sha256", "multifragment_sample_score",
    "multifragment_stage_gates", "validate_multifragment_config",
    "validate_multifragment_manifest_payload", "worst_multifragment_sample_score",
]
