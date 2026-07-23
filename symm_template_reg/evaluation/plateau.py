"""Plateau detection that requires both stagnation and input-response collapse."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_PLATEAU_DETECTION = {
    "enabled": True,
    "min_sample_exposures": 300,
    "patience_eval_records": 10,
    "metric_min_delta": 1e-4,
    "detect_static_rotation": True,
    "max_rotation_response_ratio": 0.01,
    "max_predicted_pairwise_rotation_deg": 1.0,
    "min_gt_pairwise_rotation_deg": 10.0,
    "min_static_fraction": 0.9,
    "action": "stop_with_diagnosis",
}


def _metric(record: Mapping[str, Any], name: str, default: float) -> float:
    value = record.get(name, default)
    return default if value is None else float(value)


def detect_rotation_context_plateau(
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    min_sample_exposures: float,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_PLATEAU_DETECTION, **dict(config or {})}
    patience = int(cfg["patience_eval_records"])
    if not cfg["enabled"] or len(evaluation_records) < patience + 1:
        return {"detected": False, "diagnosis": None, "reason": "insufficient_eval_records"}
    if min_sample_exposures < float(cfg["min_sample_exposures"]):
        return {"detected": False, "diagnosis": None, "reason": "insufficient_sample_exposures"}
    metric_name = "eval/oracle_best_pose_cost"
    values = [_metric(row, metric_name, float("inf")) for row in evaluation_records]
    best_before_window = min(values[:-patience])
    best_in_window = min(values[-patience:])
    improvement = best_before_window - best_in_window
    latest = evaluation_records[-1]
    response = _metric(latest, "eval/rotation_response_ratio", float("inf"))
    predicted_pairwise = _metric(
        latest, "eval/base_pose_pairwise_rotation_deg", float("inf")
    )
    gt_pairwise = _metric(latest, "eval/gt_pose_pairwise_rotation_deg", 0.0)
    static_fraction = _metric(latest, "eval/base_pose_static_fraction", 0.0)
    no_improvement = improvement < float(cfg["metric_min_delta"])
    static_rotation = (
        bool(cfg["detect_static_rotation"])
        and response <= float(cfg["max_rotation_response_ratio"])
        and predicted_pairwise <= float(cfg["max_predicted_pairwise_rotation_deg"])
        and gt_pairwise >= float(cfg["min_gt_pairwise_rotation_deg"])
        and static_fraction >= float(cfg["min_static_fraction"])
    )
    detected = no_improvement and static_rotation
    return {
        "detected": detected,
        "diagnosis": "rotation_context_collapse" if detected else None,
        "status": "plateau_with_rotation_context_collapse" if detected else "no_joint_plateau",
        "continuing_same_training_recommended": not detected,
        "no_improvement": no_improvement,
        "static_rotation": static_rotation,
        "window_improvement": improvement,
        "rotation_response_ratio": response,
        "predicted_pairwise_rotation_deg": predicted_pairwise,
        "gt_pairwise_rotation_deg": gt_pairwise,
        "base_pose_static_fraction": static_fraction,
        "action": cfg["action"] if detected else "continue",
    }


__all__ = ["DEFAULT_PLATEAU_DETECTION", "detect_rotation_context_plateau"]
