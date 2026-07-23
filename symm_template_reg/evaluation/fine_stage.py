"""Pure fine-stage gate semantics used by training reports and tests."""

from __future__ import annotations

import math
from collections.abc import Mapping


def fine_coordinate_gate(
    metrics: Mapping[str, float],
    tolerance: float = 1e-6,
    minimum_feature_variance: float = 1e-8,
) -> dict:
    p95 = float(metrics.get("aux_coordinate_p95_mm", math.inf))
    rmse = float(metrics.get("aux_coordinate_rmse_mm", math.inf))
    variance = float(metrics.get("fine_feature_variance", 0.0))
    # Older callers predate collision logging; preserve their gate semantics.
    # Current training/evaluation reports always provide the metric explicitly.
    collision = float(metrics.get("fine_feature_collision_fraction", 0.0))
    leakage = bool(metrics.get("target_leakage_detected", False))
    thresholds = {
        "aux_coordinate_p95_mm": {"operator": "<=", "value": 1.0},
        "aux_coordinate_rmse_mm": {"operator": "<=", "value": 0.5},
        "target_leakage_detected": {"operator": "==", "value": False},
        "fine_feature_collision_fraction": {"operator": "==", "value": 0.0},
        "fine_feature_variance": {
            "operator": ">", "value": float(minimum_feature_variance)
        },
    }
    checks = {
        "aux_coordinate_p95_mm": p95 <= 1.0 + tolerance,
        "aux_coordinate_rmse_mm": rmse <= 0.5 + tolerance,
        "target_leakage_detected": not leakage,
        "fine_feature_collision_fraction": (
            math.isfinite(collision) and collision <= tolerance
        ),
        "fine_feature_variance": (
            math.isfinite(variance) and variance > float(minimum_feature_variance)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def fine_triangle_gate(metrics: Mapping[str, float], tolerance: float = 1e-6) -> dict:
    checks = {
        "valid_triangle_set_top1": float(metrics.get("valid_triangle_set_top1_accuracy", -math.inf)) >= .95 - tolerance,
        "valid_triangle_set_top4": float(metrics.get("valid_triangle_set_top4_recall", -math.inf)) >= .995 - tolerance,
        "candidate_recall": float(metrics.get("candidate_recall", -math.inf)) >= 1.0 - tolerance,
        "target_index_match": float(metrics.get("triangle_target_index_mismatch_fraction", math.inf)) <= tolerance,
    }
    return {"passed": all(checks.values()), "checks": checks}


__all__ = ["fine_coordinate_gate", "fine_triangle_gate"]
