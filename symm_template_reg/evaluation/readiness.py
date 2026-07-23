"""Explicit v2 readiness gates for conditioned K1 and bounded K8."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def k1_readiness(
    metrics: Mapping[str, Any], budget: Mapping[str, Any]
) -> dict[str, bool]:
    target = budget.get("target_sample_exposures")
    reached = target is not None and float(budget.get("min_sample_exposures", 0)) >= float(target)
    valid_early_stop = bool(budget.get("valid_early_stop_after_minimum_exposures", False))
    gt_rotation_response = float(
        metrics.get("eval/gt_pose_pairwise_rotation_deg", 0)
    )
    rotation_context_distance = float(
        metrics.get("eval/rotation_context_pairwise_distance", 0)
    )
    checks = {
        "top1_success_5deg_5mm": float(metrics.get("eval/top1_pose_success_5deg_5mm", 0)) >= 0.9,
        "rotation_response_ratio": float(metrics.get("eval/rotation_response_ratio", 0)) >= 0.5,
        "base_pose_not_static": float(metrics.get("eval/base_pose_static_fraction", 1)) <= 0.0,
        "world_axis_spread": float(metrics.get("eval/world_axis_spread_deg", float("inf"))) <= 10.0,
        "world_translation_range": float(metrics.get("eval/world_translation_range_mm", float("inf"))) <= 10.0,
        "rotation_context_not_collapsed": (
            gt_rotation_response < 5.0 or rotation_context_distance > 1e-6
        ),
        "fair_budget": reached or valid_early_stop,
    }
    checks["passed"] = all(checks.values())
    return checks


def residual_codebook_gate(
    metrics: Mapping[str, Any], *, threshold: float = 0.25
) -> dict[str, bool]:
    checks = {
        "residual_query_static_fraction": float(
            metrics.get("eval/residual_query_static_fraction", 1)
        ) <= 0.25,
        "query_static_codebook_score": float(
            metrics.get("eval/query_static_codebook_score", 1)
        ) <= float(threshold),
    }
    checks["passed"] = all(checks.values())
    return checks


def k8_readiness(
    metrics: Mapping[str, Any], *, k1_base_passed: bool, codebook_threshold: float = 0.25
) -> dict[str, bool]:
    checks = {
        "k1_base_gate": bool(k1_base_passed),
        "oracle_success_5deg_5mm": float(
            metrics.get("eval/oracle_topK_pose_success_5deg_5mm", 0)
        ) >= 0.9,
        **{
            f"codebook_{key}": value
            for key, value in residual_codebook_gate(
                metrics, threshold=codebook_threshold
            ).items()
            if key != "passed"
        },
    }
    checks["passed"] = all(checks.values())
    return checks


__all__ = ["k1_readiness", "k8_readiness", "residual_codebook_gate"]
