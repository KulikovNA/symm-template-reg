"""Shortcut/failure gates for correspondence and hybrid research stages."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def evaluate_correspondence_diagnostic_gates(
    metrics: Mapping[str, Any],
    config: Mapping[str, Any] | None,
    *,
    min_sample_exposures: float,
    model_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return {"failed": False, "diagnosis": None}
    if min_sample_exposures < float(cfg.get("min_sample_exposures", 0)):
        return {"failed": False, "diagnosis": None, "reason": "insufficient_sample_exposures"}
    failures = []
    constant = dict(cfg.get("correspondence_constant_output", {}))
    if bool(constant.get("enabled", False)) and float(
        metrics.get("eval/correspondence_context_pairwise_distance", math.inf)
    ) <= float(constant.get("min_pairwise_distance_m", 1e-5)):
        failures.append("correspondence_constant_output")
    confidence = dict(cfg.get("confidence_collapse", {}))
    if bool(confidence.get("enabled", False)) and float(
        metrics.get("eval/effective_correspondence_count", math.inf)
    ) < float(confidence.get("minimum_effective_point_count", 3.0)):
        failures.append("confidence_collapse")
    rank = dict(cfg.get("procrustes_rank_failure", {}))
    if bool(rank.get("enabled", False)) and float(
        metrics.get("eval/correspondence_pose_rank_valid", 1.0)
    ) < 1.0:
        failures.append("procrustes_rank_failure")
    leakage = dict(cfg.get("correspondence_target_leakage", {}))
    if bool(leakage.get("enabled", False)) and bool(
        metrics.get("eval/target_leakage_detected", False)
    ):
        failures.append("correspondence_target_leakage")
    residual_static = dict(cfg.get("residual_static_codebook", {}))
    if bool(residual_static.get("enabled", False)):
        static_fraction = float(
            metrics.get("eval/hybrid_residual_static_fraction", 0.0)
        )
        nonidentity = (
            float(metrics.get("eval/hybrid_residual_rotation_deg", 0.0))
            >= float(residual_static.get("minimum_nonidentity_rotation_deg", 0.1))
            or float(metrics.get("eval/hybrid_residual_translation_mm", 0.0))
            >= float(residual_static.get("minimum_nonidentity_translation_mm", 0.1))
        )
        if static_fraction > float(
            residual_static.get("max_static_fraction", 0.25)
        ) and nonidentity:
            failures.append("residual_static_codebook")
    saturation = dict(cfg.get("residual_bound_saturation", {}))
    if bool(saturation.get("enabled", False)):
        head = dict((model_config or {}).get("base_pose_head", {}))
        rotation_bound = float(head.get("max_rotation_correction_deg", math.inf))
        translation_bound = float(head.get("max_translation_correction_m", math.inf)) * 1000.0
        if (
            float(metrics.get("eval/hybrid_residual_rotation_deg", 0.0)) >= 0.99 * rotation_bound
            or float(metrics.get("eval/hybrid_residual_translation_mm", 0.0)) >= 0.99 * translation_bound
        ):
            failures.append("residual_bound_saturation")
    return {
        "failed": bool(failures),
        "diagnosis": failures[0] if failures else None,
        "failures": failures,
        "action": cfg.get("action", "stop_with_diagnosis") if failures else "continue",
    }


__all__ = ["evaluate_correspondence_diagnostic_gates"]
