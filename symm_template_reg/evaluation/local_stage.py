"""Readiness gates for isolated local correspondence substages B1--B4."""

from __future__ import annotations

import math
from typing import Any, Mapping

FRACTION_TOLERANCE = 1e-6


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fraction_at_least(value: Any, threshold: float) -> bool:
    return _finite(value) and float(value) >= float(threshold) - FRACTION_TOLERANCE


def _exact_integer(value: Any) -> int | None:
    """Parse an integer metric without rounding or float coercion."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("+", "-")):
            sign, digits = text[0], text[1:]
        else:
            sign, digits = "", text
        if digits.isdigit():
            return int(sign + digits)
    return None


def check_local_substage(
    substage: str,
    metrics: Mapping[str, Any],
    *,
    nonfinite_detected: bool,
    target_leakage_detected: bool | None,
) -> dict[str, Any]:
    """Return a stage-specific gate without unrelated global losses."""

    name = str(substage).upper()
    checks: dict[str, bool] = {
        "nonfinite_false": not bool(nonfinite_detected),
        "target_leakage_false": target_leakage_detected is False,
    }
    thresholds: dict[str, Any]
    if name == "B1":
        thresholds = {
            "valid_triangle_set_top1": 0.95,
            "valid_triangle_set_top4": 0.995,
            "valid_triangle_candidate_recall": 1.0,
            "local_triangle_set_ce": "< local_triangle_random_ce",
            "triangle_target_index_mismatch_fraction": 0.0,
            "duplicate_local_candidate_fraction": 0.0,
            "min_local_candidate_count": 32,
            "max_local_candidate_count": 32,
            "invalid_candidate_count_fraction": 0.0,
        }
        checks.update(
            valid_triangle_set_top1=_finite(metrics.get("valid_triangle_set_top1"))
            and float(metrics["valid_triangle_set_top1"]) >= 0.95,
            valid_triangle_set_top4=_finite(metrics.get("valid_triangle_set_top4"))
            and float(metrics["valid_triangle_set_top4"]) >= 0.995,
            valid_triangle_candidate_recall=_fraction_at_least(
                metrics.get("valid_triangle_candidate_recall"), 1.0
            ),
            loss_below_random=_finite(metrics.get("local_triangle_set_ce"))
            and _finite(metrics.get("local_triangle_random_ce"))
            and float(metrics["local_triangle_set_ce"])
            < float(metrics["local_triangle_random_ce"]),
            no_candidate_index_mismatch=float(
                metrics.get("triangle_target_index_mismatch_fraction", 0.0)
            )
            == 0.0,
            candidates_deduplicated=float(
                metrics.get("duplicate_local_candidate_fraction", math.nan)
            )
            == 0.0,
            shared_symmetry_element_present=float(
                metrics.get("teacher_forcing_selected_symmetry_element", -1)
            )
            >= 0.0,
            min_local_candidate_count_is_32=(
                _exact_integer(metrics.get("min_local_candidate_count")) == 32
            ),
            max_local_candidate_count_is_32=(
                _exact_integer(metrics.get("max_local_candidate_count")) == 32
            ),
            invalid_candidate_count_fraction_zero=_finite(
                metrics.get("invalid_candidate_count_fraction")
            )
            and float(metrics["invalid_candidate_count_fraction"]) == 0.0,
        )
    elif name == "B2":
        thresholds = {
            "barycentric_reconstruction_p95_mm": 0.5,
            "correspondence_p95_mm": 0.5,
            "predicted_to_template_surface_p95_mm": 1e-3,
        }
        checks.update(
            barycentric_reconstruction_p95=_finite(
                metrics.get("barycentric_reconstruction_p95_mm")
            )
            and float(metrics["barycentric_reconstruction_p95_mm"]) <= 0.5,
            canonical_correspondence_p95=_finite(metrics.get("correspondence_p95_mm"))
            and float(metrics["correspondence_p95_mm"]) <= 0.5,
            point_on_triangle=_finite(
                metrics.get("predicted_to_template_surface_p95_mm")
            )
            and float(metrics["predicted_to_template_surface_p95_mm"]) <= 1e-3,
        )
    elif name == "B3":
        thresholds = {
            "correspondence_p95_mm": 0.5,
            "valid_triangle_set_top1": 0.95,
            "correspondence_rank": 3,
            "procrustes_rank_valid": True,
        }
        checks.update(
            correspondence_p95=_finite(metrics.get("correspondence_p95_mm"))
            and float(metrics["correspondence_p95_mm"]) <= 0.5,
            valid_triangle_set_top1=_finite(metrics.get("valid_triangle_set_top1"))
            and float(metrics["valid_triangle_set_top1"]) >= 0.95,
            correspondence_rank=float(metrics.get("correspondence_rank", math.nan)) == 3.0,
            procrustes_rank_valid=float(
                metrics.get("procrustes_rank_valid", math.nan)
            )
            == 1.0,
        )
    elif name == "B4":
        thresholds = {
            "correspondence_p95_mm": 0.5,
            "visible_alignment_p95_mm": 2.0,
            "rotation_error_deg": 0.5,
            "translation_total_mm": 0.5,
            "correspondence_rank": 3,
        }
        for metric, maximum in (
            ("correspondence_p95_mm", 0.5),
            ("visible_alignment_p95_mm", 2.0),
            ("rotation_error_deg", 0.5),
            ("translation_total_mm", 0.5),
        ):
            checks[metric] = _finite(metrics.get(metric)) and float(metrics[metric]) <= maximum
        checks["correspondence_rank"] = float(
            metrics.get("correspondence_rank", math.nan)
        ) == 3.0
        checks["procrustes_rank_valid"] = float(
            metrics.get("procrustes_rank_valid", math.nan)
        ) == 1.0
    else:
        raise ValueError(f"unknown local substage: {substage}")
    passed = all(checks.values())
    return {
        "local_substage": name,
        "stage_passed": passed,
        "next_stage_allowed": passed,
        "thresholds": thresholds,
        "metrics": dict(metrics),
        "checks": checks,
        "failures": [key for key, value in checks.items() if not value],
        "stop_instruction": "STOP: package this substage and request external analysis.",
    }


__all__ = ["check_local_substage"]
