"""Independent readiness gates for coarse patch classification."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


CANDIDATE_THRESHOLDS = {
    "valid_patch_set_top4_recall": 0.995,
    "valid_patch_set_in_candidate_set_fraction": 0.995,
    "minimum_unique_predicted_patches": 2,
    "maximum_most_popular_patch_fraction": 0.8,
}
TOP1_THRESHOLDS = {"valid_patch_set_top1_accuracy": 0.95}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def candidate_stage_gate(
    metrics: Mapping[str, Any],
    *,
    nonfinite_detected: bool,
    target_leakage_detected: bool | None,
    capacity_audit_passed: bool,
) -> dict[str, Any]:
    """Gate that alone authorizes teacher-forced local Stage B."""

    top4 = metrics.get("valid_patch_set_top4_recall")
    included = metrics.get(
        "valid_patch_set_in_candidate_set_fraction", top4
    )
    unique = metrics.get("unique_predicted_patches")
    popular = metrics.get("most_popular_patch_fraction")
    checks = {
        "valid_patch_set_top4_recall": _finite(top4) and float(top4) >= 0.995,
        "valid_patch_set_in_candidate_set_fraction": (
            _finite(included) and float(included) >= 0.995
        ),
        "unique_predicted_patches_gt_1": _finite(unique) and float(unique) > 1,
        "most_popular_patch_fraction_lt_0_8": (
            _finite(popular) and float(popular) < 0.8
        ),
        "nonfinite_false": not bool(nonfinite_detected),
        "target_leakage_false": target_leakage_detected is False,
        "capacity_audit_passed": bool(capacity_audit_passed),
    }
    passed = all(checks.values())
    return {
        "gate_name": "candidate_stage_gate",
        "candidate_stage_passed": passed,
        "stage_passed": passed,
        "next_stage_allowed": passed,
        "authorizes_teacher_forced_local_stage_b": passed,
        "thresholds": dict(CANDIDATE_THRESHOLDS),
        "metrics": dict(metrics),
        "checks": checks,
        "failures": [name for name, value in checks.items() if not value],
        "nonfinite_detected": bool(nonfinite_detected),
        "target_leakage_detected": target_leakage_detected,
        "capacity_audit_passed": bool(capacity_audit_passed),
    }


def top1_quality_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Diagnostic ranking-quality gate; it never authorizes/blocks Stage B."""

    value = metrics.get("valid_patch_set_top1_accuracy")
    passed = _finite(value) and float(value) >= 0.95
    return {
        "gate_name": "top1_quality_gate",
        "top1_quality_passed": passed,
        "diagnostic_only": True,
        "blocks_teacher_forced_local_stage_b": False,
        "thresholds": dict(TOP1_THRESHOLDS),
        "metrics": dict(metrics),
        "checks": {"valid_patch_set_top1_accuracy": passed},
        "failures": [] if passed else ["valid_patch_set_top1_accuracy"],
    }


def write_patch_stage_gates(
    output_dir: str | Path,
    metrics: Mapping[str, Any],
    *,
    nonfinite_detected: bool,
    target_leakage_detected: bool | None,
    capacity_audit_passed: bool,
    extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Write the two gates plus a backward-compatible linking gate."""

    output = Path(output_dir)
    candidate = candidate_stage_gate(
        metrics,
        nonfinite_detected=nonfinite_detected,
        target_leakage_detected=target_leakage_detected,
        capacity_audit_passed=capacity_audit_passed,
    )
    top1 = top1_quality_gate(metrics)
    if extra:
        candidate.update(dict(extra))
        top1.update(dict(extra))
    legacy = {
        "stage_passed": candidate["candidate_stage_passed"],
        "next_stage_allowed": candidate["candidate_stage_passed"],
        "candidate_stage_passed": candidate["candidate_stage_passed"],
        "top1_quality_passed": top1["top1_quality_passed"],
        "thresholds": {
            "candidate_stage": dict(CANDIDATE_THRESHOLDS),
            "top1_quality_diagnostic": dict(TOP1_THRESHOLDS),
        },
        "candidate_stage_gate_path": str(output / "candidate_stage_gate.json"),
        "top1_quality_gate_path": str(output / "top1_quality_gate.json"),
        "top1_quality_gate_blocks_stage_b": False,
        "failures": candidate["failures"],
    }
    if extra:
        legacy.update(dict(extra))
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("candidate_stage_gate.json", candidate),
        ("top1_quality_gate.json", top1),
        ("stage_gate.json", legacy),
    ):
        (output / name).write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )
    return candidate, top1, legacy


__all__ = [
    "CANDIDATE_THRESHOLDS",
    "TOP1_THRESHOLDS",
    "candidate_stage_gate",
    "top1_quality_gate",
    "write_patch_stage_gates",
]
