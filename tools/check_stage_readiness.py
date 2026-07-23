#!/usr/bin/env python3
"""Report whether a completed stage satisfies the explicit next-stage gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


GATES = {
    "pose_only": ("eval/oracle_topK_pose_success_5deg_5mm", "max", 0.9),
    "ranking_only": ("eval/top1_query_is_oracle", "max", 0.9),
    "regions_only": ("eval/effective_group_accuracy", "max", 0.9),
    "joint_finetune": ("eval/top1_pose_success_5deg_5mm", "max", 0.9),
}


def check_stage_readiness(stage: str, run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    reasons: list[str] = []
    metrics_path = root / "checkpoints" / "best_metrics.json"
    manifest_path = root / "checkpoints" / "best_manifest.json"
    if not metrics_path.is_file():
        reasons.append(f"missing checkpoint metrics: {metrics_path}")
        metrics: dict[str, Any] = {}
    else:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = dict(payload.get("metrics", payload))
    if not manifest_path.is_file():
        reasons.append(f"missing checkpoint manifest: {manifest_path}")
        checkpoint = None
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = Path(str(manifest.get("checkpoint_path", "")))
        if not checkpoint.is_file():
            reasons.append(f"best checkpoint does not exist: {checkpoint}")
    non_finite = [
        key
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not math.isfinite(float(value))
    ]
    if non_finite:
        reasons.append(f"NaN/Inf metrics: {non_finite}")
    if stage not in GATES:
        reasons.append(f"unknown stage {stage!r}; known stages: {sorted(GATES)}")
        gate = None
    else:
        name, mode, threshold = GATES[stage]
        gate = {"metric": name, "mode": mode, "threshold": threshold}
        value = metrics.get(name)
        if value is None:
            reasons.append(f"required metric is missing: {name}")
        elif mode == "max" and float(value) < threshold:
            reasons.append(f"{name}={float(value):.6g} < {threshold}")
        elif mode == "min" and float(value) > threshold:
            reasons.append(f"{name}={float(value):.6g} > {threshold}")
    return {
        "run_dir": str(root),
        "stage": stage,
        "ready_for_next_stage": not reasons,
        "gate": gate,
        "reasons": reasons,
        "recommendation": (
            "manual transition is allowed; it is never started automatically"
            if not reasons
            else "do not start the next stage"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(GATES))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = check_stage_readiness(args.stage, args.run_dir)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        with Path(args.output).expanduser().resolve().open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0 if report["ready_for_next_stage"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
