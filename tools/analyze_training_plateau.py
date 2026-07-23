#!/usr/bin/env python3
"""Analyze a stopped run for stagnation plus rotation-context collapse."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.evaluation.plateau import (  # noqa: E402
    DEFAULT_PLATEAU_DETECTION,
    detect_rotation_context_plateau,
)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _slope(rows: list[dict[str, Any]], key: str) -> float | None:
    pairs = [(float(row["optimizer_step"]), _finite(row.get(key))) for row in rows]
    pairs = [(x, y) for x, y in pairs if y is not None]
    if len(pairs) < 2:
        return None
    x_mean = statistics.fmean(x for x, _ in pairs)
    y_mean = statistics.fmean(y for _, y in pairs)
    denominator = sum((x - x_mean) ** 2 for x, _ in pairs)
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / max(denominator, 1e-12)


def _loss_modes(values: list[float]) -> list[dict[str, float | int]]:
    if not values:
        return []
    centers = [min(values), max(values)]
    assignments = [0] * len(values)
    for _ in range(20):
        assignments = [min(range(2), key=lambda i: abs(value - centers[i])) for value in values]
        updated = [
            statistics.fmean(value for value, group in zip(values, assignments) if group == index)
            if index in assignments else centers[index]
            for index in range(2)
        ]
        if max(abs(a - b) for a, b in zip(centers, updated)) < 1e-10:
            break
        centers = updated
    return [
        {
            "center": centers[index],
            "count": assignments.count(index),
            "fraction": assignments.count(index) / len(values),
        }
        for index in sorted(range(2), key=centers.__getitem__)
    ]


def analyze_run(run_dir: Path, config: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    history_path = run_dir / "history/history.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    eval_rows = [row for row in history if row.get("record_type") == "eval_epoch"]
    train_rows = [row for row in history if row.get("record_type") == "train_step"]
    if not eval_rows:
        raise ValueError("run contains no evaluation records")
    selected = int(json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))["resolved_runtime"]["train_samples"])
    last_step = max(int(row.get("optimizer_step", 0)) for row in history)
    samples_seen = max(int(row.get("samples_seen", 0)) for row in history)
    min_exposures = samples_seen // selected
    best = json.loads((run_dir / "checkpoints/best_metrics.json").read_text(encoding="utf-8"))
    best_step = int(best["optimizer_step"])
    best_index = max(index for index, row in enumerate(eval_rows) if int(row["optimizer_step"]) <= best_step)
    after_best = eval_rows[best_index + 1 :]
    detector = detect_rotation_context_plateau(
        eval_rows,
        min_sample_exposures=min_exposures,
        config=config or DEFAULT_PLATEAU_DETECTION,
    )
    latest = eval_rows[-1]
    train_losses = [
        float(row["train/loss_total"])
        for row in train_rows
        if _finite(row.get("train/loss_total")) is not None
    ]
    context_diagnostics = json.loads(
        sorted((run_dir / "evaluations").glob("epoch_*/context_conditioning_diagnostics.json"))[-1].read_text(encoding="utf-8")
    )
    best_context_path = run_dir / "evaluations" / f"epoch_{int(best['epoch']):04d}" / "context_conditioning_diagnostics.json"
    best_context = json.loads(best_context_path.read_text(encoding="utf-8"))
    variance = context_diagnostics.get("rotation_context_variance_per_dimension", [])
    best_variance = best_context.get("rotation_context_variance_per_dimension", [])
    summary = {
        **detector,
        "run_dir": str(run_dir),
        "optimizer_steps_completed": last_step,
        "samples_seen": samples_seen,
        "selected_samples": selected,
        "min_sample_exposures": min_exposures,
        "target_sample_exposures": 1500,
        "best_checkpoint_step": best_step,
        "best_checkpoint_min_sample_exposures": int(best["samples_seen"]) // selected,
        "updates_after_best_checkpoint": last_step - best_step,
        "eval_records_total": len(eval_rows),
        "eval_records_after_best": len(after_best),
        "pose_cost_slope_after_best_per_optimizer_step": _slope(after_best, "eval/oracle_best_pose_cost"),
        "rotation_error_slope_after_best_deg_per_optimizer_step": _slope(after_best, "eval/top1_rotation_error_deg"),
        "translation_error_slope_after_best_mm_per_optimizer_step": _slope(after_best, "eval/top1_translation_total_mm"),
        "best_pose_cost": float(best["best_metric"]),
        "latest_rotation_error_deg": _finite(latest.get("eval/top1_rotation_error_deg")),
        "latest_translation_error_mm": _finite(latest.get("eval/top1_translation_total_mm")),
        "best_rotation_context_variance_mean": statistics.fmean(best_variance) if best_variance else None,
        "latest_rotation_context_variance_mean": statistics.fmean(variance) if variance else None,
        "best_rotation_context_collapsed_dimension_fraction": _finite(best["metrics"].get("eval/collapsed_context_dimension_fraction")),
        "latest_rotation_context_collapsed_dimension_fraction": _finite(latest.get("eval/collapsed_context_dimension_fraction")),
        "train_loss_mode_scope": (
            "aggregate_train_steps; the historical run did not record sample_ids per train_step"
        ),
        "train_loss_modes": _loss_modes(train_losses),
    }
    csv_rows = [
        {
            "optimizer_step": int(row["optimizer_step"]),
            "samples_seen": int(row["samples_seen"]),
            "min_sample_exposures": int(row["samples_seen"]) // selected,
            "is_after_best": int(row["optimizer_step"]) > best_step,
            "pose_cost": row.get("eval/oracle_best_pose_cost"),
            "rotation_error_deg": row.get("eval/top1_rotation_error_deg"),
            "translation_error_mm": row.get("eval/top1_translation_total_mm"),
            "rotation_response_ratio": row.get("eval/rotation_response_ratio"),
            "predicted_pairwise_rotation_deg": row.get("eval/base_pose_pairwise_rotation_deg"),
            "gt_pairwise_rotation_deg": row.get("eval/gt_pose_pairwise_rotation_deg"),
            "base_pose_static_fraction": row.get("eval/base_pose_static_fraction"),
            "collapsed_context_dimension_fraction": row.get("eval/collapsed_context_dimension_fraction"),
        }
        for row in eval_rows
    ]
    return summary, csv_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary, rows = analyze_run(run)
    (output / "plateau_analysis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "plateau_analysis.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = [
        "# Training plateau analysis", "",
        f"- status: `{summary['status']}`",
        f"- diagnosis: `{summary['diagnosis']}`",
        f"- optimizer steps: `{summary['optimizer_steps_completed']}`",
        f"- minimum sample exposures: `{summary['min_sample_exposures']}`",
        f"- best checkpoint step: `{summary['best_checkpoint_step']}`",
        f"- updates after best: `{summary['updates_after_best_checkpoint']}`",
        f"- continuing the same training recommended: `{summary['continuing_same_training_recommended']}`",
        "", "The detector requires both metric stagnation and collapsed rotation response; pose loss alone cannot trigger it.", "",
    ]
    (output / "plateau_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
