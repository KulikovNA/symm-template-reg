#!/usr/bin/env python3
"""Summarize pose-cost and ranking-target diagnostics from an evaluation CSV."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path


FIELDS = (
    "pose_cost_min",
    "pose_cost_max",
    "pose_cost_mean",
    "pose_cost_std",
    "ranking_target_entropy",
    "ranking_predicted_entropy",
    "ranking_target_max_probability",
    "ranking_predicted_max_probability",
    "top1_query_is_oracle",
    "ranking_regret",
    "score_vs_negative_pose_cost_spearman",
)


def _parse_metric_value(value: str) -> float:
    normalized = value.strip().lower()
    if normalized == "true":
        return 1.0
    if normalized == "false":
        return 0.0
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-sample-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with Path(args.per_sample_metrics).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    stats = {}
    for field in FIELDS:
        values = [
            _parse_metric_value(row[field])
            for row in rows
            if row.get(field) not in (None, "")
        ]
        stats[field] = {
            "mean": sum(values) / len(values) if values else math.nan,
            "min": min(values) if values else math.nan,
            "max": max(values) if values else math.nan,
        }
    targets = [
        ast.literal_eval(row["ranking_target_distribution"])
        for row in rows
        if row.get("ranking_target_distribution")
    ]
    uniform_warnings = [
        index
        for index, distribution in enumerate(targets)
        if distribution and max(distribution) - min(distribution) < 0.05
    ]
    report = {
        "spearman_semantics": "positive is good: correlation(pose_logit, -pose_cost)",
        "num_samples": len(rows),
        "statistics": stats,
        "target_distributions": targets,
        "nearly_uniform_target_sample_indices": uniform_warnings,
        "warnings": (
            ["nearly uniform ranking targets detected"] if uniform_warnings else []
        ),
    }
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "ranking_diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "ranking_target_statistics.json").write_text(
        json.dumps({"target_distributions": targets, "warnings": report["warnings"]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), "num_samples": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
