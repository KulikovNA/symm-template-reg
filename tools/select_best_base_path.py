#!/usr/bin/env python3
"""Compare completed K1 base ablations without starting another run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _float(row: dict[str, str], name: str, default: float) -> float:
    value = row.get(name)
    return default if value in {None, "", "None"} else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=per_run_summary.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidates = []
    for expression in args.candidate:
        name, path_value = expression.split("=", 1)
        path = Path(path_value).expanduser().resolve()
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        passed = all(
            _float(row, "top1_pose_success_5deg_5mm", 0) >= 0.9
            and _float(row, "rotation_response_ratio", 0) >= 0.5
            and _float(row, "base_pose_static_fraction", 1) <= 0.0
            and (
                _float(row, "gt_pose_pairwise_rotation_deg", 0) < 5.0
                or _float(row, "rotation_context_pairwise_distance", 0) > 1e-6
            )
            and _float(row, "world_axis_spread_deg", float("inf")) <= 10.0
            and _float(row, "world_translation_spread_mm", float("inf")) <= 10.0
            and _float(row, "min_sample_exposures", 0)
                >= _float(row, "target_sample_exposures", float("inf"))
            for row in rows
        )
        candidates.append(
            {
                "name": name,
                "summary": str(path),
                "all_seed_readiness_passed": passed,
                "mean_success_5deg_5mm": sum(_float(row, "top1_pose_success_5deg_5mm", 0) for row in rows) / len(rows),
                "mean_rotation_response_ratio": sum(_float(row, "rotation_response_ratio", 0) for row in rows) / len(rows),
                "mean_rotation_deg": sum(_float(row, "mean_rotation_deg", float("inf")) for row in rows) / len(rows),
            }
        )
    candidates.sort(
        key=lambda row: (
            row["all_seed_readiness_passed"], row["mean_success_5deg_5mm"],
            row["mean_rotation_response_ratio"], -row["mean_rotation_deg"],
        ),
        reverse=True,
    )
    payload = {"selected": candidates[0], "candidates": candidates}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if candidates[0]["all_seed_readiness_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
