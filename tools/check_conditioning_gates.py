#!/usr/bin/env python3
"""Check the four-view conditioned K1 pose and input-response gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument(
        "--conditioning-summaries", nargs="+", required=True
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    with Path(args.run_summary).expanduser().resolve().open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        runs = list(csv.DictReader(stream))
    interventions = {}
    for path_value in args.conditioning_summaries:
        path = Path(path_value).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        interventions[str(Path(payload["run"]).resolve())] = payload
    rows = []
    for run in runs:
        run_dir = Path(run["run_dir"]).resolve()
        best = json.loads(
            (run_dir / "checkpoints/best_metrics.json").read_text(encoding="utf-8")
        )["metrics"]
        audit = interventions.get(str(run_dir))
        if audit is None:
            raise ValueError(f"missing conditioning audit for {run_dir}")
        success_2 = float(best["eval/top1_pose_success_2deg_2mm"])
        success_5 = float(best["eval/top1_pose_success_5deg_5mm"])
        static = float(best.get("eval/base_pose_static_fraction", 1.0))
        response = float(best.get("eval/rotation_response_ratio", 0.0))
        permutation_rotation = float(
            audit["input_permutation_equivariance_rotation_error_deg"]
        )
        permutation_translation = float(
            audit["input_permutation_equivariance_translation_error_mm"]
        )
        passed = (
            (success_2 >= 1.0 or success_5 >= 0.9)
            and static <= 1e-6
            and response > 0.05
            and permutation_rotation < 0.1
            and permutation_translation < 0.1
            and audit["diagnosis"] not in {"static_query_codebook", "centroid_only_shortcut"}
        )
        rows.append(
            {
                "seed": int(run["seed"]),
                "run_dir": str(run_dir),
                "top1_pose_success_2deg_2mm": success_2,
                "top1_pose_success_5deg_5mm": success_5,
                "base_pose_static_fraction": static,
                "rotation_response_ratio": response,
                "permutation_rotation_error_deg": permutation_rotation,
                "permutation_translation_error_mm": permutation_translation,
                "conditioning_diagnosis": audit["diagnosis"],
                "gate_passed": passed,
            }
        )
    with (output / "conditioning_gate_per_seed.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "criterion": (
            "(top1 2deg/2mm=1 or top1 5deg/5mm>=0.9), static_fraction=0, "
            "rotation_response_ratio>0.05, permutation error<0.1deg/0.1mm"
        ),
        "all_seeds_passed": all(row["gate_passed"] for row in rows),
        "runs": rows,
    }
    (output / "conditioning_gate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_seeds_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
