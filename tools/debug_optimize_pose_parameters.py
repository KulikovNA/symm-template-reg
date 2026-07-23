#!/usr/bin/env python3
"""Optimize raw pose parameters directly, without encoders or pose queries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import load_config  # noqa: E402
from symm_template_reg.engine.overfit_manifest import load_faces840_manifest  # noqa: E402
from symm_template_reg.engine.overfit_trainer import _build_dataset  # noqa: E402
from symm_template_reg.engine.trainer import resolve_device  # noqa: E402
from symm_template_reg.engine.view_ladder import (  # noqa: E402
    direct_optimize_pose_parameters,
)
from symm_template_reg.models import register_all_modules  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", nargs="+", type=int, default=(4, 8, 6))
    parser.add_argument("--num-starts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    register_all_modules()
    config = load_config(REPO_ROOT / "configs/debug/single_fragment/01_k8_pose_only.py")
    config["data"]["train_manifest"] = str(Path(args.manifest).expanduser().resolve())
    config["data"]["validation_manifest"] = "same_as_train"
    config["dataset"]["fragment_mesh_cache_dir"] = str(output / "cache")
    dataset = _build_dataset(config)
    manifest, manifest_file_sha = load_faces840_manifest(
        args.manifest, config, dataset
    )
    record_indices = {
        record.sample_id: index
        for index, record in enumerate(dataset.sample_records)
    }
    by_frame = {
        int(sample["frame_id"]): dataset[record_indices[sample["sample_id"]]]
        for sample in manifest["samples"]
    }
    missing = sorted(set(args.frames) - set(by_frame))
    if missing:
        raise ValueError(f"requested frames absent from manifest: {missing}")

    device = resolve_device(args.device)
    rows = []
    frame_summaries = []
    for frame in args.frames:
        sample = by_frame[int(frame)]
        results = direct_optimize_pose_parameters(
            gt_pose=sample["gt"]["T_C_from_O"],
            observed_points_C=sample["observed"]["points_C"],
            symmetry_metadata=sample["template"]["symmetry_metadata"],
            effective_group=sample["gt"]["effective_symmetry_group"],
            num_starts=args.num_starts,
            steps=args.steps,
            learning_rate=args.learning_rate,
            seed=args.seed + int(frame),
            device=device,
        )
        for result in results:
            rows.append({"frame_id": int(frame), **result})
        successes = sum(bool(result["success_0p1deg_0p1mm"]) for result in results)
        frame_summaries.append(
            {
                "frame_id": int(frame),
                "successes": successes,
                "starts": len(results),
                "criterion_passed": successes >= min(15, len(results)),
                "max_rotation_error_deg": max(
                    float(result["rotation_error_deg"]) for result in results
                ),
                "max_translation_error_mm": max(
                    float(result["translation_error_mm"]) for result in results
                ),
            }
        )
    with (output / "direct_pose_optimization.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = all(item["criterion_passed"] for item in frame_summaries)
    summary = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "manifest_file_sha256": manifest_file_sha,
        "device": str(device),
        "num_starts": args.num_starts,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "criterion": "at least 15/16 starts below 0.1 degree and 0.1 mm",
        "criterion_passed": passed,
        "diagnosis": (
            "pose_codec_rotation_loss_math_passed"
            if passed
            else "pose_codec_or_rotation_loss_problem"
        ),
        "frames": frame_summaries,
    }
    (output / "direct_pose_optimization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **summary}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
