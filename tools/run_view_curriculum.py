#!/usr/bin/env python3
"""Manually run 1 -> 2 -> 4 -> 10 views with model-only initialization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import load_config  # noqa: E402
from symm_template_reg.engine.overfit_trainer import run_overfit_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifests", nargs=4, required=True, metavar=("V1", "V2", "V4", "V10"))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifests = [Path(value).expanduser().resolve() for value in args.manifests]
    expected = [1, 2, 4, 10]
    report = []
    for seed in args.seeds:
        previous_checkpoint = None
        for level, (manifest_path, expected_views) in enumerate(zip(manifests, expected), start=1):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if len(payload.get("samples", [])) != expected_views:
                raise ValueError(f"{manifest_path} must contain {expected_views} views")
            config = load_config(args.config)
            config["seed"] = seed
            config["dataset"]["random_seed"] = 0
            config["data"]["train_manifest"] = str(manifest_path)
            config["data"]["validation_manifest"] = "same_as_train"
            config["data"]["expected_selected_samples"] = expected_views
            config["experiment"]["name"] += f"_v{expected_views}_seed{seed}"
            result = run_overfit_training(
                config,
                device_name=args.device,
                init_checkpoint=previous_checkpoint,
            )
            previous_checkpoint = result["best_checkpoint"]
            report.append(
                {
                    "seed": seed,
                    "level": level,
                    "views": expected_views,
                    "manifest": str(manifest_path),
                    "init_checkpoint": None if level == 1 else report[-1]["best_checkpoint"],
                    "best_checkpoint": previous_checkpoint,
                    "run_dir": result["run_dir"],
                    "optimizer_restored": False,
                }
            )
    payload = {"comparison": "manual progressive curriculum", "runs": report}
    (output / "curriculum_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
