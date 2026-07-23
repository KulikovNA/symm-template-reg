#!/usr/bin/env python3
"""Run only explicitly requested view-ladder configs/manifests/seeds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import apply_overrides, load_config  # noqa: E402
from symm_template_reg.engine.overfit_trainer import run_overfit_training  # noqa: E402
from symm_template_reg.engine.view_ladder import view_scaling_summary  # noqa: E402


def _num_pose_outputs(config: dict) -> int:
    model = config["model"]
    if model.get("type") == "UniformCorrespondenceProcrustesReg":
        return 1
    if "residual_pose_head" in model:
        return int(model["residual_pose_head"]["num_hypotheses"])
    return int(model["pose_head"]["num_queries"])


def _best_rows(run_dir: Path) -> tuple[dict, list[dict]]:
    best = json.loads(
        (run_dir / "checkpoints/best_metrics.json").read_text(encoding="utf-8")
    )
    epoch = int(best["epoch"])
    path = run_dir / "evaluations" / f"epoch_{epoch:04d}" / "per_sample_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        return best, list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--init-modules", nargs="+")
    parser.add_argument("--cfg-options", nargs="*")
    parser.add_argument(
        "--baseline-summary",
        help="per_run_summary.csv from the matching final-layer-only baseline",
    )
    args = parser.parse_args()
    requested_config = apply_overrides(load_config(args.config), args.cfg_options)
    requested_auxiliary = float(
        requested_config["loss"].get("pose_decoder_auxiliary_weight", 0.0)
    )
    if requested_auxiliary > 0 and not args.baseline_summary:
        raise ValueError(
            "auxiliary decoder ablation requires --baseline-summary from "
            "the matching final-layer-only run"
        )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    num_views = len(manifest["samples"])
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    run_rows = []
    for seed in args.seeds:
        config = apply_overrides(load_config(args.config), args.cfg_options)
        config["seed"] = int(seed)
        # Model/data-loader seed changes between replicas; point selection must
        # remain byte-for-byte deterministic across those replicas.
        config["dataset"]["random_seed"] = 0
        config["data"]["train_manifest"] = str(manifest_path)
        config["data"]["validation_manifest"] = "same_as_train"
        config["data"]["expected_selected_samples"] = num_views
        config["debug_visualization"]["num_samples"] = num_views
        config["experiment"]["name"] = (
            f"{config['experiment']['name']}_{manifest_path.stem}_seed{seed}"
        )
        if args.max_optimizer_steps is not None:
            config["train"]["max_optimizer_steps"] = int(args.max_optimizer_steps)
        result = run_overfit_training(
            config,
            device_name=args.device,
            init_checkpoint=args.init_checkpoint,
            init_modules=args.init_modules,
        )
        run_dir = Path(result["run_dir"])
        best, rows = _best_rows(run_dir)
        scaling = view_scaling_summary(
            rows, num_queries=_num_pose_outputs(config)
        )
        metrics = best["metrics"]
        run_rows.append(
            {
                **scaling,
                "seed": int(seed),
                "run_dir": str(run_dir),
                "best_checkpoint": result["best_checkpoint"],
                "optimizer_step": int(result["optimizer_step"]),
                "best_epoch": int(result["best_epoch"]),
                "top1_pose_success_5deg_5mm": metrics.get(
                    "eval/top1_pose_success_5deg_5mm"
                ),
                "pose_success_2deg_2mm": metrics.get(
                    "eval/pose_success_2deg_2mm",
                    metrics.get("eval/top1_pose_success_2deg_2mm"),
                ),
                "physical_normalized_score": metrics.get(
                    "eval/physical_normalized_score"
                ),
                "rotation_response_ratio": metrics.get(
                    "eval/rotation_response_ratio"
                ),
                "base_pose_static_fraction": metrics.get(
                    "eval/base_pose_static_fraction"
                ),
                "rotation_context_pairwise_distance": metrics.get(
                    "eval/rotation_context_pairwise_distance"
                ),
                "gt_pose_pairwise_rotation_deg": metrics.get(
                    "eval/gt_pose_pairwise_rotation_deg"
                ),
                "world_axis_spread_deg": metrics.get(
                    "eval/oracle_world_axis_spread_deg"
                ),
                "world_translation_spread_mm": metrics.get(
                    "eval/oracle_world_translation_range_mm"
                ),
                "auxiliary_decoder_weight": float(
                    config["loss"].get("pose_decoder_auxiliary_weight", 0.0)
                ),
                "base_pose_source": config["model"].get(
                    "base_pose_source", "legacy_absolute_queries"
                ),
                "min_sample_exposures": result.get("min_sample_exposures"),
                "mean_sample_exposures": result.get("mean_sample_exposures"),
                "max_sample_exposures": result.get("max_sample_exposures"),
                "target_sample_exposures": result.get("target_sample_exposures"),
                "computed_max_optimizer_steps": result.get(
                    "computed_max_optimizer_steps"
                ),
            }
        )
    fields = list(run_rows[0])
    for filename in ("per_run_summary.csv", "per_seed_summary.csv"):
        with (output / filename).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(run_rows)
    with (output / "view_scaling_curve.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(run_rows)
    summary = {
        "config": str(Path(args.config).expanduser().resolve()),
        "manifest": str(manifest_path),
        "num_views": num_views,
        "seeds": args.seeds,
        "criterion": "pose success 2deg/2mm must equal 1.0 for every seed",
        "criterion_passed": all(
            float(row["pose_success_2deg_2mm"]) >= 1.0 for row in run_rows
        ),
        "runs": run_rows,
    }
    if num_views == 1:
        (output / "one_frame_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    if any(float(row["auxiliary_decoder_weight"]) > 0 for row in run_rows):
        assert args.baseline_summary is not None
        with Path(args.baseline_summary).expanduser().resolve().open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            baseline_rows = list(csv.DictReader(stream))
        ablation_rows = [
            {"variant": "final_layer_only", **row} for row in baseline_rows
        ] + [{"variant": "auxiliary_decoder_0p5", **row} for row in run_rows]
        with (output / "aux_decoder_ablation.csv").open(
            "x", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["variant", *fields], extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(ablation_rows)
    report = [
        "# View scaling run",
        "",
        f"- views: `{num_views}`",
        f"- seeds: `{args.seeds}`",
        f"- one-frame gate passed: `{summary['criterion_passed']}`",
        "",
        "This runner executes only the supplied config, manifest and seeds. It does "
        "not advance to another ladder level automatically.",
        "",
    ]
    (output / "view_scaling_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
