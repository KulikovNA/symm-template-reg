#!/usr/bin/env python3
"""Re-evaluate an existing local B1--B4 checkpoint without mutating its run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symm_template_reg.engine.checkpoint import load_checkpoint  # noqa: E402
from symm_template_reg.engine.overfit_manifest import load_faces840_manifest  # noqa: E402
from symm_template_reg.engine.overfit_trainer import (  # noqa: E402
    _amp_settings,
    _build_pose_criterion,
    _evaluate,
    _write_evaluation,
)
from symm_template_reg.evaluation.local_stage import check_local_substage  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.registry import (  # noqa: E402
    COLLATE_FUNCTIONS,
    DATASETS,
    build_from_cfg,
)


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _source_signature(run: Path) -> dict[str, Any]:
    names = (
        "resolved_config.json",
        "checkpoints/best.pth",
        "checkpoints/best_metrics.json",
        "stage_gate.json",
        "final_summary.json",
        "history/history.jsonl",
    )
    result: dict[str, Any] = {}
    for name in names:
        path = run / name
        if not path.is_file():
            result[name] = None
            continue
        stat = path.stat()
        result[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _target_leakage(config: dict[str, Any]) -> tuple[bool | None, str | None]:
    raw = config.get("target_leakage_policy", {}).get("audit_path")
    path = None if not raw else Path(str(raw)).expanduser()
    if path is None or not path.is_file():
        return None, None if path is None else str(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("target_leakage_detected", True)), str(path)


def recheck(
    run: Path,
    checkpoint: Path,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    before = _source_signature(run)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    substage = str(
        config.get("stage_gate_dependencies", {}).get("local_substage", "")
    ).upper()
    if substage not in {"B1", "B2", "B3", "B4"}:
        raise ValueError("source run is not a local B1--B4 substage")

    dataset_cfg = deepcopy(config["dataset"])
    data_cfg = config["data"]
    dataset_cfg["fragment_mesh_cache_dir"] = str(
        output / "cache" / "fragment_mesh_metadata"
    )
    dataset_cfg["fragment_mesh_filter"] = deepcopy(data_cfg["fragment_mesh_filter"])
    dataset_cfg["observed_filter"] = deepcopy(data_cfg["observed_filter"])
    dataset_cfg["symmetry_region_activity"] = deepcopy(
        data_cfg.get("symmetry_region_activity", {})
    )
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    manifest_path = Path(data_cfg["train_manifest"]).expanduser()
    manifest, digest = load_faces840_manifest(manifest_path, config, dataset)
    index_by_id = {
        record.sample_id: index for index, record in enumerate(dataset.sample_records)
    }
    indices = [index_by_id[row["sample_id"]] for row in manifest["samples"]]
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(data_cfg.get("validation_batch_size", 2)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    model = build_model(config["model"]).to(device)
    checkpoint_payload = load_checkpoint(
        checkpoint, model=model, map_location=device, strict=True
    )
    epoch = int(checkpoint_payload.get("epoch", 0))
    criterion = _build_pose_criterion(config)
    amp_enabled, amp_dtype, _ = _amp_settings(device, config["train"])
    metrics, rows = _evaluate(
        model,
        loader,
        device,
        criterion,
        amp_enabled,
        amp_dtype,
        config["loss"],
        epoch=epoch,
        max_epochs=None,
    )
    _write_evaluation(output, epoch, metrics, rows)
    best = output / "best_evaluation"
    best.mkdir()
    _json(best / "evaluation_summary.json", {"epoch": epoch, **metrics})
    _write_rows(best / "per_sample_metrics.csv", rows)

    local_metrics = dict(rows[0])
    required_finite = (
        "valid_triangle_set_top1",
        "valid_triangle_set_top4",
        "valid_triangle_candidate_recall",
        "local_triangle_set_ce",
        "local_triangle_random_ce",
        "triangle_target_index_mismatch_fraction",
        "invalid_candidate_count_fraction",
    )
    nonfinite = any(
        key not in local_metrics or not math.isfinite(float(local_metrics[key]))
        for key in required_finite
    )
    leakage, leakage_path = _target_leakage(config)
    gate = check_local_substage(
        substage,
        local_metrics,
        nonfinite_detected=nonfinite,
        target_leakage_detected=leakage,
    )
    gate.update(
        source_run_dir=str(run),
        source_checkpoint=str(checkpoint),
        checkpoint_epoch=epoch,
        manifest_file_sha256=digest,
        recheck_only=True,
        target_leakage_audit_path=leakage_path,
    )
    _json(output / "stage_gate.json", gate)
    if not gate["stage_passed"]:
        _json(output / "diagnostic_failure.json", gate)

    if substage == "B1":
        _json(
            output / "triangle_target_contract.json",
            {
                key: local_metrics.get(key)
                for key in (
                    "triangle_target_index_mismatch_fraction",
                    "valid_triangle_candidate_recall",
                    "min_local_candidate_count",
                    "max_local_candidate_count",
                    "invalid_candidate_count_fraction",
                    "duplicate_local_candidate_fraction",
                    "teacher_forcing_selected_symmetry_element",
                )
            },
        )
        _json(
            output / "triangle_classifier_metrics.json",
            {"epoch": epoch, **local_metrics},
        )
        _json(
            output / "random_baseline.json",
            {
                "local_triangle_set_ce": local_metrics.get("local_triangle_set_ce"),
                "random_cross_entropy": local_metrics.get("local_triangle_random_ce"),
                "warning": (
                    "local_triangle_classifier_worse_than_uniform"
                    if not gate["checks"].get("loss_below_random", False)
                    else None
                ),
            },
        )

    after = _source_signature(run)
    unchanged = before == after
    summary = {
        "run_status": "ok",
        "source_run_dir": str(run),
        "source_checkpoint": str(checkpoint),
        "source_run_unchanged": unchanged,
        "source_signature_before": before,
        "source_signature_after": after,
        "checkpoint_epoch": epoch,
        "local_substage": substage,
        "stage_passed": gate["stage_passed"],
        "failures": gate["failures"],
        "checks": gate["checks"],
        "output_dir": str(output),
    }
    _json(output / "recheck_summary.json", summary)
    _json(output / "final_summary.json", summary)
    if not unchanged:
        raise RuntimeError("source run changed during read-only recheck")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run = Path(args.run_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else (
        run / "checkpoints" / "best.pth"
    )
    output = Path(args.output_dir).expanduser().resolve()
    register_all_modules()
    summary = recheck(run, checkpoint, output, torch.device(args.device))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
