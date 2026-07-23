#!/usr/bin/env python3
"""Evaluate a staged-overfit checkpoint with full pose/ranking/region metrics."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from copy import deepcopy
from pathlib import Path

from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import apply_overrides, load_config  # noqa: E402
from symm_template_reg.engine.checkpoint import load_checkpoint  # noqa: E402
from symm_template_reg.engine.overfit_manifest import (  # noqa: E402
    WARNING_FLAGS,
    load_faces840_manifest,
)
from symm_template_reg.engine.overfit_trainer import (  # noqa: E402
    _amp_settings,
    _build_pose_criterion,
    _evaluate,
    _write_evaluation,
)
from symm_template_reg.engine.trainer import resolve_device  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.registry import (  # noqa: E402
    COLLATE_FUNCTIONS,
    DATASETS,
    build_from_cfg,
)


def _unique_output(checkpoint: Path) -> Path:
    root = checkpoint.parent.parent / "manual_evaluations"
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        path = root / (stamp if not suffix else f"{stamp}_{suffix:03d}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"cannot allocate unique evaluation directory below {root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--cfg-options", nargs="*")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.cfg_options)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _unique_output(checkpoint_path)
    )
    register_all_modules()
    dataset_cfg = deepcopy(config["dataset"])
    dataset_cfg["fragment_mesh_cache_dir"] = output.parent / (
        output.name + "_cache"
    )
    data = config["data"]
    dataset_cfg["fragment_mesh_filter"] = deepcopy(data["fragment_mesh_filter"])
    dataset_cfg["observed_filter"] = deepcopy(data["observed_filter"])
    dataset_cfg["symmetry_region_activity"] = deepcopy(
        data.get("symmetry_region_activity", {})
    )
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    manifest_path = Path(args.manifest or data["train_manifest"]).expanduser()
    manifest, digest = load_faces840_manifest(manifest_path, config, dataset)
    indices = {
        record.sample_id: index for index, record in enumerate(dataset.sample_records)
    }
    subset_indices = [indices[sample["sample_id"]] for sample in manifest["samples"]]
    subset = Subset(dataset, subset_indices)
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    loader = DataLoader(
        subset,
        batch_size=int(data.get("validation_batch_size", 2)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    device = resolve_device(args.device)
    model = build_model(config["model"]).to(device)
    load_checkpoint(checkpoint_path, model=model, map_location=device, strict=True)
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
        epoch=0,
        max_epochs=None,
    )
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    report = {
        **WARNING_FLAGS,
        "manifest_file_sha256": digest,
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output),
        "metrics": metrics,
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "per_sample_metrics.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    # Keep the standalone evaluator's compact artifacts identical to the
    # trainer evaluation snapshots.  The root copies are what the runbook asks
    # the user to archive; the epoch_0000 directory preserves provenance.
    _write_evaluation(output, 0, metrics, rows)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
