#!/usr/bin/env python3
"""Profile one forward/backward per batch size without optimizer updates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import apply_overrides, load_config  # noqa: E402
from symm_template_reg.engine.runtime import move_to_device  # noqa: E402
from symm_template_reg.engine.production_trainer import _loss  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(2, 4, 8, 16))
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cfg-options", nargs="*")
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = apply_overrides(load_config(args.config), args.cfg_options)
    register_all_modules()
    dataset_cfg = deepcopy(config["data"]["train"])
    dataset_cfg.update(
        dataset_root=args.dataset_root,
        split="train",
        max_samples=args.max_samples,
        index_cache_dir=str(output / "dataset_index"),
        boundary_augmentation={"enabled": False},
        point_sampling="farthest_point_up_to_max",
    )
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    target_effective = int(config["data"].get("effective_batch_size", 16))
    rows = []
    for batch_size in args.batch_sizes:
        model = None
        try:
            loader = DataLoader(
                dataset,
                batch_size=int(batch_size),
                shuffle=False,
                num_workers=0,
                collate_fn=collate,
            )
            batch = next(iter(loader))
            moved = move_to_device(batch, device)
            model = build_model(config["model"]).to(device).train()
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = model(moved)
            losses = _loss(prediction, moved, config["loss"], optimizer_step=0)
            losses["loss_total"].backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            valid = moved["observed"]["valid_mask"]
            points = int(valid.sum())
            capacity = int(valid.numel())
            rows.append(
                {
                    "batch_size": int(batch_size),
                    "status": "ok",
                    "gradient_accumulation_steps": max(
                        1, math.ceil(target_effective / int(batch_size))
                    ),
                    "effective_batch_size": int(batch_size)
                    * max(1, math.ceil(target_effective / int(batch_size))),
                    "peak_gpu_memory_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else 0
                    ),
                    "runtime_seconds": elapsed,
                    "samples_per_second": len(batch["sample_id"]) / elapsed,
                    "points_per_second": points / elapsed,
                    "padding_ratio": 1.0 - points / max(capacity, 1),
                    "loss": float(losses["loss_total"].detach()),
                }
            )
        except torch.OutOfMemoryError as error:
            rows.append(
                {
                    "batch_size": int(batch_size),
                    "status": "cuda_out_of_memory",
                    "error": str(error),
                }
            )
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    successful = [row for row in rows if row["status"] == "ok"]
    selected = (
        max(successful, key=lambda row: row["samples_per_second"])
        if successful
        else None
    )
    report = {
        "config": str(Path(args.config).resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "device": str(device),
        "optimizer_steps_performed": 0,
        "rows": rows,
        "selected": selected,
        "selected_batch_size": (
            None if selected is None else selected["batch_size"]
        ),
        "gradient_accumulation_steps": (
            None
            if selected is None
            else selected["gradient_accumulation_steps"]
        ),
        "effective_batch_size": (
            None if selected is None else selected["effective_batch_size"]
        ),
        "peak_gpu_memory": (
            None if selected is None else selected["peak_gpu_memory_bytes"]
        ),
        "samples_per_second": (
            None if selected is None else selected["samples_per_second"]
        ),
        "points_per_second": (
            None if selected is None else selected["points_per_second"]
        ),
        "padding_ratio": (
            None if selected is None else selected["padding_ratio"]
        ),
    }
    (output / "training_profile.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Training batch profile",
        "",
        "Профиль выполняет forward/backward без `optimizer.step`.",
        "",
        "| batch | status | peak GiB | samples/s | points/s | padding | accumulation |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['status']} | "
            f"{row.get('peak_gpu_memory_bytes', 0) / 2**30:.3f} | "
            f"{row.get('samples_per_second', 0):.3f} | "
            f"{row.get('points_per_second', 0):.1f} | "
            f"{row.get('padding_ratio', 0):.3f} | "
            f"{row.get('gradient_accumulation_steps', 0)} |"
        )
    (output / "training_profile.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
