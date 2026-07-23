"""Production train/validation loop for CoordinateGuidedSurfaceRegistrationV3."""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from symm_template_reg.datasets import LengthBucketBatchSampler
from symm_template_reg.models import build_model, register_all_modules
from symm_template_reg.models.losses.clean_coordinate_pose_loss_v3 import (
    CleanCoordinatePoseLossV3,
)
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg

from .runtime import move_to_device
from .production_evaluator import (
    evaluate_production,
    format_validation_report,
    write_evaluation_report,
    write_validation_tracking,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _unique_run_dir(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        candidate = root / (
            f"{name}_{stamp}" if not suffix else f"{name}_{stamp}_{suffix:03d}"
        )
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"cannot allocate run directory below {root}")


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parameter_groups(
    model: torch.nn.Module,
    *,
    default_lr: float,
    prefix_learning_rates: Mapping[str, float],
) -> list[dict[str, Any]]:
    rules = tuple(
        (str(prefix), float(lr))
        for prefix, lr in prefix_learning_rates.items()
    )
    groups: dict[tuple[str, float], list[torch.nn.Parameter]] = defaultdict(list)
    matched = {prefix: 0 for prefix, _ in rules}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        selected = [
            (prefix, lr)
            for prefix, lr in rules
            if name == prefix or name.startswith(prefix + ".")
        ]
        if len(selected) > 1:
            raise ValueError(f"overlapping optimizer prefixes for {name}: {selected}")
        label, lr = selected[0] if selected else ("<default>", default_lr)
        if selected:
            matched[label] += parameter.numel()
        groups[(label, lr)].append(parameter)
    missing = [prefix for prefix, count in matched.items() if count == 0]
    if missing:
        raise ValueError(f"optimizer prefixes matched no parameters: {missing}")
    return [
        {"params": parameters, "lr": lr, "group_name": label}
        for (label, lr), parameters in groups.items()
    ]


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    max_steps: int,
    warmup_steps: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def factor(step_index: int) -> float:
        completed = step_index + 1
        if completed <= warmup_steps:
            return completed / max(warmup_steps, 1)
        progress = (completed - warmup_steps) / max(max_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        minimum = min_lr / max(max(base_lrs), 1e-12)
        return minimum + (1.0 - minimum) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    batch_in_epoch: int,
    batch_step: int,
    optimizer_step: int,
    samples_seen: int,
    best_metric: float,
    train_dataset: Any,
    validation_dataset: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "amp_scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "batch_step": int(batch_step),
        "optimizer_step": int(optimizer_step),
        "global_step": int(optimizer_step),
        "samples_seen": int(samples_seen),
        "best_metric": float(best_metric),
        "train_dataset_index_fingerprint": train_dataset.index_fingerprint,
        "validation_dataset_index_fingerprint": validation_dataset.index_fingerprint,
        "resolved_config": deepcopy(dict(config)),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _loader(
    dataset: Any,
    *,
    collate: Any,
    batch_size: int,
    shuffle: bool,
    data_config: Mapping[str, Any],
    seed: int,
) -> tuple[DataLoader, LengthBucketBatchSampler]:
    sampler = LengthBucketBatchSampler(
        dataset.observed_lengths,
        batch_size,
        shuffle=shuffle,
        drop_last=bool(data_config.get("drop_last", False)),
        bucket_size_multiplier=int(data_config.get("bucket_size_multiplier", 20)),
        seed=seed,
    )
    workers = int(data_config.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "collate_fn": collate,
        "num_workers": workers,
        "pin_memory": bool(data_config.get("pin_memory", True)),
        "persistent_workers": bool(
            data_config.get("persistent_workers", True) and workers > 0
        ),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(data_config.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs), sampler


def _loss(
    prediction: Any,
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
    optimizer_step: int,
) -> dict[str, Any]:
    values = dict(config)
    values.pop("type", None)
    values.pop("enabled", None)
    values["current_epoch"] = int(optimizer_step)
    criterion = CleanCoordinatePoseLossV3(**values)
    return criterion(
        predicted_normalized_O=prediction.correspondence_auxiliary[
            "fine_aux_coordinate_normalized"
        ],
        observed_points_C=batch["observed"]["points_C"],
        target_points_O=batch["gt"]["points_O_corresponding"],
        valid_mask=prediction.observed_valid_mask,
        gt_pose_T_C_from_O=batch["gt"]["T_C_from_O"],
        symmetry_metadata=batch["template_symmetry_metadata"],
        effective_symmetry_groups=batch["gt"]["effective_symmetry_group"],
        template_mesh_vertices_O=batch["template_mesh_vertices_O"],
    )


def run_production_training(
    config: Mapping[str, Any],
    *,
    device_name: str,
    work_dir_override: str | Path | None,
    resume: str | Path | None,
    from_scratch: bool,
) -> dict[str, Any]:
    if from_scratch == (resume is not None):
        raise ValueError("select exactly one of --from-scratch or --resume")
    resolved = deepcopy(dict(config))
    dataset_root = resolved.get("data", {}).get("dataset_root")
    if not dataset_root:
        dataset_root = os.environ.get("FRAG_DATASET_ROOT")
    if not dataset_root:
        raise ValueError(
            "dataset root is required via data.dataset_root or FRAG_DATASET_ROOT"
        )
    work_root = (
        Path(work_dir_override).expanduser().resolve()
        if work_dir_override is not None
        else Path(os.environ.get("FRAG_WORK_DIR", "work_dirs")).expanduser().resolve()
    )
    seed = int(resolved.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    register_all_modules()
    name = str(
        resolved.get("experiment", {}).get(
            "name", "coordinate_guided_surface_v3"
        )
    )
    run_dir = _unique_run_dir(work_root, name)
    data_cfg = deepcopy(dict(resolved["data"]))
    index_root = run_dir / "dataset_indexes"
    train_dataset_cfg = deepcopy(dict(data_cfg["train"]))
    validation_dataset_cfg = deepcopy(dict(data_cfg["validation"]))
    train_split = str(data_cfg.get("train_split", "train"))
    validation_split = str(data_cfg.get("validation_split", "val"))
    validation_cfg = dict(resolved.get("validation", {}))
    evaluation_role = str(
        validation_cfg.get(
            "evaluation_role",
            "overfit_validation"
            if validation_split == "train"
            else "validation",
        )
    )
    if train_split != "train":
        raise ValueError("production trainer always requires train_split='train'")
    if validation_split != "val" and not str(
        resolved.get("config_role", "")
    ).startswith("debug_"):
        raise ValueError(
            "validation_split other than 'val' is allowed only by an explicit "
            "debug config"
        )
    for split, dataset_cfg in (
        (train_split, train_dataset_cfg),
        (validation_split, validation_dataset_cfg),
    ):
        dataset_cfg["dataset_root"] = str(dataset_root)
        dataset_cfg["split"] = split
        dataset_cfg["index_cache_dir"] = str(index_root / split)
    train_dataset = build_from_cfg(train_dataset_cfg, DATASETS)
    validation_dataset = build_from_cfg(validation_dataset_cfg, DATASETS)
    train_dataset.write_index_artifacts(run_dir / "train_dataset")
    validation_dataset.write_index_artifacts(run_dir / "validation_dataset")
    collate = build_from_cfg(resolved["collate"], COLLATE_FUNCTIONS)
    train_loader, train_sampler = _loader(
        train_dataset,
        collate=collate,
        batch_size=int(data_cfg["train_batch_size"]),
        shuffle=True,
        data_config=data_cfg,
        seed=seed,
    )
    validation_loader, validation_sampler = _loader(
        validation_dataset,
        collate=collate,
        batch_size=int(data_cfg["validation_batch_size"]),
        shuffle=False,
        data_config={**data_cfg, "drop_last": False},
        seed=seed,
    )
    model = build_model(resolved["model"]).to(device)
    train_cfg = dict(resolved["train"])
    optimizer_cfg = dict(resolved["optimizer"])
    if optimizer_cfg.pop("type") != "AdamW":
        raise ValueError("production trainer supports AdamW only")
    default_lr = float(optimizer_cfg.pop("lr"))
    groups = _parameter_groups(
        model,
        default_lr=default_lr,
        prefix_learning_rates=optimizer_cfg.pop("prefix_learning_rates"),
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=default_lr,
        weight_decay=float(optimizer_cfg.pop("weight_decay")),
        **optimizer_cfg,
    )
    if optimizer_cfg:
        raise ValueError(f"unsupported optimizer fields: {sorted(optimizer_cfg)}")
    max_steps = int(train_cfg["max_optimizer_steps"])
    max_epochs = int(train_cfg["max_epochs"])
    scheduler_cfg = dict(resolved["scheduler"])
    scheduler = _scheduler(
        optimizer,
        max_steps=max_steps,
        warmup_steps=int(scheduler_cfg["warmup_optimizer_steps"]),
        min_lr=float(scheduler_cfg["min_lr"]),
    )
    amp_enabled = bool(train_cfg.get("amp", False))
    if amp_enabled:
        raise ValueError(
            "main production config is fp32; mixed precision requires a separate audit"
        )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    counters = {
        "epoch": 0,
        "batch_in_epoch": 0,
        "batch_step": 0,
        "optimizer_step": 0,
        "samples_seen": 0,
    }
    best_metric = math.inf
    if resume is not None:
        payload = torch.load(
            Path(resume).expanduser().resolve(),
            map_location=device,
            weights_only=False,
        )
        if (
            payload["train_dataset_index_fingerprint"]
            != train_dataset.index_fingerprint
            or payload["validation_dataset_index_fingerprint"]
            != validation_dataset.index_fingerprint
        ):
            raise ValueError("resume dataset index fingerprints do not match")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        for key in counters:
            counters[key] = int(payload.get(key, 0))
        best_metric = float(payload.get("best_metric", math.inf))
        rng_state = payload.get("rng_state", {})
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.random.set_rng_state(rng_state["torch"])
            if device.type == "cuda" and rng_state.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
    resolved["data"]["dataset_root"] = str(dataset_root)
    resolved["resolved_runtime"] = {
        "device": str(device),
        "run_dir": str(run_dir),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "train_split": train_split,
        "validation_split": validation_split,
        "validation_evaluation_role": evaluation_role,
        "train_dataset_index_fingerprint": train_dataset.index_fingerprint,
        "validation_dataset_index_fingerprint": validation_dataset.index_fingerprint,
    }
    _atomic_json(run_dir / "resolved_config.json", resolved)
    history_path = run_dir / "history.jsonl"
    accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
    eval_interval = int(train_cfg.get("eval_interval_optimizer_steps", 1000))
    checkpoint_interval = int(
        train_cfg.get("latest_checkpoint_interval_optimizer_steps", eval_interval)
    )
    gradient_clip = float(train_cfg.get("gradient_clip_norm", 1.0))
    augmentation_totals: dict[str, float] = defaultdict(float)
    augmentation_events = 0
    stop = False
    batches_per_epoch = len(train_loader)
    resume_batch = counters["batch_in_epoch"]
    if resume_batch >= batches_per_epoch:
        start_epoch = counters["epoch"] + 1
        resume_batch = 0
    else:
        start_epoch = max(counters["epoch"], 1)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, max_epochs + 1):
        counters["epoch"] = epoch
        skip_batches = resume_batch if epoch == start_epoch else 0
        counters["batch_in_epoch"] = skip_batches
        train_dataset.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        validation_sampler.set_epoch(epoch)
        model.train()
        progress = tqdm(
            train_loader,
            desc=f"train {epoch:04d}/{max_epochs} opt={counters['optimizer_step']}/{max_steps}",
            leave=True,
        )
        for batch_index, batch in enumerate(progress):
            if batch_index < skip_batches:
                continue
            counters["batch_in_epoch"] = batch_index + 1
            counters["batch_step"] += 1
            counters["samples_seen"] += len(batch["sample_id"])
            moved = move_to_device(batch, device)
            prediction = model(moved)
            losses = _loss(
                prediction,
                moved,
                resolved["loss"],
                counters["optimizer_step"],
            )
            loss = losses["loss_total"] / accumulation
            loss.backward()
            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if not should_step:
                continue
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            counters["optimizer_step"] += 1
            for meta in batch["meta"]:
                augmentation = dict(meta.get("augmentation_metadata", {}))
                augmentation_events += 1
                for key in (
                    "removed_shell_count",
                    "added_total_count",
                    "added_fracture_count",
                    "added_depth_ring_count",
                    "removed_fraction",
                    "added_fraction",
                ):
                    augmentation_totals[key] += float(augmentation.get(key, 0.0))
            record = {
                "record_type": "train",
                **counters,
                "loss": float(losses["loss_total"].detach()),
                "gradient_norm": float(gradient_norm),
                "lr": [float(group["lr"]) for group in optimizer.param_groups],
                "warmup_progress": float(losses["warmup_progress"]),
            }
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, default=str) + "\n")
            progress.set_postfix(
                loss=f"{record['loss']:.4f}",
                grad=f"{record['gradient_norm']:.3f}",
            )
            needs_eval = (
                counters["optimizer_step"] % eval_interval == 0
                or counters["optimizer_step"] >= max_steps
            )
            if needs_eval:
                evaluation_dir = (
                    run_dir / "evaluations"
                    / f"step_{counters['optimizer_step']:06d}"
                )
                summary, rows = evaluate_production(
                    model,
                    validation_loader,
                    device,
                    source_dataset_split=validation_split,
                    evaluation_role=evaluation_role,
                    max_batches=validation_cfg.get("max_batches"),
                    candidate_k=int(
                        resolved.get("evaluation", {}).get("candidate_k", 16)
                    ),
                    projection_chunk_size=int(
                        resolved.get("evaluation", {}).get(
                            "projection_chunk_size", 64
                        )
                    ),
                )
                write_evaluation_report(evaluation_dir, summary, rows)
                write_validation_tracking(
                    run_dir,
                    summary,
                    counters,
                    combined_history_path=history_path,
                )
                tqdm.write(format_validation_report(summary, counters))
                metric = float(summary["validation/p90_physical_score"])
                checkpoint = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    batch_in_epoch=counters["batch_in_epoch"],
                    batch_step=counters["batch_step"],
                    optimizer_step=counters["optimizer_step"],
                    samples_seen=counters["samples_seen"],
                    best_metric=min(best_metric, metric),
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    config=resolved,
                )
                _save_checkpoint(run_dir / "checkpoints" / "latest.pth", checkpoint)
                _atomic_json(
                    run_dir / "checkpoints" / "latest_training_state.json",
                    {**counters, "validation": summary},
                )
                if metric < best_metric:
                    best_metric = metric
                    checkpoint["best_metric"] = best_metric
                    _save_checkpoint(
                        run_dir / "checkpoints" / "best.pth", checkpoint
                    )
                    _atomic_json(
                        run_dir / "checkpoints" / "best_metrics.json", summary
                    )
            elif counters["optimizer_step"] % checkpoint_interval == 0:
                checkpoint = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    batch_in_epoch=counters["batch_in_epoch"],
                    batch_step=counters["batch_step"],
                    optimizer_step=counters["optimizer_step"],
                    samples_seen=counters["samples_seen"],
                    best_metric=best_metric,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    config=resolved,
                )
                _save_checkpoint(run_dir / "checkpoints" / "latest.pth", checkpoint)
            if counters["optimizer_step"] >= max_steps:
                stop = True
                break
        resume_batch = 0
        if stop:
            break
    if not (run_dir / "checkpoints" / "latest.pth").is_file():
        checkpoint = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=counters["epoch"],
            batch_in_epoch=counters["batch_in_epoch"],
            batch_step=counters["batch_step"],
            optimizer_step=counters["optimizer_step"],
            samples_seen=counters["samples_seen"],
            best_metric=best_metric,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            config=resolved,
        )
        _save_checkpoint(run_dir / "checkpoints" / "latest.pth", checkpoint)
    augmentation_statistics = {
        "events": augmentation_events,
        **{
            f"mean_{key}": value / max(augmentation_events, 1)
            for key, value in sorted(augmentation_totals.items())
        },
    }
    _atomic_json(run_dir / "augmentation_statistics.json", augmentation_statistics)
    final = {
        "status": "ok",
        "run_dir": str(run_dir),
        **counters,
        "best_metric_name": "validation/p90_physical_score",
        "best_metric": best_metric,
        "stop_reason": (
            "max_optimizer_steps"
            if counters["optimizer_step"] >= max_steps
            else "max_epochs"
        ),
        "test_was_indexed": False,
        "test_was_evaluated": False,
        "augmentation_statistics": augmentation_statistics,
    }
    _atomic_json(run_dir / "final_summary.json", final)
    return final


__all__ = ["run_production_training"]
