"""Crash-resilient incremental history for epoch-based debug overfit runs."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Mapping


STANDARD_FIELDS = (
    "timestamp",
    "record_type",
    "run_id",
    "epoch",
    "global_step",
    "batch_step",
    "optimizer_step",
    "samples_seen",
    "phase",
    "stage_name",
    "frozen_module_prefixes",
    "trainable_parameter_count",
    "train/loss_total",
    "train/loss_symmetry_pose",
    "train/loss_pose_classification",
    "eval/symmetry_pose_loss",
    "eval/pose_query_classification_loss",
    "eval/top1_rotation_error_deg",
    "eval/oracle_topK_rotation_error_deg",
    "eval/top1_translation_total_mm",
    "eval/oracle_topK_translation_total_mm",
    "eval/query_positive_accuracy",
    "eval/duplicate_query_fraction",
    "learning_rate",
    "gradient_norm",
    "amp_scale",
    "epoch_time_sec",
    "data_time_sec",
    "gpu_memory_allocated_mb",
    "gpu_memory_reserved_mb",
    "gpu_peak_allocated_mb",
    "gpu_peak_reserved_mb",
    "is_best",
    "current_best_metric",
    "best_epoch",
    "best_checkpoint",
    "debug_visualization_paths",
    "warnings",
)


class TrainingHistory:
    def __init__(
        self,
        run_dir: str | Path,
        run_id: str,
        config: Mapping[str, Any],
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.path = self.run_dir / str(
            self.config.get("filename", "history/history.jsonl")
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every_record = bool(
            self.config.get("flush_every_record", True)
        )
        self.fsync = bool(self.config.get("fsync", False))
        self.epoch_rows: list[dict[str, Any]] = []

    def record(self, record_type: str, **values: Any) -> dict[str, Any]:
        payload = {field: None for field in STANDARD_FIELDS}
        payload.update(values)
        payload.update(
            {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "record_type": str(record_type),
                "run_id": self.run_id,
            }
        )
        if payload["debug_visualization_paths"] is None:
            payload["debug_visualization_paths"] = []
        if payload["warnings"] is None:
            payload["warnings"] = []
        if self.enabled:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
                if self.flush_every_record:
                    stream.flush()
                if self.fsync:
                    os.fsync(stream.fileno())
        if record_type in {"train_epoch", "eval_epoch"}:
            self.epoch_rows.append(payload)
        return payload

    def write_epoch_csv(self) -> Path | None:
        if not bool(self.config.get("save_epoch_csv", True)):
            return None
        destination = self.run_dir / "history" / "epoch_metrics.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in self.epoch_rows for key in row})
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.epoch_rows)
        return destination


__all__ = ["TrainingHistory"]
