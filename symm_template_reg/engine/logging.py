"""Append-only JSONL logging with deterministic CSV export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


class MetricLogger:
    def __init__(self, work_dir: str | Path) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.work_dir / "metrics.jsonl"
        self.rows: list[dict[str, Any]] = []

    def log(self, row: Mapping[str, Any]) -> None:
        payload = dict(row)
        self.rows.append(payload)
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def write_csv(self) -> Path:
        path = self.work_dir / "metrics.csv"
        fieldnames = sorted({key for row in self.rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return path


__all__ = ["MetricLogger"]
