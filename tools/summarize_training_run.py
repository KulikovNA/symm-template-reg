#!/usr/bin/env python3
"""Summarize JSONL training/evaluation metrics for one debug run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    work = Path(args.work_dir).resolve()
    rows = [
        json.loads(line)
        for line in (work / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train = [row for row in rows if row.get("phase") == "train"]
    evaluation = [row for row in rows if row.get("phase") == "eval"]
    summary = {
        "debug_training_on_test_split": True,
        "results_are_not_final_evaluation": True,
        "work_dir": str(work),
        "train_steps": len(train),
        "initial_loss": train[0].get("loss_total") if train else None,
        "final_loss": train[-1].get("loss_total") if train else None,
        "evaluation_events": len(evaluation),
        "final_evaluation": evaluation[-1] if evaluation else None,
    }
    (work / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
