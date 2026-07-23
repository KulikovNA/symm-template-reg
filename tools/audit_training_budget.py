#!/usr/bin/env python3
"""Preview optimizer-step and exposure budgets for view-ladder manifests."""

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
from symm_template_reg.engine.training_budget import resolve_training_budget  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = config["data"]
    train = config["train"]
    multi = dict(config.get("multi_view_batch", {}))
    rows = []
    for path in sorted(Path(args.manifest_dir).expanduser().glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            continue
        count = len(samples)
        batch_size = (
            min(int(multi.get("views_per_group", count)), count)
            if bool(multi.get("enabled", False))
            else int(data.get("train_batch_size", 1))
        )
        budget = resolve_training_budget(
            config.get("train_budget"),
            selected_samples=count,
            batch_size=batch_size,
            gradient_accumulation_steps=int(
                train.get("gradient_accumulation_steps", 1)
            ),
            drop_last=bool(data.get("drop_last", False)),
            configured_max_optimizer_steps=(
                int(train["max_optimizer_steps"])
                if train.get("max_optimizer_steps") is not None
                else None
            ),
            configured_max_epochs=int(train["max_epochs"]),
        )
        rows.append({"manifest": str(path.resolve()), **budget.to_dict()})
    if not rows:
        raise ValueError("manifest directory contains no non-empty manifests")
    with (output / "training_budget_audit.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "config": str(Path(args.config).expanduser().resolve()),
        "manifest_dir": str(Path(args.manifest_dir).expanduser().resolve()),
        "budgets": rows,
    }
    (output / "training_budget_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    fields = ("selected_samples", "batch_size", "target_sample_exposures", "computed_max_optimizer_steps")
    print("manifest".ljust(32), *(name.rjust(28) for name in fields))
    for row in rows:
        print(Path(row["manifest"]).stem.ljust(32), *(str(row[name]).rjust(28) for name in fields))
    print(json.dumps({"output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
