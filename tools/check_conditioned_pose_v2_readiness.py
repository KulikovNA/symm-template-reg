#!/usr/bin/env python3
"""Evaluate fair-budget K1/K8 readiness from one completed run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.evaluation.readiness import k1_readiness, k8_readiness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=("k1", "k8"), required=True)
    parser.add_argument("--k1-base-run")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run).expanduser().resolve()
    best = json.loads((run / "checkpoints/best_metrics.json").read_text(encoding="utf-8"))
    budget = json.loads((run / "training_budget.json").read_text(encoding="utf-8"))
    if args.mode == "k1":
        checks = k1_readiness(best["metrics"], budget)
    else:
        if not args.k1_base_run:
            raise ValueError("K8 readiness requires --k1-base-run")
        base = Path(args.k1_base_run).expanduser().resolve()
        base_best = json.loads((base / "checkpoints/best_metrics.json").read_text(encoding="utf-8"))
        base_budget = json.loads((base / "training_budget.json").read_text(encoding="utf-8"))
        checks = k8_readiness(
            best["metrics"],
            k1_base_passed=k1_readiness(base_best["metrics"], base_budget)["passed"],
        )
    payload = {"run": str(run), "mode": args.mode, "checks": checks}
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if checks["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
