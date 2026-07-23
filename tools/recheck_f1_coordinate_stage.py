#!/usr/bin/env python3
"""Re-evaluate an existing F1 gate without writing into the source run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symm_template_reg.evaluation.fine_stage import fine_coordinate_gate  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_signatures(run: Path) -> dict[str, str]:
    names = (
        "stage_gate.json", "final_summary.json", "coordinate_metrics.json",
        "fine_feature_metrics.json", "resolved_config.json",
        "checkpoints/best_metrics.json", "checkpoints/best.pth",
    )
    return {name: _sha256(run / name) for name in names}


def recheck(run_dir: str | Path, output_dir: str | Path) -> dict:
    run = Path(run_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if not run.is_dir():
        raise FileNotFoundError(run)
    before = _source_signatures(run)
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "coordinate_metrics.json").read_text(encoding="utf-8"))
    leakage_path = config.get("target_leakage_policy", {}).get("audit_path")
    leakage_verified = bool(leakage_path) and Path(str(leakage_path)).is_file()
    leakage = True
    if leakage_verified:
        leakage = bool(json.loads(Path(str(leakage_path)).read_text())["target_leakage_detected"])
    metrics["target_leakage_detected"] = leakage
    gate_cfg = config.get("fine_stage_gate", {})
    fine_gate = fine_coordinate_gate(
        metrics,
        minimum_feature_variance=float(gate_cfg.get("minimum_feature_variance", 1e-8)),
    )
    output.mkdir(parents=True)
    report = {
        "run_status": "ok",
        "stage_readiness": "passed" if fine_gate["passed"] else "failed",
        "stage_passed": bool(fine_gate["passed"]),
        "next_stage_allowed": bool(fine_gate["passed"]),
        "source_run": str(run),
        "source_best_checkpoint": str(run / "checkpoints" / "best.pth"),
        "best_epoch": int(metrics.get("epoch", -1)),
        "thresholds": fine_gate["thresholds"],
        "checks": fine_gate["checks"],
        "failures": fine_gate["failures"],
        "metrics": metrics,
        "target_leakage_audit_path": leakage_path,
        "target_leakage_verified": leakage_verified,
    }
    (output / "stage_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    after = _source_signatures(run)
    integrity = {
        "source_run": str(run), "before": before, "after": after,
        "source_unchanged": before == after,
    }
    (output / "source_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    (output / "recheck_summary.json").write_text(
        json.dumps({**report, "source_unchanged": before == after}, indent=2) + "\n"
    )
    if before != after:
        raise RuntimeError("source F1 run changed during recheck")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = recheck(args.run_dir, args.output_dir)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
