#!/usr/bin/env python3
"""Recheck a completed Stage A checkpoint without training or mutating its run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from audit_patch_target_ambiguity import audit_checkpoint  # noqa: E402
from symm_template_reg.evaluation.patch_stage import write_patch_stage_gates  # noqa: E402
from symm_template_reg.models import register_all_modules  # noqa: E402


def _signature(run: Path) -> dict[str, Any]:
    names = (
        "resolved_config.json",
        "checkpoints/best.pth",
        "checkpoints/best_metrics.json",
        "stage_gate.json",
        "final_summary.json",
    )
    result = {}
    for name in names:
        path = run / name
        stat = path.stat()
        result[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def _dependency_status(config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    dependencies = config.get("stage_gate_dependencies", {})
    configured = dependencies.get("parameterization_capacity_path")
    paths = configured if isinstance(configured, list) else [configured]
    required = str(
        dependencies.get(
            "parameterization_capacity_required_field", "free_capacity_passed"
        )
    )
    reports = []
    for raw in paths:
        payload = None
        if raw and Path(str(raw)).is_file():
            payload = json.loads(Path(str(raw)).read_text(encoding="utf-8"))
        reports.append({"path": raw, "required_field": required, "payload": payload})
    passed = bool(reports) and all(
        row["payload"] is not None and bool(row["payload"].get(required, False))
        for row in reports
    )
    return passed, {"parameterization_capacity": reports}


def recheck(
    run: Path, checkpoint: Path, output: Path, device: torch.device
) -> dict[str, Any]:
    before = _signature(run)
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    old_final = json.loads((run / "final_summary.json").read_text(encoding="utf-8"))
    summary, _ = audit_checkpoint(run, checkpoint, output, device)
    audit_path = config.get("target_leakage_policy", {}).get("audit_path")
    leakage = None
    if audit_path and Path(str(audit_path)).is_file():
        leakage_payload = json.loads(Path(str(audit_path)).read_text(encoding="utf-8"))
        leakage = bool(leakage_payload.get("target_leakage_detected", True))
    capacity_passed, dependencies = _dependency_status(config)
    extra = {
        "source_run_dir": str(run),
        "source_checkpoint": str(checkpoint),
        "source_run_status": old_final.get("run_status", old_final.get("status")),
        "target_leakage_audit_path": audit_path,
        "dependency_reports": dependencies,
    }
    candidate, top1, _ = write_patch_stage_gates(
        output,
        summary,
        nonfinite_detected=bool(summary["nonfinite_detected"]),
        target_leakage_detected=leakage,
        capacity_audit_passed=capacity_passed,
        extra=extra,
    )
    after = _signature(run)
    unchanged = before == after
    result = {
        "run_status": "ok",
        "technical_run_completed": old_final.get("status") == "ok",
        "stage_readiness": (
            "passed" if candidate["candidate_stage_passed"] else "failed"
        ),
        "candidate_stage_passed": candidate["candidate_stage_passed"],
        "top1_quality_passed": top1["top1_quality_passed"],
        "single_owner_top1": summary["single_owner_top1_accuracy"],
        "valid_set_top1": summary["valid_patch_set_top1_accuracy"],
        "valid_set_top4": summary["valid_patch_set_top4_recall"],
        "source_run_unchanged": unchanged,
        "source_signature_before": before,
        "source_signature_after": after,
        "diagnostic_failure_path": (
            None
            if candidate["candidate_stage_passed"]
            else str(output / "candidate_stage_gate.json")
        ),
        "output_dir": str(output),
    }
    (output / "recheck_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output / "final_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if not unchanged:
        raise RuntimeError("source run changed during read-only recheck")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run = Path(args.run_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    register_all_modules()
    result = recheck(run, checkpoint, output, torch.device(args.device))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
