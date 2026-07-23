#!/usr/bin/env python3
"""Check every best-checkpoint sample against the physical joint-stage gate."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from symm_template_reg.evaluation.joint_stage import check_joint_stage  # noqa: E402

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    report = check_joint_stage(args.run_dir)
    print(json.dumps(report, indent=2))
    return 0 if report["stage_passed"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
