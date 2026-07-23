#!/usr/bin/env python3
"""Train the production CoordinateGuidedSurfaceRegistrationV3 model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import apply_overrides, load_config  # noqa: E402
from symm_template_reg.engine.production_trainer import (  # noqa: E402
    run_production_training,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--work-dir")
    parser.add_argument("--resume")
    parser.add_argument(
        "--from-scratch", action="store_true",
        help="require a fresh deterministic model/optimizer/scheduler state",
    )
    parser.add_argument("--cfg-options", nargs="*")
    args = parser.parse_args()
    if args.from_scratch and args.resume:
        parser.error("--from-scratch is mutually exclusive with --resume")
    config = apply_overrides(load_config(args.config), args.cfg_options)
    if config.get("runtime") != "production":
        raise ValueError("tools/train.py accepts production configs only")
    result = run_production_training(
        config,
        device_name=args.device,
        work_dir_override=args.work_dir,
        resume=args.resume,
        from_scratch=args.from_scratch,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
