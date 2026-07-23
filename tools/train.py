#!/usr/bin/env python3
"""Train the current baseline on a strictly validated debug manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import apply_overrides, load_config  # noqa: E402
from symm_template_reg.engine.trainer import run_training  # noqa: E402
from symm_template_reg.engine.overfit_trainer import run_overfit_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--work-dir")
    parser.add_argument("--manifest")
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--init-modules", nargs="+")
    parser.add_argument(
        "--from-scratch", action="store_true",
        help="require a fresh deterministic model/optimizer/scheduler state",
    )
    parser.add_argument("--cfg-options", nargs="*")
    args = parser.parse_args()
    if args.from_scratch and (args.resume or args.init_checkpoint or args.init_modules):
        parser.error("--from-scratch is mutually exclusive with --resume/--init-checkpoint/--init-modules")
    config = apply_overrides(load_config(args.config), args.cfg_options)
    if "train" in config and "experiment" in config:
        if args.max_steps is not None or args.manifest is not None:
            raise ValueError(
                "epoch-based config uses train.max_epochs/data.train_manifest; "
                "override them with --cfg-options"
            )
        result = run_overfit_training(
            config,
            device_name=args.device,
            work_dir_override=args.work_dir,
            resume=args.resume,
            init_checkpoint=args.init_checkpoint,
            init_modules=args.init_modules,
            from_scratch=args.from_scratch,
        )
    else:
        if args.from_scratch:
            raise ValueError("--from-scratch is supported by epoch-based debug configs")
        result = run_training(
            config,
            device_name=args.device,
            max_steps_override=args.max_steps,
            work_dir_override=args.work_dir,
            manifest_override=args.manifest,
            resume=args.resume,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
