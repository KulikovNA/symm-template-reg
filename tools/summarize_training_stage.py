#!/usr/bin/env python3
"""Print a compact immutable summary assembled from one staged run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.run_dir).expanduser().resolve()
    sources = {}
    for relative in (
        "final_summary.json",
        "run_manifest.json",
        "checkpoints/best_metrics.json",
        "gradient_summary.json",
    ):
        path = root / relative
        sources[relative] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        )
    report = {"run_dir": str(root), "sources": sources}
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        with Path(args.output).expanduser().resolve().open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
