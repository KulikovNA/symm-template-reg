#!/usr/bin/env python3
"""Create a compact JSON/CSV/JSONL/MD-only joint-stage report archive."""

from __future__ import annotations
import argparse
import tarfile
from pathlib import Path

ALLOWED = {".json", ".csv", ".jsonl", ".md"}

def package_joint_stage_report(run_dir: str | Path, output: str | Path) -> Path:
    run = Path(run_dir).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in run.rglob("*") if path.is_file() and path.suffix.lower() in ALLOWED)
    with tarfile.open(destination, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(Path(run.name) / path.relative_to(run)), recursive=False)
    return destination

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = package_joint_stage_report(args.run_dir, args.output)
    print(result)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
