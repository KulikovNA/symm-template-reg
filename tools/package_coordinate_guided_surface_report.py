#!/usr/bin/env python3
"""Package coordinate-guided diagnostics as a JSON/CSV/JSONL/MD-only archive."""

from __future__ import annotations

import argparse
import io
import json
import shlex
import tarfile
from pathlib import Path

ALLOWED = {".json", ".csv", ".jsonl", ".md"}
FORBIDDEN = {".ply", ".pth", ".pt", ".npy", ".npz"}


def package(inputs: list[Path], output: Path) -> dict:
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    with tarfile.open(output, "w:gz") as archive:
        for root in inputs:
            if not root.is_dir(): raise FileNotFoundError(root)
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                if path.suffix.lower() not in ALLOWED:
                    continue
                arcname = Path(root.name) / path.relative_to(root)
                archive.add(path, arcname=str(arcname), recursive=False)
                files.append(str(arcname))
        manifest = {
            "inputs": list(map(str, inputs)), "files": files,
            "allowed_extensions": sorted(ALLOWED),
            "excluded_extensions": sorted(FORBIDDEN),
            "stop_recommendation": "STOP: inspect coordinate_projection_gate.json before any later stage.",
        }
        encoded = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("packaging_manifest.json"); info.size = len(encoded)
        archive.addfile(info, io.BytesIO(encoded))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inputs = [Path(value).expanduser().resolve() for value in args.input]
    output = Path(args.output).expanduser().resolve()
    result = package(inputs, output)
    exact = "python tools/package_coordinate_guided_surface_report.py " + " ".join(
        f"--input {shlex.quote(str(path))}" for path in inputs
    ) + f" --output {shlex.quote(str(output))}"
    print(json.dumps({**result, "archive": str(output), "exact_package_command": exact}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
