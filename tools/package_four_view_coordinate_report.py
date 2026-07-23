#!/usr/bin/env python3
"""Package compact four-view reports while excluding all binary geometry/weights."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

ALLOWED = {".json", ".csv", ".jsonl", ".md"}
EXCLUDED = {".pth", ".ply", ".pt", ".npy", ".npz"}


def package_four_view_report(inputs: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    with tarfile.open(output, "w:gz") as archive:
        for root in inputs:
            if root.is_file() and root.suffix.lower() in ALLOWED:
                candidates = [root]
                base = root.parent
            elif root.is_dir():
                candidates = sorted(path for path in root.rglob("*") if path.is_file())
                base = root
            else:
                raise FileNotFoundError(root)
            for path in candidates:
                suffix = path.suffix.lower()
                if suffix not in ALLOWED:
                    continue
                arcname = Path(root.name) / path.relative_to(base)
                archive.add(path, arcname=str(arcname), recursive=False)
                files.append(str(arcname))
        manifest = {
            "report_type": "four_view_coordinate_compact",
            "inputs": list(map(str, inputs)),
            "files": files,
            "allowed_extensions": sorted(ALLOWED),
            "excluded_extensions": sorted(EXCLUDED),
            "contains_checkpoint": False,
            "contains_geometry": False,
        }
        encoded = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo("packaging_manifest.json")
        info.size = len(encoded)
        archive.addfile(info, io.BytesIO(encoded))
    return {"archive": str(output), "file_count": len(files), **manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inputs = [Path(value).expanduser().resolve() for value in args.input]
    result = package_four_view_report(
        inputs, Path(args.output).expanduser().resolve()
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

