#!/usr/bin/env python3
"""Package compact eight-view reports, excluding weights and geometry binaries."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

ALLOWED = {".json", ".csv", ".jsonl", ".md"}
EXCLUDED = {".pth", ".ply", ".pt", ".npy", ".npz"}


def package_eight_view_report(inputs: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    with tarfile.open(output, "w:gz") as archive:
        for root in inputs:
            if root.is_file() and root.suffix.lower() in ALLOWED:
                candidates, base = [root], root.parent
            elif root.is_dir():
                candidates = sorted(path for path in root.rglob("*") if path.is_file())
                base = root
            else:
                raise FileNotFoundError(root)
            for path in candidates:
                if path.suffix.lower() not in ALLOWED:
                    continue
                arcname = Path(root.name) / path.relative_to(base)
                archive.add(path, arcname=str(arcname), recursive=False)
                files.append(str(arcname))
        manifest = {
            "report_type": "eight_view_coordinate_compact",
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
    result = package_eight_view_report(
        [Path(value).expanduser().resolve() for value in args.input],
        Path(args.output).expanduser().resolve(),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
