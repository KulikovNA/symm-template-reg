#!/usr/bin/env python3
"""Package compact performance evidence without weights, geometry, or traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


ALLOWED = {".json", ".csv", ".jsonl", ".md"}
FORBIDDEN_NAMES = {"torch_profiler_trace.json"}


def run(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    files = []
    for raw in args.input_dir:
        root = Path(raw).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            if (
                path.is_file() and path.suffix.lower() in ALLOWED
                and path.name not in FORBIDDEN_NAMES
                and "trace" not in path.name.lower()
            ):
                files.append((root, path))
    manifest = {
        "archive": str(output),
        "allowed_extensions": sorted(ALLOWED),
        "excluded": ["PTH", "PT", "PLY", "NPY", "NPZ", "full profiler trace"],
        "files": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        for root, path in files:
            source_label = root.name
            arcname = Path(source_label) / path.relative_to(root)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["files"].append({
                "path": str(arcname), "sha256": digest, "size_bytes": path.stat().st_size,
            })
            archive.add(path, arcname=str(arcname), recursive=False)
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo("performance_report_manifest.json")
        info.size = len(payload); info.mtime = 0
        import io
        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps({"archive": str(output), "file_count": len(files)}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    return run(parser.parse_args())


if __name__ == "__main__": raise SystemExit(main())
