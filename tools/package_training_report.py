#!/usr/bin/env python3
"""Собрать компактный воспроизводимый отчёт без весов и тяжёлой геометрии."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


ALLOWED_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt"}
EXCLUDED_SUFFIXES = {".pth", ".pt", ".ply", ".npy", ".npz"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    entries: list[tuple[Path, Path]] = []
    for raw in args.input_dir:
        root = Path(raw).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        entries.extend(
            (root, path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix.lower() in ALLOWED_SUFFIXES
            and path.suffix.lower() not in EXCLUDED_SUFFIXES
            and "trace" not in path.name.lower()
        )
    manifest = {
        "format": "symm-template-reg-compact-report-v1",
        "excluded_by_default": sorted(EXCLUDED_SUFFIXES),
        "files": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        used: set[str] = set()
        for root, path in entries:
            prefix = root.name
            arcname = str(Path(prefix) / path.relative_to(root))
            if arcname in used:
                arcname = str(Path(hashlib.sha256(str(root).encode()).hexdigest()[:8]) / arcname)
            used.add(arcname)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["files"].append(
                {"path": arcname, "sha256": digest, "size_bytes": path.stat().st_size}
            )
            archive.add(path, arcname=arcname, recursive=False)
        payload = (
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode()
        info = tarfile.TarInfo("training_report_manifest.json")
        info.size = len(payload)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps({"archive": str(output), "file_count": len(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
