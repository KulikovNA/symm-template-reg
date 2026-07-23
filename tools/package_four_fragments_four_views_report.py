#!/usr/bin/env python3
"""Package compact 4x4 reports while excluding weights and geometry."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

ALLOWED = {".json", ".csv", ".jsonl", ".md"}
EXCLUDED = {".pth", ".ply", ".pt", ".npy", ".npz"}


def package(inputs: list[Path], output: Path) -> dict:
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    with tarfile.open(output, "w:gz") as archive:
        for root in inputs:
            if not root.exists(): raise FileNotFoundError(root)
            candidates = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
            base = root.parent if root.is_file() else root
            for path in candidates:
                suffix = path.suffix.lower()
                if suffix in EXCLUDED or suffix not in ALLOWED: continue
                arcname = Path(root.name) / path.relative_to(base)
                archive.add(path, arcname=str(arcname), recursive=False); files.append(str(arcname))
        manifest = {
            "debug_training_on_test_split": True,
            "train_and_validation_use_same_samples": True,
            "results_are_not_final_evaluation": True,
            "report_type": "four_fragments_four_views_clean_v3_scratch_compact",
            "inputs": list(map(str, inputs)), "files": files,
            "allowed_extensions": sorted(ALLOWED), "excluded_extensions": sorted(EXCLUDED),
            "contains_checkpoint": False, "contains_geometry": False,
            "stop_after_packaging": True,
        }
        encoded = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("packaging_manifest.json"); info.size = len(encoded)
        archive.addfile(info, io.BytesIO(encoded))
    return {"archive": str(output), "file_count": len(files), **manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = package([Path(value).expanduser().resolve() for value in args.input], Path(args.output).expanduser().resolve())
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
