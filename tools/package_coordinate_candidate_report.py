#!/usr/bin/env python3
"""Package candidate diagnostics without meshes, checkpoints or arrays."""

from __future__ import annotations
import argparse, io, json, shlex, tarfile
from pathlib import Path

ALLOWED = {".json", ".csv", ".jsonl", ".md"}
EXCLUDED = {".ply", ".pth", ".pt", ".npy", ".npz"}


def package(inputs, output):
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True); files = []
    with tarfile.open(output, "w:gz") as archive:
        for root in inputs:
            if not root.is_dir(): raise FileNotFoundError(root)
            for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED):
                name = Path(root.name) / path.relative_to(root)
                archive.add(path, arcname=str(name), recursive=False); files.append(str(name))
        payload = json.dumps({"inputs": list(map(str, inputs)), "files": files, "allowed": sorted(ALLOWED), "excluded": sorted(EXCLUDED)}, indent=2).encode()
        info = tarfile.TarInfo("packaging_manifest.json"); info.size = len(payload); archive.addfile(info, io.BytesIO(payload))
    return {"archive": str(output), "file_count": len(files), "allowed": sorted(ALLOWED), "excluded": sorted(EXCLUDED)}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--input", action="append", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); inputs = [Path(v).expanduser().resolve() for v in args.input]; output = Path(args.output).expanduser().resolve()
    result = package(inputs, output)
    command = "python tools/package_coordinate_candidate_report.py " + " ".join(f"--input {shlex.quote(str(p))}" for p in inputs) + f" --output {shlex.quote(str(output))}"
    print(json.dumps({**result, "exact_package_command": command}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
