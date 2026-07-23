#!/usr/bin/env python3
"""Package one V4 audit/stage as a compact JSON/CSV/JSONL/MD-only archive."""

from __future__ import annotations

import argparse
import io
import json
import shlex
import tarfile
from pathlib import Path

ALLOWED = {".json", ".csv", ".jsonl", ".md"}
FORBIDDEN = {".ply", ".pth", ".pt", ".npy", ".npz"}


def _status(root: Path) -> tuple[bool | None, list[Path]]:
    candidate_gates = sorted(root.rglob("candidate_stage_gate.json"))
    if candidate_gates:
        key_files = candidate_gates + sorted(root.rglob("top1_quality_gate.json"))
        key_files += sorted(root.rglob("stage_gate.json"))
        values = [
            bool(json.loads(path.read_text(encoding="utf-8"))["candidate_stage_passed"])
            for path in candidate_gates
        ]
        return all(values), key_files
    candidates = sorted(root.rglob("stage_gate.json")) + sorted(
        path for path in root.rglob("*summary.json") if path.is_file()
    )
    values = []
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("stage_passed", "audit_passed"):
            if key in payload:
                values.append(bool(payload[key])); break
    return (all(values) if values else None), candidates


def package(inputs: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records, statuses, key_files = [], [], []
    with tarfile.open(output, "w:gz") as archive:
        for root in inputs:
            if not root.is_dir():
                raise FileNotFoundError(root)
            status, status_files = _status(root); statuses.append(status)
            key_files.extend(map(str, status_files))
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                if path.suffix.lower() in FORBIDDEN or path.suffix.lower() not in ALLOWED:
                    continue
                arcname = Path(root.name) / path.relative_to(root)
                archive.add(path, arcname=str(arcname), recursive=False)
                records.append(str(arcname))
        passed = None if any(value is None for value in statuses) else all(statuses)
        manifest = {
            "stage_passed": passed,
            "stop_recommendation": (
                "STOP: package and analyze this stage; do not start the next stage automatically."
                if passed
                else "STOP: gate failed or is unavailable; diagnose before any later stage."
            ),
            "inputs": list(map(str, inputs)), "files": records,
            "excluded_extensions": sorted(FORBIDDEN), "key_files": key_files,
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
    command = "python tools/package_correspondence_head_stage.py " + " ".join(
        f"--input {shlex.quote(str(path))}" for path in inputs
    ) + f" --output {shlex.quote(str(output))}"
    print(json.dumps({**result, "archive": str(output), "archive_command": command}, indent=2))
    return 0 if result["stage_passed"] is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
