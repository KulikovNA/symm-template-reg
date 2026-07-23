#!/usr/bin/env python3
"""Package compact ten-view scratch reports without weights or geometry."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

ALLOWED = {".json", ".csv", ".jsonl", ".md"}
EXCLUDED = {".pth", ".ply", ".pt", ".npy", ".npz"}
LEGACY_REPORT_NAMES = {
    "patch_confusion_matrix.json", "region_confusion_matrix.json",
    "region_class_distribution.json", "region_class_distribution.csv",
    "region_metrics.json", "ranking_diagnostics.json",
    "ranking_target_statistics.json", "query_assignment_matrix.csv",
    "query_assignment_diagnostics.json", "oracle_pose_metrics.json",
    "top1_vs_oracle_summary.json", "triangle_classifier_metrics.json",
    "barycentric_metrics.json",
}


def _manifest_report_files(inputs: list[Path]) -> list[Path]:
    """Discover the manifest and its validation/audit beside a run config."""

    discovered: set[Path] = set()
    for root in inputs:
        resolved = root / "resolved_config.json" if root.is_dir() else None
        if resolved is None or not resolved.is_file():
            continue
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        raw_manifest = payload.get("data", {}).get("train_manifest")
        if not raw_manifest:
            continue
        manifest = Path(str(raw_manifest)).expanduser().resolve()
        candidates = (
            manifest,
            manifest.with_name(manifest.stem + "_validation.json"),
            manifest.with_name("ten_view_manifest_audit.json"),
            manifest.with_name("ten_view_manifest_audit.md"),
        )
        discovered.update(path for path in candidates if path.is_file())
    return sorted(discovered)


def package_ten_view_scratch_report(inputs: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    with tarfile.open(output, "w:gz") as archive:
        for root in inputs:
            if root.is_file() and root.suffix.lower() in ALLOWED:
                candidates, base = [root], root.parent
            elif root.is_dir():
                candidates, base = sorted(p for p in root.rglob("*") if p.is_file()), root
            else:
                raise FileNotFoundError(root)
            for path in candidates:
                if path.suffix.lower() not in ALLOWED:
                    continue
                if path.name in LEGACY_REPORT_NAMES:
                    continue
                arcname = Path(root.name) / path.relative_to(base)
                archive.add(path, arcname=str(arcname), recursive=False)
                files.append(str(arcname))
        for path in _manifest_report_files(inputs):
            arcname = Path("manifest_audit") / path.name
            archive.add(path, arcname=str(arcname), recursive=False)
            files.append(str(arcname))
        manifest = {
            "report_type": "ten_view_clean_v3_scratch_compact",
            "inputs": list(map(str, inputs)), "files": files,
            "allowed_extensions": sorted(ALLOWED),
            "excluded_extensions": sorted(EXCLUDED),
            "excluded_legacy_report_names": sorted(LEGACY_REPORT_NAMES),
            "contains_checkpoint": False, "contains_geometry": False,
        }
        encoded = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("packaging_manifest.json"); info.size = len(encoded)
        archive.addfile(info, io.BytesIO(encoded))
    return {"archive": str(output), "file_count": len(files), **manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = package_ten_view_scratch_report(
        [Path(value).expanduser().resolve() for value in args.input],
        Path(args.output).expanduser().resolve(),
    )
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
