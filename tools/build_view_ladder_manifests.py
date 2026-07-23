#!/usr/bin/env python3
"""Build deterministic one-, two-, four- and ten-view manifest subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.datasets.fragment_mesh_filter import sha256_file  # noqa: E402
from symm_template_reg.engine.single_fragment import (  # noqa: E402
    manifest_content_sha256,
    validate_single_fragment_manifest_payload,
)
from symm_template_reg.engine.view_ladder import subset_view_manifest  # noqa: E402


LADDER = {
    "frame04_only.json": (4,),
    "frame08_only.json": (8,),
    "frame06_only.json": (6,),
    "frames04_08.json": (4, 8),
    "frames04_05_02_08.json": (4, 5, 2, 8),
    "all_10_views.json": tuple(range(10)),
}


def _write_manifest(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def build_view_ladder(source_manifest: str | Path, output_dir: str | Path) -> dict:
    source_path = Path(source_manifest).expanduser().resolve()
    source_sidecar = source_path.with_suffix(source_path.suffix + ".sha256")
    if not source_sidecar.is_file():
        raise FileNotFoundError(f"manifest SHA sidecar is missing: {source_sidecar}")
    expected = source_sidecar.read_text(encoding="ascii").split()[0]
    if sha256_file(source_path) != expected:
        raise ValueError("source manifest file SHA256 mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("manifest_sha256") != manifest_content_sha256(source):
        raise ValueError("source manifest internal SHA256 mismatch")
    validate_single_fragment_manifest_payload(source, expected_samples=10)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    for filename, frames in LADDER.items():
        payload = subset_view_manifest(source, frames)
        validate_single_fragment_manifest_payload(
            payload, expected_samples=len(frames), min_num_faces=840
        )
        path = destination / filename
        file_sha = _write_manifest(path, payload)
        entries.append(
            {
                "path": str(path),
                "frame_ids": list(frames),
                "samples": len(frames),
                "manifest_sha256": payload["manifest_sha256"],
                "manifest_file_sha256": file_sha,
                "scene_id": payload["scene_id"],
                "fragment_id": payload["fragment_id"],
                "fragment_mesh_sha256": payload["fragment_mesh_sha256"],
                "deterministic_points": True,
                "train_validation_same_samples": True,
                "augmentation_enabled": False,
            }
        )
    report = {
        "source_manifest": str(source_path),
        "output_dir": str(destination),
        "manifests": entries,
    }
    (destination / "view_ladder_manifest_index.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_view_ladder(args.manifest, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
