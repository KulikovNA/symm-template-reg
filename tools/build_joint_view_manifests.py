#!/usr/bin/env python3
"""Build and strictly validate the nested 2/4/8-view joint manifests."""

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
    "fragment0002_views02.json": (4, 8),
    "fragment0002_views04.json": (4, 5, 2, 8),
    "fragment0002_views08.json": (4, 5, 2, 8, 0, 1, 6, 9),
}
WARNING_FLAGS = {
    "debug_training_on_test_split": True,
    "train_and_validation_use_same_samples": True,
    "results_are_not_final_evaluation": True,
}


def _write(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2) + "\n").encode()
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def build_joint_view_manifests(source: str | Path, output_dir: str | Path) -> dict:
    source_path = Path(source).expanduser().resolve()
    sidecar = source_path.with_suffix(source_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"source manifest SHA sidecar is missing: {sidecar}")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    if sha256_file(source_path) != expected:
        raise ValueError("source manifest file SHA256 mismatch")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("manifest_sha256") != manifest_content_sha256(payload):
        raise ValueError("source manifest internal SHA256 mismatch")
    validate_single_fragment_manifest_payload(payload, expected_samples=10)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    previous: set[int] = set()
    shared_mesh_hash = payload["fragment_mesh_sha256"]
    for filename, frame_ids in LADDER.items():
        current = set(frame_ids)
        if previous and not previous < current:
            raise AssertionError("view manifests must be strictly nested")
        previous = current
        subset = subset_view_manifest(payload, frame_ids)
        subset.update(WARNING_FLAGS)
        # Flags are content, therefore refresh the manifest's internal digest.
        subset["manifest_sha256"] = manifest_content_sha256(subset)
        validate_single_fragment_manifest_payload(
            subset, expected_samples=len(frame_ids), min_num_faces=840
        )
        if subset["scene_id"] != "scene_000000" or int(subset["fragment_id"]) != 2:
            raise ValueError("unexpected physical fragment in nested manifest")
        if subset["fragment_mesh_sha256"] != shared_mesh_hash:
            raise ValueError("fragment mesh hash changed across view subsets")
        groups = {str(sample["effective_symmetry_group"]) for sample in subset["samples"]}
        if not all("C2" in group or "order': 2" in group or '\"order\": 2' in group for group in groups):
            raise ValueError(f"all samples must use effective C2, got {groups}")
        if any(sample.get("data_contract_errors") for sample in subset["samples"]):
            raise ValueError("manifest contains data contract errors")
        path = destination / filename
        file_digest = _write(path, subset)
        records.append({
            "path": str(path), "frame_ids": list(frame_ids),
            "sample_count": len(frame_ids), "manifest_file_sha256": file_digest,
            "manifest_sha256": subset["manifest_sha256"],
            "fragment_mesh_sha256": shared_mesh_hash,
            "effective_group": "C2", "deterministic_point_selection": True,
            **WARNING_FLAGS,
        })
    report = {"source_manifest": str(source_path), "manifests": records, **WARNING_FLAGS}
    _write(destination / "joint_view_manifest_index.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_joint_view_manifests(args.manifest, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
