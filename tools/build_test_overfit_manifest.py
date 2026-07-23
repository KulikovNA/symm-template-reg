#!/usr/bin/env python3
"""Build the single content-addressed faces>=840 train/validation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import load_config  # noqa: E402
from symm_template_reg.datasets import FragmentTemplateRegistrationDataset  # noqa: E402
from symm_template_reg.datasets.fragment_mesh_filter import sha256_file  # noqa: E402


WARNING_FLAGS = {
    "debug_training_on_test_split": True,
    "train_and_validation_use_same_samples": True,
    "results_are_not_final_evaluation": True,
}


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    value = dict(payload)
    value.pop("manifest_sha256", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sample_entry(record: Any) -> dict[str, Any]:
    mesh = record.fragment_mesh_metadata
    return {
        "sample_id": record.sample_id,
        "scene_id": record.scene_id,
        "frame_id": record.frame_id,
        "fragment_id": record.fragment_id,
        "fragment_key": record.fragment_key,
        "num_observed_points": record.num_observed_points,
        "visible_points_path": str(record.visible_points_path),
        "fragment_mesh_path": str(mesh.mesh_path),
        "fragment_mesh_sha256": mesh.sha256,
        "fragment_num_vertices": mesh.num_vertices,
        "fragment_num_faces": mesh.num_faces,
        "fragment_surface_area_m2": mesh.surface_area_m2,
        "fragment_bbox_diagonal_m": mesh.bbox_diagonal_m,
    }


def build_manifest(config_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    for key, expected in WARNING_FLAGS.items():
        if config.get(key) is not expected:
            raise ValueError(f"faces840 debug config requires {key} = true")
    data = dict(config["data"])
    fragment_filter = dict(data["fragment_mesh_filter"])
    if not fragment_filter.get("enabled") or fragment_filter.get("min_num_faces") != 840:
        raise ValueError("faces840 manifest requires enabled min_num_faces=840")
    dataset_config = deepcopy(dict(config["dataset"]))
    dataset_config.pop("type", None)
    dataset_config["fragment_mesh_filter"] = fragment_filter
    dataset_config["observed_filter"] = deepcopy(data["observed_filter"])
    dataset = FragmentTemplateRegistrationDataset(**dataset_config)
    report = dataset.index_report
    counts = (
        report["total_physical_fragments"],
        report["accepted_physical_fragments"],
        report["rejected_physical_fragments"],
        report["total_frame_observations"],
        report["accepted_frame_observations_before_max_samples"],
        report["rejected_because_physical_fragment"]
        + report["rejected_observed_points_too_few"],
    )
    if counts != (40, 36, 4, 400, 360, 40):
        raise RuntimeError(
            "faces840 dataset counts changed: expected (40,36,4,400,360,40), "
            f"got {counts}"
        )
    records = list(dataset.sample_records)
    if len(records) != 360 or len({record.sample_id for record in records}) != 360:
        raise RuntimeError("faces840 manifest must contain 360 unique observations")
    first_template = dataset.template_repository.get(records[0].object_model_id)
    template_path = Path(str(first_template["mesh_path"])).resolve()
    sidecar_path = Path(str(first_template["symmetry_sidecar_path"])).resolve()
    accepted = []
    rejected = []
    for decision in dataset.fragment_filter_decisions.values():
        entry = {
            "scene_id": decision.metadata.scene_id,
            "fragment_id": decision.metadata.fragment_id,
            "fragment_key": decision.metadata.fragment_key,
            "num_faces": decision.metadata.num_faces,
            "mesh_sha256": decision.metadata.sha256,
        }
        (accepted if decision.accepted else rejected).append(entry)
    payload: dict[str, Any] = {
        **WARNING_FLAGS,
        "manifest_type": "test_faces840_all",
        "dataset_root": str(dataset.dataset_root),
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "symmetry_sidecar_path": str(sidecar_path),
        "symmetry_sidecar_sha256": sha256_file(sidecar_path),
        "fragment_filter": fragment_filter,
        "observed_filter": dict(data["observed_filter"]),
        "physical_fragments_total": 40,
        "accepted_physical_fragments": 36,
        "rejected_physical_fragments": 4,
        "observations_total": 400,
        "accepted_observations": 360,
        "rejected_observations": 40,
        "accepted_fragment_ids": accepted,
        "rejected_fragment_ids": rejected,
        "train_sample_ids": [record.sample_id for record in records],
        "validation_sample_ids": [record.sample_id for record in records],
        "samples": [_sample_entry(record) for record in records],
    }
    content_digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    payload["manifest_sha256"] = content_digest
    output = Path(output_root).expanduser().resolve() / "manifests"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"test_faces840_all_{content_digest[:12]}.json"
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to overwrite different manifest: {path}")
    else:
        path.write_bytes(encoded)
    file_digest = hashlib.sha256(encoded).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_text = f"{file_digest}  {path.name}\n"
    if sidecar.exists() and sidecar.read_text(encoding="ascii") != sidecar_text:
        raise FileExistsError(f"refusing to overwrite manifest SHA sidecar: {sidecar}")
    sidecar.write_text(sidecar_text, encoding="ascii")
    dataset.write_filter_artifacts(output / f"test_faces840_all_{content_digest[:12]}_filter")
    return {
        **WARNING_FLAGS,
        "manifest_path": str(path),
        "manifest_sha256": content_digest,
        "manifest_file_sha256": file_digest,
        "accepted_physical_fragments": 36,
        "rejected_physical_fragments": 4,
        "accepted_observations": 360,
        "rejected_observations": 40,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/debug/test_overfit_faces840_gpu.py"
    )
    parser.add_argument(
        "--output-root",
        default="/home/nikita/disser/fragment-template-registration-lab/work_dirs",
    )
    args = parser.parse_args()
    result = build_manifest(args.config, args.output_root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
