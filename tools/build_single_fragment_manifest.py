#!/usr/bin/env python3
"""Build a content-addressed manifest for one fragment in every scene view."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.datasets import FragmentTemplateRegistrationDataset  # noqa: E402
from symm_template_reg.datasets.fragment_mesh_filter import sha256_file  # noqa: E402
from symm_template_reg.engine.overfit_manifest import WARNING_FLAGS  # noqa: E402
from symm_template_reg.engine.single_fragment import (  # noqa: E402
    manifest_content_sha256,
    validate_single_fragment_manifest_payload,
)


def _tensor_list(value: Any) -> Any:
    return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value


def build_manifest(
    dataset_root: str | Path,
    scene_id: str,
    fragment_id: int,
    min_fragment_faces: int,
    min_observed_points: int,
    max_observed_points: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fragment_filter = {
        "enabled": True,
        "min_num_faces": int(min_fragment_faces),
        "max_num_faces": None,
        "min_num_vertices": None,
        "min_surface_area_m2": None,
        "min_bbox_diagonal_m": None,
        "exclude_entire_fragment": True,
        "missing_mesh_policy": "error",
        "manifest_mismatch_policy": "error",
        "cache_metadata": True,
        "train_policy": "exclude",
        "debug_eval_policy": "exclude",
        "validation_policy": "exclude",
    }
    observed_filter = {
        "min_observed_points": int(min_observed_points),
        "max_observed_points": int(max_observed_points),
        "point_policy": "farthest_point_up_to_max",
    }
    dataset = FragmentTemplateRegistrationDataset(
        dataset_root=dataset_root,
        fragment_mesh_filter=fragment_filter,
        observed_filter=observed_filter,
        symmetry_region_activity={
            "min_points": 1,
            "min_fraction": 0.0,
            "boundary_tolerance_m": 1e-6,
        },
        fragment_mesh_cache_dir=destination / "cache",
        template_fine_points=2048,
        template_coarse_points=512,
    )
    selected = [
        (index, record)
        for index, record in enumerate(dataset.sample_records)
        if record.scene_id == scene_id and record.fragment_id == int(fragment_id)
    ]
    if len(selected) != 10:
        raise ValueError(
            f"expected 10 usable observations for {scene_id}/fragment_{fragment_id:04d}, "
            f"got {len(selected)}"
        )
    entries: list[dict[str, Any]] = []
    for index, record in selected:
        sample = dataset[index]
        metadata = record.fragment_mesh_metadata
        T_W_from_C = sample["gt"].get("T_W_from_C")
        entries.append(
            {
                "sample_id": record.sample_id,
                "scene_id": record.scene_id,
                "frame_id": int(record.frame_id),
                "fragment_id": int(record.fragment_id),
                "fragment_key": record.fragment_key,
                "num_observed_points": int(record.num_observed_points),
                "visible_points_path": str(record.visible_points_path),
                "fragment_mesh_path": str(metadata.mesh_path),
                "fragment_mesh_sha256": metadata.sha256,
                "fragment_num_vertices": int(metadata.num_vertices),
                "fragment_num_faces": int(metadata.num_faces),
                "fragment_surface_area_m2": float(metadata.surface_area_m2),
                "fragment_bbox_diagonal_m": float(metadata.bbox_diagonal_m),
                "T_C_from_O": _tensor_list(sample["gt"]["T_C_from_O"]),
                "T_W_from_C": _tensor_list(T_W_from_C),
                "T_W_from_C_available": isinstance(T_W_from_C, torch.Tensor),
                "active_symmetry_regions": _tensor_list(
                    sample["gt"]["active_symmetry_regions"]
                ),
                "effective_symmetry_group": sample["gt"]["effective_symmetry_group"],
                "data_contract_errors": [],
            }
        )
    record = selected[0][1]
    template = dataset.template_repository.get(record.object_model_id)
    template_path = Path(str(template["mesh_path"])).resolve()
    sidecar_path = Path(str(template["symmetry_sidecar_path"])).resolve()
    sample_ids = [entry["sample_id"] for entry in entries]
    payload: dict[str, Any] = {
        **WARNING_FLAGS,
        "manifest_type": "single_fragment_overfit",
        "dataset_root": str(dataset.dataset_root),
        "scene_id": scene_id,
        "fragment_id": int(fragment_id),
        "physical_fragment_count": 1,
        "accepted_observations": len(entries),
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "symmetry_sidecar_path": str(sidecar_path),
        "symmetry_sidecar_sha256": sha256_file(sidecar_path),
        "fragment_mesh_path": entries[0]["fragment_mesh_path"],
        "fragment_mesh_sha256": entries[0]["fragment_mesh_sha256"],
        "fragment_filter": fragment_filter,
        "observed_filter": observed_filter,
        "min_num_faces": int(min_fragment_faces),
        "observed_point_policy": observed_filter["point_policy"],
        "train_sample_ids": sample_ids,
        "validation_sample_ids": sample_ids,
        "samples": entries,
    }
    payload["manifest_sha256"] = manifest_content_sha256(payload)
    validate_single_fragment_manifest_payload(
        payload,
        expected_samples=10,
        min_num_faces=int(min_fragment_faces),
    )
    path = destination / (
        f"single_fragment_scene{int(scene_id.rsplit('_', 1)[-1]):06d}_"
        f"fragment{int(fragment_id):04d}_{payload['manifest_sha256'][:12]}.json"
    )
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to overwrite different manifest: {path}")
    else:
        path.write_bytes(encoded)
    file_digest = hashlib.sha256(encoded).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_text = f"{file_digest}  {path.name}\n"
    if sidecar.exists():
        if sidecar.read_text(encoding="ascii") != sidecar_text:
            raise FileExistsError(f"refusing to overwrite SHA sidecar: {sidecar}")
    else:
        sidecar.write_text(sidecar_text, encoding="ascii")
    return {
        **WARNING_FLAGS,
        "manifest_path": str(path),
        "manifest_sha256": payload["manifest_sha256"],
        "manifest_file_sha256": file_digest,
        "scene_id": scene_id,
        "fragment_id": int(fragment_id),
        "samples": len(entries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--fragment-id", type=int, required=True)
    parser.add_argument("--min-fragment-faces", type=int, default=840)
    parser.add_argument("--min-observed-points", type=int, default=128)
    parser.add_argument("--max-observed-points", type=int, default=4096)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_manifest(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
