#!/usr/bin/env python3
"""Build reproducible test-split debug manifests after physical mesh filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import load_config  # noqa: E402
from symm_template_reg.datasets import FragmentTemplateRegistrationDataset  # noqa: E402
from symm_template_reg.datasets.fragment_mesh_filter import (  # noqa: E402
    REQUIRED_TEST_SPLIT_FLAGS,
)
from symm_template_reg.engine.manifest import validate_debug_training_flags  # noqa: E402


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _filter_configs(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    data = config.get("data", {})
    dataset = config.get("dataset", {})
    fragment = data.get("fragment_mesh_filter", dataset.get("fragment_mesh_filter"))
    observed = data.get("observed_filter", dataset.get("observed_filter"))
    if not isinstance(fragment, Mapping) or not isinstance(observed, Mapping):
        raise ValueError("config must define data.fragment_mesh_filter and observed_filter")
    return dict(fragment), dict(observed)


def _sample_entry(dataset: FragmentTemplateRegistrationDataset, record: Any) -> dict[str, Any]:
    metadata = record.fragment_mesh_metadata
    return {
        "sample_id": record.sample_id,
        "scene_id": record.scene_id,
        "frame_id": record.frame_id,
        "fragment_id": record.fragment_id,
        "fragment_key": record.fragment_key,
        "num_observed_points": record.num_observed_points,
        "visible_points_path": str(record.visible_points_path),
        "fragment_mesh_path": str(metadata.mesh_path),
        "fragment_mesh_sha256": metadata.sha256,
        "fragment_num_vertices": metadata.num_vertices,
        "fragment_num_faces": metadata.num_faces,
        "fragment_surface_area_m2": metadata.surface_area_m2,
        "fragment_bbox_diagonal_m": metadata.bbox_diagonal_m,
    }


def _base_payload(
    dataset: FragmentTemplateRegistrationDataset,
    fragment_filter: Mapping[str, Any],
    observed_filter: Mapping[str, Any],
) -> dict[str, Any]:
    report = dataset.index_report
    return {
        **REQUIRED_TEST_SPLIT_FLAGS,
        "dataset_root": str(dataset.dataset_root),
        "fragment_filter": dict(fragment_filter),
        "observed_filter": dict(observed_filter),
        "accepted_physical_fragments": report["accepted_physical_fragments"],
        "rejected_physical_fragments": report["rejected_physical_fragments"],
        "accepted_observations": report[
            "accepted_frame_observations_before_max_samples"
        ],
        "rejected_observations": (
            report["rejected_because_physical_fragment"]
            + report["rejected_observed_points_too_few"]
        ),
        "fragment_mesh_cache_hits": report["fragment_mesh_cache_hits"],
        "fragment_mesh_cache_misses": report["fragment_mesh_cache_misses"],
    }


def _balanced_tiny_indices(dataset: FragmentTemplateRegistrationDataset) -> list[int]:
    candidates = [
        index
        for index, record in enumerate(dataset.sample_records)
        if record.scene_id in {"scene_000000", "scene_000001", "scene_000002"}
    ]
    groups: dict[str, list[int]] = defaultdict(list)
    for index in candidates:
        sample = dataset[index]
        group = sample["gt"].get("effective_symmetry_group")
        key = json.dumps(group, sort_keys=True) if group is not None else "none"
        groups[key].append(index)
    for indices in groups.values():
        indices.sort(
            key=lambda index: (
                dataset.sample_records[index].num_observed_points,
                dataset.sample_records[index].sample_id,
            )
        )
        if len(indices) > 2:
            # Interleave short/long observations rather than selecting one end.
            ordered: list[int] = []
            while indices:
                ordered.append(indices.pop(0))
                if indices:
                    ordered.append(indices.pop(-1))
            indices.extend(ordered)
    selected: list[int] = []
    group_keys = sorted(groups)
    cursor = 0
    while len(selected) < 16 and any(groups.values()):
        key = group_keys[cursor % len(group_keys)]
        cursor += 1
        if groups[key]:
            selected.append(groups[key].pop(0))
    return selected


def build_manifests(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    min_fragment_faces: int | None = None,
    min_observed_points: int | None = None,
    max_observed_points: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_debug_training_flags(config)
    fragment_filter, observed_filter = _filter_configs(config)
    if min_fragment_faces is not None:
        fragment_filter["enabled"] = True
        fragment_filter["min_num_faces"] = int(min_fragment_faces)
    if min_observed_points is not None:
        observed_filter["min_observed_points"] = int(min_observed_points)
    if max_observed_points is not None:
        observed_filter["max_observed_points"] = int(max_observed_points)
    if bool(fragment_filter.get("enabled")) and fragment_filter.get("min_num_faces") is None:
        raise ValueError(
            "Fragment face threshold is enabled but min_num_faces is not configured."
        )
    dataset_config = deepcopy(config["dataset"])
    dataset_config.pop("type", None)
    dataset_config["fragment_mesh_filter"] = fragment_filter
    dataset_config["observed_filter"] = observed_filter
    dataset = FragmentTemplateRegistrationDataset(**dataset_config)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset.write_filter_artifacts(output / "data_filter")
    base = _base_payload(dataset, fragment_filter, observed_filter)
    all_records = list(dataset.sample_records)
    tiny_indices = _balanced_tiny_indices(dataset)
    partial = {
        **base,
        "tiny_candidate_samples": len(tiny_indices),
        "scene_000000_candidate_samples": sum(
            record.scene_id == "scene_000000" for record in all_records
        ),
    }
    if len(tiny_indices) < 16:
        (output / "partial_manifest_audit.json").write_text(
            json.dumps(partial, indent=2) + "\n", encoding="utf-8"
        )
        raise ValueError(
            f"physical/observed filters leave only {len(tiny_indices)} tiny-overfit samples; need 16"
        )
    manifests: dict[str, dict[str, Any]] = {}
    tiny_samples = [_sample_entry(dataset, all_records[index]) for index in tiny_indices]
    manifests["tiny_overfit_16.json"] = {
        **base,
        "manifest_type": "tiny_overfit_16",
        "selection": "16 unique real observations balanced across effective groups when possible",
        "samples": tiny_samples,
    }
    scene_records = [
        record for record in all_records if record.scene_id == "scene_000000"
    ]
    rejected_scene_fragments = sorted(
        decision.metadata.fragment_id
        for key, decision in dataset.fragment_filter_decisions.items()
        if key[0] == "scene_000000" and not decision.accepted
    )
    manifests["scene_000000_overfit.json"] = {
        **base,
        "manifest_type": "scene_000000_overfit",
        "excluded_fragment_ids": rejected_scene_fragments,
        "samples": [_sample_entry(dataset, record) for record in scene_records],
    }
    split_samples = []
    for record in all_records:
        scene_number = int(record.scene_id.rsplit("_", 1)[1])
        split = "train" if scene_number < 8 else "validation" if scene_number == 8 else "test"
        entry = _sample_entry(dataset, record)
        entry["debug_split"] = split
        split_samples.append(entry)
    manifests["debug_scene_split_8_1_1.json"] = {
        **base,
        "manifest_type": "debug_scene_split_8_1_1",
        "scene_split": {
            "train": [f"scene_{index:06d}" for index in range(8)],
            "validation": ["scene_000008"],
            "test": ["scene_000009"],
        },
        "samples": split_samples,
    }
    rejected_decisions = [
        decision.to_dict()
        for decision in dataset.fragment_filter_decisions.values()
        if not decision.accepted
    ]
    manifests["rejected_small_fragments.json"] = {
        **base,
        "manifest_type": "rejected_small_fragments",
        "rejected_fragments": rejected_decisions,
        "samples": [],
    }
    result = {"output_dir": str(output), "manifests": {}}
    for name, payload in manifests.items():
        path = output / name
        digest = _write_manifest(path, payload)
        result["manifests"][name] = {
            "sha256": digest,
            "samples": len(payload.get("samples", [])),
        }
    (output / "manifest_build_summary.json").write_text(
        json.dumps({**REQUIRED_TEST_SPLIT_FLAGS, **result}, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-fragment-faces", type=int)
    parser.add_argument("--min-observed-points", type=int)
    parser.add_argument("--max-observed-points", type=int)
    args = parser.parse_args()
    result = build_manifests(
        args.config,
        args.output_dir,
        min_fragment_faces=args.min_fragment_faces,
        min_observed_points=args.min_observed_points,
        max_observed_points=args.max_observed_points,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
