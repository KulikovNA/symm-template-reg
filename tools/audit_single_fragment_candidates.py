#!/usr/bin/env python3
"""Audit every physical fragment candidate in one scene without choosing one."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.datasets import FragmentTemplateRegistrationDataset  # noqa: E402


WARNING_FLAGS = {
    "debug_training_on_test_split": True,
    "train_and_validation_use_same_samples": True,
    "results_are_not_final_evaluation": True,
}


def _matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return matrix


def _atomic_new_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def audit_candidates(
    dataset_root: str | Path,
    scene_id: str,
    min_fragment_faces: int,
    min_observed_points: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    dataset = FragmentTemplateRegistrationDataset(
        dataset_root=dataset_root,
        fragment_mesh_filter={
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
        },
        observed_filter={
            "min_observed_points": int(min_observed_points),
            "max_observed_points": 4096,
            "point_policy": "farthest_point_up_to_max",
        },
        symmetry_region_activity={
            "min_points": 1,
            "min_fraction": 0.0,
            "boundary_tolerance_m": 1e-6,
        },
        fragment_mesh_cache_dir=destination / "cache",
        template_fine_points=2048,
        template_coarse_points=512,
    )
    root = Path(dataset.dataset_root)
    scene_dir = root / scene_id
    gt = json.loads((scene_dir / "gt_annotations.json").read_text(encoding="utf-8"))
    frames = {int(frame["frame_id"]): frame for frame in gt.get("frames", [])}
    records_by_fragment: dict[int, list[tuple[int, Any]]] = {}
    for index, record in enumerate(dataset.sample_records):
        if record.scene_id == scene_id:
            records_by_fragment.setdefault(record.fragment_id, []).append((index, record))

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    metadata_items = sorted(
        (
            metadata
            for (candidate_scene, _), metadata in dataset.fragment_metadata_by_id.items()
            if candidate_scene == scene_id
        ),
        key=lambda item: item.fragment_id,
    )
    for metadata in metadata_items:
        fragment_id = int(metadata.fragment_id)
        fragment_records = records_by_fragment.get(fragment_id, [])
        observations: list[dict[str, Any]] = []
        errors: list[str] = []
        angles: list[float] = []
        distances: list[float] = []
        counts: list[int] = []
        so2_flags: list[bool] = []
        for index, record in fragment_records:
            try:
                sample = dataset[index]
                frame = frames[int(record.frame_id)]
                T_C_from_W = _matrix(frame.get("T_C_from_W"), "T_C_from_W")
                T_W_from_C = np.linalg.inv(T_C_from_W)
                T_C_from_O = _matrix(record.gt_fragment.get("T_C_from_O"), "T_C_from_O")
                _matrix(record.gt_fragment.get("T_C_from_F"), "T_C_from_F")
                metadata_symmetry = sample["template"]["symmetry_metadata"]
                axis_O = np.asarray(metadata_symmetry.axis.direction, dtype=np.float64)
                axis_C = T_C_from_O[:3, :3] @ axis_O
                view = T_C_from_O[:3, 3]
                distance = float(np.linalg.norm(view))
                cosine = abs(float(np.dot(axis_C, view / max(distance, 1e-12))))
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                active = sample["gt"]["active_symmetry_regions"].tolist()
                active_names = [
                    region.region_id
                    for region, enabled in zip(metadata_symmetry.regions, active)
                    if enabled
                ]
                group = sample["gt"]["effective_symmetry_group"]
                counts.append(int(record.num_observed_points))
                angles.append(angle)
                distances.append(distance)
                so2_flags.append(str(group.get("type")) == "SO2")
                observations.append(
                    {
                        "sample_id": sample["sample_id"],
                        "frame_id": int(record.frame_id),
                        "num_observed_points": int(record.num_observed_points),
                        "active_symmetry_regions": active_names,
                        "effective_group": group,
                        "view_to_object_axis_angle_deg": angle,
                        "camera_to_object_distance_m": distance,
                        "T_C_from_O_available": True,
                        "T_C_from_F_available": True,
                        "T_W_from_C_available": bool(np.isfinite(T_W_from_C).all()),
                        "data_contract_errors": [],
                    }
                )
            except Exception as exc:  # audit must report all candidates
                error = f"frame_{record.frame_id:06d}: {type(exc).__name__}: {exc}"
                errors.append(error)
                observations.append(
                    {
                        "sample_id": record.sample_id,
                        "frame_id": int(record.frame_id),
                        "num_observed_points": int(record.num_observed_points),
                        "data_contract_errors": [error],
                    }
                )
        criteria = {
            "passes_min_num_faces": metadata.num_faces >= int(min_fragment_faces),
            "present_in_all_10_frames": len(fragment_records) == 10,
            "all_views_min_observed_points": bool(counts)
            and min(counts) >= int(min_observed_points),
            "wide_view_range": bool(angles) and max(angles) - min(angles) >= 30.0,
            "not_so2_only_in_all_views": bool(so2_flags) and not all(so2_flags),
            "no_annotation_or_transform_errors": not errors,
        }
        recommended = all(criteria.values())
        row = {
            "fragment_id": fragment_id,
            "num_faces": int(metadata.num_faces),
            "num_vertices": int(metadata.num_vertices),
            "surface_area_m2": float(metadata.surface_area_m2),
            "bbox_diagonal_m": float(metadata.bbox_diagonal_m),
            "available_frames": len(fragment_records),
            "usable_observations": len(observations) - len(errors),
            "observed_points_min": min(counts) if counts else 0,
            "observed_points_median": statistics.median(counts) if counts else 0,
            "observed_points_max": max(counts) if counts else 0,
            "view_axis_angle_min_deg": min(angles) if angles else math.nan,
            "view_axis_angle_max_deg": max(angles) if angles else math.nan,
            "view_axis_angle_range_deg": max(angles) - min(angles) if angles else math.nan,
            "camera_distance_min_m": min(distances) if distances else math.nan,
            "camera_distance_max_m": max(distances) if distances else math.nan,
            "all_gt_transforms_available": not errors and len(observations) == 10,
            "all_T_W_from_C_available": not errors and len(observations) == 10,
            "data_contract_error_count": len(errors),
            **criteria,
            "recommended": recommended,
        }
        rows.append(row)
        details.append({**row, "observations": observations, "errors": errors})

    recommended_ids = [row["fragment_id"] for row in rows if row["recommended"]]
    report = {
        **WARNING_FLAGS,
        "dataset_root": str(root),
        "scene_id": scene_id,
        "criteria": {
            "min_fragment_faces": int(min_fragment_faces),
            "min_observed_points": int(min_observed_points),
            "expected_frames": 10,
            "wide_view_range_min_deg": 30.0,
        },
        "selection_policy": "no hidden winner; all equally passing fragment IDs are listed",
        "recommended_fragment_ids": recommended_ids,
        "candidates": details,
    }
    csv_path = destination / "single_fragment_candidates.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_new_text(
        destination / "single_fragment_candidates.json",
        json.dumps(report, indent=2) + "\n",
    )
    markdown = [
        "# Single-fragment candidates",
        "",
        f"Scene: `{scene_id}`. No fragment is selected automatically.",
        "",
        "| fragment | faces | frames | points min/median/max | angle range | errors | recommended |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['fragment_id']} | {row['num_faces']} | {row['usable_observations']} | "
            f"{row['observed_points_min']}/{row['observed_points_median']}/{row['observed_points_max']} | "
            f"{row['view_axis_angle_range_deg']:.2f} deg | {row['data_contract_error_count']} | "
            f"{'yes' if row['recommended'] else 'no'} |"
        )
    markdown.extend(
        [
            "",
            "Recommendation: "
            + (
                ", ".join(f"fragment_id={value}" for value in recommended_ids)
                if recommended_ids
                else "no candidate satisfies every explicit criterion"
            ),
            "",
        ]
    )
    _atomic_new_text(destination / "single_fragment_candidates.md", "\n".join(markdown))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--min-fragment-faces", type=int, default=840)
    parser.add_argument("--min-observed-points", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = audit_candidates(**vars(args))
    print(json.dumps({
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "recommended_fragment_ids": report["recommended_fragment_ids"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
