#!/usr/bin/env python3
"""Audit and freeze the scene-0 4-fragment x 4-view shell-only overfit set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symm_template_reg.datasets import FragmentTemplateRegistrationDataset  # noqa: E402
from symm_template_reg.datasets.fragment_mesh_filter import sha256_file  # noqa: E402
from symm_template_reg.datasets.template_repository import load_ply  # noqa: E402
from symm_template_reg.engine.multifragment_overfit import (  # noqa: E402
    WARNING_FLAGS, equal_sample_weights, manifest_content_sha256,
    validate_multifragment_manifest_payload,
)
from symm_template_reg.models.pose.pose_representation import (  # noqa: E402
    invert_transform, transform_points,
)


DEFAULT_MANIFEST = Path(
    "/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/"
    "multifragment_overfit/scene000000_fragments0000_0003_"
    "frames0002_0004_0005_0008_shell_only.json"
)


def _filter(min_faces: int) -> dict[str, Any]:
    return {
        "enabled": True, "min_num_faces": int(min_faces), "max_num_faces": None,
        "min_num_vertices": None, "min_surface_area_m2": None,
        "min_bbox_diagonal_m": None, "exclude_entire_fragment": True,
        "missing_mesh_policy": "error", "manifest_mismatch_policy": "error",
        "cache_metadata": True, "train_policy": "exclude",
        "debug_eval_policy": "exclude", "validation_policy": "exclude",
    }


def _observed_filter() -> dict[str, Any]:
    return {"min_observed_points": 128, "max_observed_points": 4096, "point_policy": "farthest_point_up_to_max"}


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {"p50": float(np.quantile(values, 0.50)), "p95": float(np.quantile(values, 0.95)), "max": float(values.max(initial=0.0))}


def _tensor_list(value: Any) -> Any:
    return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value


def _group_name(group: Any) -> str:
    if not isinstance(group, dict):
        return "unknown"
    return "SO2" if group.get("type") == "SO2" else f"C{int(group.get('order', 1))}"


def _safe_write(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to overwrite different file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _dataset_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    fragments = tuple(map(int, args.fragment_ids))
    frames = tuple(map(int, args.frame_ids))
    if fragments != (0, 1, 2, 3) or frames != (2, 4, 5, 8):
        raise ValueError("this controlled audit requires fragments 0 1 2 3 and frames 2 4 5 8")
    dataset = FragmentTemplateRegistrationDataset(
        dataset_root=args.dataset_root, fragment_mesh_filter=_filter(args.min_num_faces),
        observed_filter=_observed_filter(), registration_point_selection="shell_only",
        symmetry_region_activity={"min_points": 1, "min_fraction": 0.0, "boundary_tolerance_m": 1e-6},
        fragment_mesh_cache_dir=output / "cache", template_fine_points=2048,
        template_coarse_points=512, random_seed=0,
    )
    index = {(record.scene_id, int(record.frame_id), int(record.fragment_id)): i for i, record in enumerate(dataset.sample_records)}
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    first_template = None
    template_mesh = None
    for fragment in fragments:
        for frame in frames:
            key = (args.scene_id, frame, fragment)
            errors: list[str] = []
            warnings: list[str] = []
            if key not in index:
                rows.append({"scene_id": args.scene_id, "frame_id": frame, "fragment_id": fragment, "sample_id": f"{args.scene_id}/frame_{frame:06d}/fragment_{fragment:04d}", "errors": "sample_missing", "warnings": ""})
                continue
            sample = dataset[index[key]]
            record = dataset.sample_records[index[key]]
            metadata = record.fragment_mesh_metadata
            visible_path = Path(record.visible_points_path)
            source_paths.extend((visible_path, metadata.mesh_path, metadata.annotation_path))
            with np.load(visible_path, allow_pickle=False) as arrays:
                fragment_mask = arrays["fragment_id"] == fragment
                labels = arrays["surface_label"][fragment_mask]
                shell_count_raw = int(np.count_nonzero(labels == 0))
                fracture_count_raw = int(np.count_nonzero(labels == 1))
                other_count_raw = int(np.count_nonzero((labels != 0) & (labels != 1)))
            q_O = sample["gt"]["points_O_corresponding"].float()
            p_C = sample["observed"]["points_C"].float()
            T = sample["gt"]["T_C_from_O"].float()
            round_trip = torch.linalg.vector_norm(transform_points(invert_transform(T), transform_points(T, q_O)) - q_O, dim=-1).numpy()
            target_error = torch.linalg.vector_norm(transform_points(T, q_O) - p_C, dim=-1).numpy()
            # Dataset q_O comes from the generator's exact face+barycentric target.
            # The model template is the same content-addressed mesh; target-to-surface
            # residual is therefore checked against the stored analytic reconstruction.
            with np.load(visible_path, allow_pickle=False) as arrays:
                mask = (arrays["fragment_id"] == fragment) & (arrays["surface_label"] == 0)
                raw_q = np.asarray(arrays["points_O"][mask], dtype=np.float64)
                raw_p = np.asarray(arrays["points_C"][mask], dtype=np.float64)
                if template_mesh is None:
                    template_mesh = load_ply(sample["template"]["mesh_path"])
                face_ids = np.asarray(arrays["face_id"][mask], dtype=np.int64)
                barycentric = np.asarray(arrays["barycentric"][mask], dtype=np.float64)
                vertices = np.asarray(template_mesh["points"], dtype=np.float64)
                faces_array = np.asarray(template_mesh["faces"], dtype=np.int64)
                valid_faces = (face_ids >= 0) & (face_ids < len(faces_array))
                reconstructed = np.full_like(raw_q, np.nan)
                triangles = vertices[faces_array[face_ids[valid_faces]]]
                reconstructed[valid_faces] = (
                    triangles * barycentric[valid_faces, :, None]
                ).sum(axis=1)
                analytic_error = np.linalg.norm(reconstructed[valid_faces] - raw_q[valid_faces], axis=-1)
                if not np.all(valid_faces):
                    errors.append("invalid_generator_face_id")
                raw_transform_error = np.linalg.norm(
                    (raw_q @ T[:3, :3].cpu().numpy().T + T[:3, 3].cpu().numpy()) - raw_p,
                    axis=-1,
                )
            if metadata.num_faces < args.min_num_faces:
                errors.append("fragment_num_faces_below_min")
            if len(q_O) < 128:
                errors.append("shell_point_count_below_128")
            if fracture_count_raw and sample["meta"]["registration_point_selection"] != "shell_only":
                errors.append("fracture_points_passed_to_model")
            if sample["gt"].get("T_W_from_C") is None:
                errors.append("T_W_from_C_missing")
            if _summary(round_trip)["p95"] >= 1e-4:
                errors.append("round_trip_p95_ge_1e-4m")
            if _summary(target_error)["p95"] >= 1e-4 or _summary(raw_transform_error)["p95"] >= 1e-4:
                errors.append("canonical_target_transform_inconsistent")
            if first_template is None:
                first_template = sample["template"]
            elif sample["template"]["mesh_path"] != first_template["mesh_path"]:
                errors.append("template_changed")
            group = sample["gt"]["effective_symmetry_group"]
            active = _tensor_list(sample["gt"]["active_symmetry_regions"])
            base = {
                "sample_id": sample["sample_id"], "scene_id": sample["scene_id"],
                "frame_id": frame, "fragment_id": fragment,
                "fragment_key": record.fragment_key, "visible_points_path": str(visible_path),
                "visible_points_sha256": sha256_file(visible_path),
                "fragment_mesh_path": str(metadata.mesh_path), "fragment_mesh_sha256": metadata.sha256,
                "fragment_num_vertices": metadata.num_vertices, "fragment_num_faces": metadata.num_faces,
                "fragment_surface_area_m2": metadata.surface_area_m2,
                "fragment_bbox_diagonal_m": metadata.bbox_diagonal_m,
                "shell_point_count": len(q_O), "shell_point_count_raw": shell_count_raw,
                "fracture_point_count": 0, "fracture_point_count_raw": fracture_count_raw,
                "other_point_count": 0, "other_point_count_raw": other_count_raw,
                "points_passed_to_model": len(q_O), "registration_point_selection": "shell_only",
                "T_C_from_O": _tensor_list(T), "T_W_from_C": _tensor_list(sample["gt"].get("T_W_from_C")),
                "T_W_from_C_available": sample["gt"].get("T_W_from_C") is not None,
                "active_symmetry_regions": active, "effective_symmetry_group": group,
                "effective_symmetry_group_name": _group_name(group),
                "round_trip_error_p50_m": _summary(round_trip)["p50"],
                "round_trip_error_p95_m": _summary(round_trip)["p95"],
                "round_trip_error_max_m": _summary(round_trip)["max"],
                "gt_qO_surface_p50_m": _summary(analytic_error)["p50"],
                "gt_qO_surface_p95_m": _summary(analytic_error)["p95"],
                "gt_qO_surface_max_m": _summary(analytic_error)["max"],
                "T_gt_qO_to_pC_p50_m": _summary(target_error)["p50"],
                "T_gt_qO_to_pC_p95_m": _summary(target_error)["p95"],
                "T_gt_qO_to_pC_max_m": _summary(target_error)["max"],
                "data_contract_errors": errors,
            }
            manifest_rows.append(base)
            rows.append({**base, "errors": ";".join(errors), "warnings": ";".join(warnings)})
    all_errors = [error for row in manifest_rows for error in row["data_contract_errors"]]
    group_distribution = Counter(row["effective_symmetry_group_name"] for row in manifest_rows)
    selection_passed = len(manifest_rows) == 16 and not all_errors
    alternatives = []
    for frame in range(10):
        if frame in frames:
            continue
        available = all((args.scene_id, frame, fragment) in index for fragment in fragments)
        minimum_shell = min((dataset[index[(args.scene_id, frame, fragment)]]["observed"]["points_C"].shape[0] for fragment in fragments), default=0)
        alternatives.append({"frame_id": frame, "all_fragments_available": available, "minimum_shell_points": int(minimum_shell), "valid_common_frame": bool(available and minimum_shell >= 128)})
    report = {
        **WARNING_FLAGS, "status": "ok" if selection_passed else "selection_failed",
        "selection_passed": selection_passed, "scene_id": args.scene_id,
        "fragment_ids": list(fragments), "frame_ids": list(frames), "sample_count": len(manifest_rows),
        "symmetry_group_distribution": dict(sorted(group_distribution.items())),
        "errors": all_errors, "valid_alternative_common_frames": alternatives,
    }
    fields = sorted({key for row in rows for key in row if key not in {"T_C_from_O", "T_W_from_C", "active_symmetry_regions", "effective_symmetry_group", "data_contract_errors"}})
    with (output / "samples.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    (output / "audit.json").write_text(json.dumps({**report, "samples": manifest_rows}, indent=2) + "\n", encoding="utf-8")
    lines = ["# Four fragments × four views data audit", "", f"- status: `{report['status']}`", f"- samples: `{len(manifest_rows)}/16`", f"- symmetry groups: `{dict(group_distribution)}`", "", "| fragment | frame | shell | faces | group | errors |", "|---:|---:|---:|---:|---|---|"]
    lines.extend(f"| {row['fragment_id']} | {row['frame_id']} | {row['shell_point_count']} | {row['fragment_num_faces']} | {row['effective_symmetry_group_name']} | {row['errors'] or '-'} |" for row in rows)
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not selection_passed:
        return report
    assert first_template is not None
    template_path = Path(first_template["mesh_path"]).resolve()
    sidecar_path = Path(first_template["symmetry_sidecar_path"]).resolve()
    source_paths.extend((template_path, sidecar_path))
    sample_ids = [row["sample_id"] for row in manifest_rows]
    payload: dict[str, Any] = {
        **WARNING_FLAGS, "manifest_type": "four_fragments_four_views_overfit",
        "experiment_type": "four_fragments_four_views_overfit", "initialization_mode": "scratch",
        "pretrained_checkpoint": None, "checkpoint_sources": [],
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "dataset_contract_sha256": _dataset_digest(source_paths), "scene_id": args.scene_id,
        "fragment_ids": list(fragments), "frame_ids": list(frames),
        "physical_fragment_count": 4, "accepted_observations": 16,
        "template_path": str(template_path), "template_sha256": sha256_file(template_path),
        "symmetry_sidecar_path": str(sidecar_path), "symmetry_sidecar_sha256": sha256_file(sidecar_path),
        "fragment_meshes": {str(fragment): {"path": next(row["fragment_mesh_path"] for row in manifest_rows if row["fragment_id"] == fragment), "sha256": next(row["fragment_mesh_sha256"] for row in manifest_rows if row["fragment_id"] == fragment)} for fragment in fragments},
        "fragment_filter": _filter(args.min_num_faces), "observed_filter": _observed_filter(),
        "min_num_faces": int(args.min_num_faces), "registration_point_selection": "shell_only",
        "point_selection_policy": "deterministic_shell_only_all_points_up_to_4096",
        "train_sample_ids": sample_ids, "validation_sample_ids": sample_ids,
        "symmetry_group_distribution": dict(sorted(group_distribution.items())),
        "loss_reduction": "per_sample_mean_then_batch_mean",
        "equal_weighting": equal_sample_weights(manifest_rows), "samples": manifest_rows,
    }
    payload["manifest_sha256"] = manifest_content_sha256(payload)
    validate_multifragment_manifest_payload(payload, min_num_faces=args.min_num_faces)
    manifest_path = Path(args.manifest_output).expanduser().resolve()
    encoded = (json.dumps(payload, indent=2) + "\n").encode()
    _safe_write(manifest_path, encoded)
    file_sha = hashlib.sha256(encoded).hexdigest()
    _safe_write(manifest_path.with_suffix(manifest_path.suffix + ".sha256"), f"{file_sha}  {manifest_path.name}\n".encode("ascii"))
    report.update({"manifest_path": str(manifest_path), "manifest_sha256": payload["manifest_sha256"], "manifest_file_sha256": file_sha})
    (output / "audit.json").write_text(json.dumps({**report, "samples": manifest_rows}, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene-id", default="scene_000000")
    parser.add_argument("--fragment-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--frame-ids", nargs="+", type=int, default=[2, 4, 5, 8])
    parser.add_argument("--min-num-faces", type=int, default=840)
    parser.add_argument("--manifest-output", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args); print(json.dumps(result, indent=2))
    return 0 if result["selection_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
