#!/usr/bin/env python3
"""Validate the on-disk dataset and write a factual, reproducible inventory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.datasets.fragment_template_dataset import resolve_split_root
from symm_template_reg.datasets.template_repository import inspect_ply_header, load_ply

try:
    from PIL import Image
except ImportError:  # the core loader does not require Pillow
    Image = None


EXPECTED_SCENE_PATHS = (
    "camera_info.json",
    "gt_annotations.json",
    "scene_meta.json",
    "fragments/fragment_annotations.json",
    "images",
    "depth",
    "instance_masks",
    "surface_masks",
    "visible_points",
    "scene_gt",
    "scene_gt/support_plane_gt.json",
)
ALIGNED_VISIBLE_FIELDS = (
    "u",
    "v",
    "fragment_id",
    "surface_label",
    "points_C",
    "points_F",
    "points_O",
    "face_id",
    "barycentric",
)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def _array_range(array: np.ndarray) -> tuple[float | int | None, float | int | None]:
    if not array.size or array.dtype.kind not in "biuf":
        return None, None
    return np.asarray(array).min().item(), np.asarray(array).max().item()


def _record_array_schema(fields: dict[str, dict[str, Any]], key: str, value: np.ndarray) -> None:
    low, high = _array_range(value)
    entry = fields.setdefault(
        key,
        {
            "dtype_values": set(),
            "ndim_values": set(),
            "trailing_shapes": set(),
            "first_dim_min": None,
            "first_dim_max": None,
            "value_min": None,
            "value_max": None,
            "present_in_files": 0,
        },
    )
    entry["dtype_values"].add(str(value.dtype))
    entry["ndim_values"].add(int(value.ndim))
    entry["trailing_shapes"].add(tuple(int(item) for item in value.shape[1:]))
    first = int(value.shape[0]) if value.ndim else 1
    entry["first_dim_min"] = first if entry["first_dim_min"] is None else min(entry["first_dim_min"], first)
    entry["first_dim_max"] = first if entry["first_dim_max"] is None else max(entry["first_dim_max"], first)
    if low is not None:
        entry["value_min"] = low if entry["value_min"] is None else min(entry["value_min"], low)
        entry["value_max"] = high if entry["value_max"] is None else max(entry["value_max"], high)
    entry["present_in_files"] += 1


def _serialize_schema(fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        key: {
            **value,
            "dtype_values": sorted(value["dtype_values"]),
            "ndim_values": sorted(value["ndim_values"]),
            "trailing_shapes": [list(shape) for shape in sorted(value["trailing_shapes"])],
        }
        for key, value in sorted(fields.items())
    }


def _duplicates(values: list[int]) -> list[int]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def inspect_dataset(dataset_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    root = resolve_split_root(dataset_root)
    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = sorted(path for path in root.glob("scene_*") if path.is_dir())
    templates = sorted((root / "models").glob("*.ply"))
    warnings: list[dict[str, Any]] = []

    def issue(code: str, severity: str, message: str, **context: Any) -> None:
        warnings.append({"code": code, "severity": severity, "message": message, **context})

    def read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
        if not path.is_file():
            if required:
                issue("missing_required_file", "error", f"Required JSON file is absent: {path}", path=str(path))
            return None
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            issue("invalid_json", "error", f"Could not read {path}: {error}", path=str(path))
            return None
        if not isinstance(value, dict):
            issue("invalid_json_root", "error", f"Expected a JSON object in {path}", path=str(path))
            return None
        return value

    directory_counts: Counter[str] = Counter()
    json_top_level_keys: dict[str, set[str]] = defaultdict(set)
    visible_schema_fields: dict[str, dict[str, Any]] = {}
    fragment_sample_fields: dict[str, dict[str, Any]] = {}
    npz_key_orders: Counter[tuple[str, ...]] = Counter()
    raster_trackers: dict[str, dict[str, Any]] = {}
    scene_summaries: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    frame_point_counts: list[int] = []
    referenced_visible_npz: set[Path] = set()
    inspected_visible_npz: set[Path] = set()
    transform_errors_O: list[float] = []
    transform_errors_F: list[float] = []
    depth_errors_m: list[np.ndarray] = []
    rotation_determinants: list[float] = []
    last_row_errors: list[float] = []
    visible_pixel_differences: list[int] = []
    fragment_vertices: list[int] = []
    fragment_faces: list[int] = []
    row_invariant_checks = Counter()
    row_invariant_failures = Counter()
    scene_gt_counts = Counter()
    camera_examples: list[dict[str, Any]] = []

    def inspect_raster(path: Path, kind: str) -> np.ndarray | None:
        if not path.is_file():
            issue("missing_referenced_raster", "error", f"Referenced {kind} file is absent: {path}", path=str(path))
            return None
        if Image is None:
            return None
        try:
            with Image.open(path) as image:
                mode = image.mode
                size = tuple(int(value) for value in image.size)
                array = np.asarray(image).copy()
        except (OSError, ValueError) as error:
            issue("invalid_raster", "error", f"Could not read {path}: {error}", path=str(path))
            return None
        tracker = raster_trackers.setdefault(
            kind,
            {
                "num_files_inspected": 0,
                "modes": set(),
                "sizes_wh": set(),
                "shapes": set(),
                "dtypes": set(),
                "value_min": None,
                "value_max": None,
                "unique_values": set(),
            },
        )
        tracker["num_files_inspected"] += 1
        tracker["modes"].add(mode)
        tracker["sizes_wh"].add(size)
        tracker["shapes"].add(tuple(int(value) for value in array.shape))
        tracker["dtypes"].add(str(array.dtype))
        low, high = _array_range(array)
        if low is not None:
            tracker["value_min"] = low if tracker["value_min"] is None else min(tracker["value_min"], low)
            tracker["value_max"] = high if tracker["value_max"] is None else max(tracker["value_max"], high)
        if kind in {"instance_mask", "surface_mask"}:
            tracker["unique_values"].update(int(value) for value in np.unique(array))
        return array

    if Image is None:
        issue(
            "pillow_unavailable",
            "warning",
            "Pillow is unavailable; raster shapes/dtypes/encodings and mask joins were not inspected.",
        )

    for scene_dir in scenes:
        missing = [relative for relative in EXPECTED_SCENE_PATHS if not (scene_dir / relative).exists()]
        for relative in missing:
            issue(
                "missing_expected_path",
                "error",
                f"Expected scene path is absent: {scene_dir.name}/{relative}",
                scene_id=scene_dir.name,
                path=relative,
            )
        for subdir in ("images", "depth", "instance_masks", "surface_masks", "visible_points"):
            directory_counts[subdir] += len([path for path in (scene_dir / subdir).glob("*") if path.is_file()])

        camera = read_json(scene_dir / "camera_info.json")
        gt = read_json(scene_dir / "gt_annotations.json")
        fragment_meta = read_json(scene_dir / "fragments" / "fragment_annotations.json")
        scene_meta = read_json(scene_dir / "scene_meta.json", required=False) or {}
        support_gt = read_json(scene_dir / "scene_gt" / "support_plane_gt.json")
        for relative, value in (
            ("camera_info.json", camera),
            ("gt_annotations.json", gt),
            ("fragments/fragment_annotations.json", fragment_meta),
            ("scene_meta.json", scene_meta),
        ):
            if value:
                json_top_level_keys[relative].update(value.keys())
        if gt is None or fragment_meta is None:
            scene_summaries.append({"scene_id": scene_dir.name, "status": "uninspectable", "missing_paths": missing})
            continue
        if gt.get("scene_id") != scene_dir.name:
            issue("scene_id_mismatch", "error", "Directory and gt_annotations scene IDs disagree.", scene_id=scene_dir.name, annotation_scene_id=gt.get("scene_id"))
        if scene_meta and scene_meta.get("scene_id") not in (None, scene_dir.name):
            issue("scene_meta_id_mismatch", "error", "Directory and scene_meta scene IDs disagree.", scene_id=scene_dir.name)
        if camera:
            camera_examples.append({"scene_id": scene_dir.name, **camera})
        width = int((camera or {}).get("width", 0))
        height = int((camera or {}).get("height", 0))
        depth_scale_m = float((camera or {}).get("depth_scale_m", 0.001))

        fragment_entries = fragment_meta.get("fragments", [])
        fragment_entry_ids = [int(entry["fragment_id"]) for entry in fragment_entries if "fragment_id" in entry]
        duplicate_fragment_meta = _duplicates(fragment_entry_ids)
        if duplicate_fragment_meta:
            issue("duplicate_fragment_metadata_id", "error", "Duplicate fragment IDs in fragment_annotations.", scene_id=scene_dir.name, fragment_ids=duplicate_fragment_meta)
        known_fragments = set(fragment_entry_ids)
        fragment_face_counts = {
            int(entry["fragment_id"]): int(entry["num_faces"])
            for entry in fragment_entries
            if "fragment_id" in entry and entry.get("num_faces") is not None
        }
        fragment_meshes = sorted((scene_dir / "fragments").glob("fragment_*.ply"))
        fragment_samples = sorted((scene_dir / "fragments" / "samples").glob("*.npz"))
        directory_counts["fragment_meshes"] += len(fragment_meshes)
        directory_counts["fragment_sample_npz"] += len(fragment_samples)
        directory_counts["fragment_face_label_npy"] += len(list((scene_dir / "fragments" / "labels").glob("*.npy")))
        for path in fragment_meshes:
            try:
                header = inspect_ply_header(path)
            except (OSError, ValueError) as error:
                issue("invalid_fragment_ply", "error", f"Could not inspect {path}: {error}", path=str(path))
                continue
            elements = {entry["name"]: int(entry["count"]) for entry in header["elements"]}
            fragment_vertices.append(elements.get("vertex", 0))
            fragment_faces.append(elements.get("face", 0))
        for path in fragment_samples:
            try:
                with np.load(path, allow_pickle=False) as arrays:
                    for key in arrays.files:
                        _record_array_schema(fragment_sample_fields, key, arrays[key])
            except (OSError, ValueError) as error:
                issue("invalid_fragment_sample_npz", "error", f"Could not inspect {path}: {error}", path=str(path))

        frame_entries = gt.get("frames", [])
        frame_ids = [int(frame["frame_id"]) for frame in frame_entries if "frame_id" in frame]
        duplicate_frames = _duplicates(frame_ids)
        if duplicate_frames:
            issue("duplicate_frame_id", "error", "Duplicate frame IDs in gt_annotations.", scene_id=scene_dir.name, frame_ids=duplicate_frames)

        support_frames: dict[int, dict[str, Any]] = {}
        if support_gt:
            if support_gt.get("scene_id") not in (None, scene_dir.name):
                issue("scene_gt_id_mismatch", "error", "support_plane_gt scene ID disagrees with its directory.", scene_id=scene_dir.name)
            raw_support_frames = support_gt.get("frames", [])
            support_ids = [int(entry["frame_id"]) for entry in raw_support_frames if "frame_id" in entry]
            duplicate_support = _duplicates(support_ids)
            if duplicate_support:
                issue("duplicate_scene_gt_frame_id", "error", "Duplicate frame IDs in support_plane_gt.", scene_id=scene_dir.name, frame_ids=duplicate_support)
            support_frames = {int(entry["frame_id"]): entry for entry in raw_support_frames if "frame_id" in entry}
            scene_gt_counts["support_plane_files"] += 1
            scene_gt_counts["planes_W"] += len(support_gt.get("planes_W", []))
            world_mesh_rel = support_gt.get("static_scene_mesh_W")
            if world_mesh_rel:
                world_mesh = scene_dir / str(world_mesh_rel)
                if world_mesh.is_file():
                    scene_gt_counts["static_scene_mesh_W"] += 1
                    try:
                        inspect_ply_header(world_mesh)
                    except (OSError, ValueError) as error:
                        issue("invalid_scene_gt_mesh", "error", f"Could not inspect {world_mesh}: {error}", path=str(world_mesh))
                else:
                    issue("missing_scene_gt_mesh", "error", f"Missing world scene mesh: {world_mesh}", path=str(world_mesh))

        scene_sample_count = 0
        for frame in frame_entries:
            if "frame_id" not in frame:
                issue("frame_id_absent", "error", "GT frame entry has no frame_id.", scene_id=scene_dir.name)
                continue
            frame_id = int(frame["frame_id"])
            image_path = scene_dir / str(frame.get("image", f"images/frame_{frame_id:06d}.png"))
            depth_path = scene_dir / str(frame.get("depth", f"depth/frame_{frame_id:06d}.png"))
            instance_path = scene_dir / str(frame.get("instance_mask", f"instance_masks/frame_{frame_id:06d}.png"))
            surface_path = scene_dir / str(frame.get("surface_mask", f"surface_masks/frame_{frame_id:06d}.png"))
            npz_path = scene_dir / str(frame.get("visible_points", f"visible_points/frame_{frame_id:06d}.npz"))
            referenced_visible_npz.add(npz_path.resolve())
            image = inspect_raster(image_path, "image")
            depth = inspect_raster(depth_path, "depth")
            instance_mask = inspect_raster(instance_path, "instance_mask")
            surface_mask = inspect_raster(surface_path, "surface_mask")
            expected_shape = (height, width)
            for kind, array in (("image", image), ("depth", depth), ("instance_mask", instance_mask), ("surface_mask", surface_mask)):
                if array is not None and expected_shape != (0, 0) and tuple(array.shape[:2]) != expected_shape:
                    issue("raster_shape_mismatch", "error", f"{kind} shape disagrees with camera_info.", scene_id=scene_dir.name, frame_id=frame_id, shape=list(array.shape), expected=list(expected_shape))

            if not npz_path.is_file():
                issue("missing_visible_npz", "error", f"Referenced visible-point NPZ is absent: {npz_path}", scene_id=scene_dir.name, frame_id=frame_id)
                continue
            try:
                arrays_context = np.load(npz_path, allow_pickle=False)
            except (OSError, ValueError) as error:
                issue("invalid_visible_npz", "error", f"Could not open {npz_path}: {error}", path=str(npz_path))
                continue
            with arrays_context as arrays:
                inspected_visible_npz.add(npz_path.resolve())
                npz_key_orders[tuple(arrays.files)] += 1
                for key in arrays.files:
                    _record_array_schema(visible_schema_fields, key, arrays[key])
                missing_fields = [key for key in ALIGNED_VISIBLE_FIELDS if key not in arrays]
                if missing_fields:
                    issue("visible_npz_missing_fields", "error", "Visible NPZ misses required row-aligned fields.", path=str(npz_path), fields=missing_fields)
                    continue
                count = len(arrays["points_C"])
                frame_point_counts.append(count)
                row_invariant_checks["aligned_lengths"] += 1
                bad_lengths = {key: len(arrays[key]) for key in ALIGNED_VISIBLE_FIELDS if len(arrays[key]) != count}
                if bad_lengths:
                    row_invariant_failures["aligned_lengths"] += 1
                    issue("npz_row_length_mismatch", "error", "Row-aligned NPZ arrays have different first dimensions.", path=str(npz_path), expected=count, observed=bad_lengths)
                    continue
                labels = np.asarray(arrays["surface_label"])
                for key, label_value in (("shell_indices", 0), ("fracture_indices", 1)):
                    row_invariant_checks[key] += 1
                    if key not in arrays:
                        row_invariant_failures[key] += 1
                        issue("npz_index_array_absent", "error", f"{key} is absent.", path=str(npz_path))
                        continue
                    indices = np.asarray(arrays[key], dtype=np.int64)
                    expected = np.flatnonzero(labels == label_value)
                    if np.any(indices < 0) or np.any(indices >= count) or not np.array_equal(indices, expected):
                        row_invariant_failures[key] += 1
                        issue("npz_index_array_mismatch", "error", f"{key} does not exactly index surface_label=={label_value}.", path=str(npz_path))
                u = np.asarray(arrays["u"], dtype=np.int64)
                v = np.asarray(arrays["v"], dtype=np.int64)
                row_invariant_checks["uv_bounds"] += 1
                uv_valid = (u >= 0) & (v >= 0) & (u < width) & (v < height)
                if width <= 0 or height <= 0 or not bool(uv_valid.all()):
                    row_invariant_failures["uv_bounds"] += 1
                    issue("uv_out_of_bounds", "error", "NPZ u/v rows fall outside camera dimensions.", path=str(npz_path), invalid=int((~uv_valid).sum()))

                annotation_entries = frame.get("fragments", [])
                annotation_ids = [int(entry["fragment_id"]) for entry in annotation_entries if "fragment_id" in entry]
                duplicate_annotations = _duplicates(annotation_ids)
                if duplicate_annotations:
                    issue("duplicate_frame_fragment_id", "error", "Duplicate fragment IDs in one GT frame.", scene_id=scene_dir.name, frame_id=frame_id, fragment_ids=duplicate_annotations)
                annotations = {int(entry["fragment_id"]): entry for entry in annotation_entries if "fragment_id" in entry}
                npz_fragment_ids = set(int(value) for value in np.unique(arrays["fragment_id"]))
                if npz_fragment_ids != set(annotations):
                    issue("frame_fragment_join_mismatch", "error", "NPZ and frame GT fragment ID sets disagree.", scene_id=scene_dir.name, frame_id=frame_id, npz_ids=sorted(npz_fragment_ids), gt_ids=sorted(annotations))
                unknown_fragment_ids = npz_fragment_ids.difference(known_fragments)
                if unknown_fragment_ids:
                    issue("fragment_metadata_join_mismatch", "error", "Visible fragment IDs are absent from fragment_annotations.", scene_id=scene_dir.name, frame_id=frame_id, fragment_ids=sorted(unknown_fragment_ids))

                for fragment_id in sorted(npz_fragment_ids):
                    mask = np.asarray(arrays["fragment_id"] == fragment_id)
                    num_points = int(mask.sum())
                    annotation = annotations.get(fragment_id)
                    if annotation is None:
                        continue
                    points_C = np.asarray(arrays["points_C"][mask], dtype=np.float64)
                    points_O = np.asarray(arrays["points_O"][mask], dtype=np.float64)
                    points_F = np.asarray(arrays["points_F"][mask], dtype=np.float64)
                    transform_O = np.asarray(annotation.get("T_C_from_O"), dtype=np.float64)
                    transform_F = np.asarray(annotation.get("T_C_from_F"), dtype=np.float64) if annotation.get("T_C_from_F") is not None else None
                    error_O = None
                    if transform_O.shape == (4, 4):
                        predicted = points_O @ transform_O[:3, :3].T + transform_O[:3, 3]
                        error_O = float(np.max(np.abs(predicted - points_C))) if num_points else 0.0
                        transform_errors_O.append(error_O)
                        rotation_determinants.append(float(np.linalg.det(transform_O[:3, :3])))
                        last_row_errors.append(float(np.max(np.abs(transform_O[3] - [0.0, 0.0, 0.0, 1.0]))))
                    else:
                        issue("invalid_T_C_from_O", "error", "T_C_from_O is not 4x4.", scene_id=scene_dir.name, frame_id=frame_id, fragment_id=fragment_id)
                    error_F = None
                    if transform_F is not None and transform_F.shape == (4, 4):
                        predicted_F = points_F @ transform_F[:3, :3].T + transform_F[:3, 3]
                        error_F = float(np.max(np.abs(predicted_F - points_C))) if num_points else 0.0
                        transform_errors_F.append(error_F)

                    shell_count = int(((labels == 0) & mask).sum())
                    fracture_count = int(((labels == 1) & mask).sum())
                    unknown_count = int(((labels == 255) & mask).sum())
                    row_invariant_checks["face_and_barycentric_ranges"] += 1
                    face_ids = np.asarray(arrays["face_id"][mask], dtype=np.int64)
                    barycentric = np.asarray(arrays["barycentric"][mask], dtype=np.float64)
                    face_limit = fragment_face_counts.get(fragment_id)
                    bad_face_ids = bool(np.any(face_ids < 0)) or (
                        face_limit is not None and bool(np.any(face_ids >= face_limit))
                    )
                    bad_barycentric = (
                        not bool(np.isfinite(barycentric).all())
                        or bool(np.any(barycentric < -1e-3))
                        or bool(np.any(barycentric > 1.001))
                        or not bool(np.allclose(barycentric.sum(axis=1), 1.0, atol=2e-4, rtol=0.0))
                    )
                    if bad_face_ids or bad_barycentric:
                        row_invariant_failures["face_and_barycentric_ranges"] += 1
                        issue("face_barycentric_range_mismatch", "error", "face_id or barycentric rows violate fragment mesh bounds.", scene_id=scene_dir.name, frame_id=frame_id, fragment_id=fragment_id)
                    gt_shell = annotation.get("visible_shell_pixels")
                    gt_fracture = annotation.get("visible_fracture_pixels")
                    visible_pixels = annotation.get("visible_pixels")
                    if gt_shell is not None and gt_fracture is not None and num_points != int(gt_shell) + int(gt_fracture):
                        issue("npz_gt_surface_count_mismatch", "error", "NPZ row count differs from GT shell+fracture visible counts.", scene_id=scene_dir.name, frame_id=frame_id, fragment_id=fragment_id)
                    omitted_visible = int(visible_pixels) - num_points if visible_pixels is not None else None
                    if omitted_visible is not None:
                        visible_pixel_differences.append(omitted_visible)

                    instance_value = annotation.get("instance_mask_value")
                    instance_pixels = None
                    surface_shell_pixels = None
                    surface_fracture_pixels = None
                    surface_invalid_pixels = None
                    surface_unlabeled_pixels = None
                    if instance_mask is not None and instance_value is not None:
                        region = instance_mask == int(instance_value)
                        instance_pixels = int(region.sum())
                        if visible_pixels is not None and instance_pixels != int(visible_pixels):
                            issue("instance_mask_count_mismatch", "error", "Instance-mask pixel count differs from GT visible_pixels.", scene_id=scene_dir.name, frame_id=frame_id, fragment_id=fragment_id)
                        if surface_mask is not None:
                            surface_shell_pixels = int(((surface_mask == 1) & region).sum())
                            surface_fracture_pixels = int(((surface_mask == 2) & region).sum())
                            surface_invalid_pixels = int(((surface_mask == 255) & region).sum())
                            surface_unlabeled_pixels = int(((surface_mask == 0) & region).sum())
                            if gt_shell is not None and surface_shell_pixels != int(gt_shell):
                                issue("surface_mask_shell_count_mismatch", "error", "Surface-mask shell count differs from GT.", scene_id=scene_dir.name, frame_id=frame_id, fragment_id=fragment_id)
                            if gt_fracture is not None and surface_fracture_pixels != int(gt_fracture):
                                issue("surface_mask_fracture_count_mismatch", "error", "Surface-mask fracture count differs from GT.", scene_id=scene_dir.name, frame_id=frame_id, fragment_id=fragment_id)
                            if omitted_visible is not None and surface_unlabeled_pixels != omitted_visible:
                                issue("surface_mask_unlabeled_count_mismatch", "error", "Surface-mask value-0 pixels inside the instance do not explain visible pixels omitted from NPZ.", scene_id=scene_dir.name, frame_id=frame_id, fragment_id=fragment_id)
                    if instance_mask is not None and surface_mask is not None and bool(uv_valid.all()) and instance_value is not None:
                        selected_instance = instance_mask[v[mask], u[mask]]
                        selected_surface = surface_mask[v[mask], u[mask]]
                        expected_surface = np.where(labels[mask] == 0, 1, np.where(labels[mask] == 1, 2, 255))
                        row_invariant_checks["mask_pixel_join"] += 1
                        if not bool((selected_instance == int(instance_value)).all()) or not np.array_equal(selected_surface, expected_surface):
                            row_invariant_failures["mask_pixel_join"] += 1
                            issue("mask_pixel_join_mismatch", "error", "NPZ u/v/labels do not map to the expected instance/surface mask pixels.", scene_id=scene_dir.name, frame_id=frame_id, fragment_id=fragment_id)
                    if depth is not None and bool(uv_valid.all()):
                        rendered_depth_m = depth[v[mask], u[mask]].astype(np.float64) * depth_scale_m
                        depth_errors_m.append(np.abs(rendered_depth_m - points_C[:, 2]))

                    sample_rows.append(
                        {
                            "sample_id": f"{scene_dir.name}/frame_{frame_id:06d}/fragment_{fragment_id:04d}",
                            "scene_id": scene_dir.name,
                            "frame_id": frame_id,
                            "fragment_id": fragment_id,
                            "object_model_id": Path(str(scene_meta.get("object_model", fragment_meta.get("object_model", "")))).stem,
                            "num_points": num_points,
                            "num_shell": shell_count,
                            "num_fracture": fracture_count,
                            "num_unknown": unknown_count,
                            "visible_pixels_gt": visible_pixels if visible_pixels is not None else "",
                            "visible_pixels_omitted_from_npz": omitted_visible if omitted_visible is not None else "",
                            "surface_invalid_pixels": surface_invalid_pixels if surface_invalid_pixels is not None else "",
                            "surface_unlabeled_pixels": surface_unlabeled_pixels if surface_unlabeled_pixels is not None else "",
                            "T_C_from_O_points_max_abs_error_m": error_O if error_O is not None else "",
                            "T_C_from_F_points_max_abs_error_m": error_F if error_F is not None else "",
                            "visible_points": str(npz_path.relative_to(root)),
                        }
                    )
                    scene_sample_count += 1

            support_frame = support_frames.get(frame_id)
            if support_gt and support_frame is None:
                issue("scene_gt_frame_join_mismatch", "error", "GT frame has no support_plane_gt frame entry.", scene_id=scene_dir.name, frame_id=frame_id)
            if support_frame:
                scene_gt_counts["frame_entries"] += 1
                mesh_rel = support_frame.get("static_scene_mesh_C")
                depth_rel = support_frame.get("static_scene_depth_C")
                if mesh_rel:
                    mesh_path = scene_dir / str(mesh_rel)
                    if mesh_path.is_file():
                        scene_gt_counts["static_scene_mesh_C"] += 1
                        try:
                            inspect_ply_header(mesh_path)
                        except (OSError, ValueError) as error:
                            issue("invalid_scene_gt_mesh", "error", f"Could not inspect {mesh_path}: {error}", path=str(mesh_path))
                    else:
                        issue("missing_scene_gt_mesh", "error", f"Missing camera-frame scene mesh: {mesh_path}", path=str(mesh_path))
                if depth_rel:
                    if inspect_raster(scene_dir / str(depth_rel), "scene_gt_depth") is not None:
                        scene_gt_counts["static_scene_depth_C"] += 1
                if support_frame.get("T_C_from_W") is not None and frame.get("T_C_from_W") is not None:
                    if not np.allclose(support_frame["T_C_from_W"], frame["T_C_from_W"], atol=1e-9, rtol=0.0):
                        issue("scene_gt_transform_join_mismatch", "error", "T_C_from_W differs between GT and support_plane_gt.", scene_id=scene_dir.name, frame_id=frame_id)

        all_visible_files = {path.resolve() for path in (scene_dir / "visible_points").glob("*.npz")}
        scene_referenced = {path for path in referenced_visible_npz if path.parent.parent == scene_dir.resolve()}
        orphans = sorted(all_visible_files - scene_referenced)
        if orphans:
            issue("orphan_visible_npz", "warning", "Visible-point NPZ files are not referenced by GT.", scene_id=scene_dir.name, paths=[str(path) for path in orphans])
        scene_summaries.append(
            {
                "scene_id": scene_dir.name,
                "status": "inspected",
                "num_frames": len(frame_entries),
                "num_fragments_total": fragment_meta.get("num_fragments_total"),
                "num_fragments_annotated": fragment_meta.get("num_fragments_annotated"),
                "num_samples": scene_sample_count,
                "object_model": scene_meta.get("object_model", fragment_meta.get("object_model")),
                "missing_paths": missing,
                "orphan_visible_npz": [str(path.relative_to(root)) for path in orphans],
                "scene_gt_frame_entries": len(support_frames),
            }
        )

    template_inventory: list[dict[str, Any]] = []
    missing_sidecar_templates: list[str] = []
    for template_path in templates:
        try:
            header = inspect_ply_header(template_path)
            mesh = load_ply(template_path)
        except (OSError, ValueError) as error:
            issue("invalid_template_ply", "error", f"Could not inspect {template_path}: {error}", path=str(template_path))
            continue
        meta_path = template_path.with_suffix(".meta.json")
        metadata = read_json(meta_path, required=False)
        object_id = str((metadata or {}).get("object_id", template_path.stem.split("__", 1)[0]))
        candidates = [template_path.with_suffix(".symmetry.json"), template_path.parent / f"{object_id}.symmetry.json"]
        sidecar = next((path for path in candidates if path.is_file()), None)
        if sidecar is None:
            missing_sidecar_templates.append(template_path.stem)
            issue(
                "symmetry_sidecar_absent",
                "warning",
                f"No symmetry sidecar is present for template {template_path.stem}; symmetry_available must remain false.",
                object_model_id=template_path.stem,
            )
        points = mesh["points"]
        template_inventory.append(
            {
                "object_model_id": template_path.stem,
                "object_id": object_id,
                "path": str(template_path.relative_to(root)),
                "format": header["format"],
                "elements": header["elements"],
                "comments": header["comments"],
                "num_vertices": int(len(points)),
                "num_faces": int(len(mesh["faces"])) if mesh["faces"] is not None else 0,
                "has_vertex_normals": mesh["normals"] is not None,
                "bounds_min_m": points.min(axis=0).tolist(),
                "bounds_max_m": points.max(axis=0).tolist(),
                "meta_path": str(meta_path.relative_to(root)) if meta_path.is_file() else None,
                "meta": metadata,
                "symmetry_sidecar": str(sidecar.relative_to(root)) if sidecar else None,
            }
        )

    if "normals_C" not in visible_schema_fields:
        issue("observed_normals_absent", "info", "Visible-point NPZ files do not contain camera-frame normals; loader returns normals_C=None.")
    if "profile" not in visible_schema_fields and "profile" not in fragment_sample_fields:
        issue("profile_absent", "info", "Neither visible-point nor fragment-sample NPZ files contain a profile array.")
    sample_counts = [int(row["num_points"]) for row in sample_rows]
    below_default = [row["sample_id"] for row in sample_rows if int(row["num_points"]) < 128]
    if below_default:
        issue("samples_below_default_minimum", "warning", f"{len(below_default)} samples contain fewer than 128 points and are filtered by the default loader.", examples=below_default[:10])
    orphan_global = sorted(
        {path.resolve() for scene in scenes for path in (scene / "visible_points").glob("*.npz")}
        - referenced_visible_npz
    )
    uninspected_referenced = sorted(referenced_visible_npz - inspected_visible_npz)

    raster_inventory: dict[str, Any] = {}
    for kind, tracker in sorted(raster_trackers.items()):
        raster_inventory[kind] = {
            **tracker,
            "modes": sorted(tracker["modes"]),
            "sizes_wh": [list(value) for value in sorted(tracker["sizes_wh"])],
            "shapes": [list(value) for value in sorted(tracker["shapes"])],
            "dtypes": sorted(tracker["dtypes"]),
            "unique_values": sorted(tracker["unique_values"]),
        }
    depth_values = np.concatenate(depth_errors_m) if depth_errors_m else np.empty(0)
    first_camera = camera_examples[0] if camera_examples else {}
    inventory = {
        "dataset_root": str(root),
        "split_name": root.name,
        "num_scenes": len(scenes),
        "num_gt_frames": sum(int(summary.get("num_frames", 0)) for summary in scene_summaries),
        "num_visible_npz_inspected": len(inspected_visible_npz),
        "num_samples": len(sample_rows),
        "num_templates": len(templates),
        "observed_points": {
            "minimum_per_sample": min(sample_counts) if sample_counts else None,
            "maximum_per_sample": max(sample_counts) if sample_counts else None,
            "mean_per_sample": float(np.mean(sample_counts)) if sample_counts else None,
            "samples_below_128": len(below_default),
            "samples_above_4096": sum(count > 4096 for count in sample_counts),
        },
        "file_counts": dict(sorted(directory_counts.items())),
        "camera_schema": {
            "top_level_keys": sorted(json_top_level_keys.get("camera_info.json", set())),
            "width": first_camera.get("width"),
            "height": first_camera.get("height"),
            "K": first_camera.get("K"),
            "fx": first_camera.get("fx"),
            "fy": first_camera.get("fy"),
            "cx": first_camera.get("cx"),
            "cy": first_camera.get("cy"),
            "depth_format": first_camera.get("depth_format"),
            "depth_units": first_camera.get("depth_units"),
            "depth_scale_m": first_camera.get("depth_scale_m"),
            "coordinate_convention": first_camera.get("coordinate_convention"),
            "extrinsics_convention": first_camera.get("extrinsics_convention"),
        },
        "raster_inventory": raster_inventory,
        "json_top_level_keys": {key: sorted(values) for key, values in sorted(json_top_level_keys.items())},
        "coordinate_contract": {
            "matrix_convention": "BOP/OpenCV, column-vector homogeneous transforms",
            "point_coordinate_unit": "m (scene_unit in metadata)",
            "row_correspondence": "validated by aligned lengths, transforms, label index arrays, u/v bounds and mask-pixel joins",
            "T_C_from_O_max_abs_point_error_m": max(transform_errors_O, default=None),
            "T_C_from_F_max_abs_point_error_m": max(transform_errors_F, default=None),
            "T_C_from_O_rotation_det_min": min(rotation_determinants, default=None),
            "T_C_from_O_rotation_det_max": max(rotation_determinants, default=None),
            "T_C_from_O_last_row_max_abs_error": max(last_row_errors, default=None),
            "depth_at_uv_vs_points_C_z_abs_error_m": {
                "count": int(depth_values.size),
                "mean": float(depth_values.mean()) if depth_values.size else None,
                "p95": float(np.quantile(depth_values, 0.95)) if depth_values.size else None,
                "max": float(depth_values.max()) if depth_values.size else None,
                "note": "Measured diagnostic only; source depth and exported visible geometry are not assumed identical.",
            },
        },
        "row_invariant_validation": {
            "checks": dict(row_invariant_checks),
            "failures": dict(row_invariant_failures),
            "all_passed": not any(row_invariant_failures.values()),
        },
        "visible_pixel_filtering": {
            "relationship": "NPZ rows equal GT visible_shell_pixels + visible_fracture_pixels; GT visible_pixels additionally includes instance pixels whose surface-mask value is 0 (unlabeled/background encoding).",
            "omitted_pixels_min": min(visible_pixel_differences, default=None),
            "omitted_pixels_max": max(visible_pixel_differences, default=None),
            "all_samples_have_omitted_pixels": bool(visible_pixel_differences) and all(value > 0 for value in visible_pixel_differences),
        },
        "fragment_geometry": {
            "mesh_count": len(fragment_vertices),
            "vertices_min": min(fragment_vertices, default=None),
            "vertices_max": max(fragment_vertices, default=None),
            "faces_min": min(fragment_faces, default=None),
            "faces_max": max(fragment_faces, default=None),
        },
        "scene_gt": dict(scene_gt_counts),
        "orphan_visible_npz": [str(path) for path in orphan_global],
        "uninspected_referenced_visible_npz": [str(path) for path in uninspected_referenced],
        "scenes": scene_summaries,
    }
    npz_schema = {
        "num_referenced_files": len(referenced_visible_npz),
        "num_files_inspected": len(inspected_visible_npz),
        "orphan_files": [str(path) for path in orphan_global],
        "uninspected_referenced_files": [str(path) for path in uninspected_referenced],
        "key_orders": [{"keys": list(keys), "num_files": count} for keys, count in npz_key_orders.most_common()],
        "fields": _serialize_schema(visible_schema_fields),
        "frame_total_points_min": min(frame_point_counts, default=None),
        "frame_total_points_max": max(frame_point_counts, default=None),
        "row_alignment": {
            "aligned_with_points_C": list(ALIGNED_VISIBLE_FIELDS),
            "index_arrays": ["shell_indices", "fracture_indices"],
            "validation": inventory["row_invariant_validation"],
        },
        "fragment_sample_npz": {
            "num_files": int(directory_counts["fragment_sample_npz"]),
            "fields": _serialize_schema(fragment_sample_fields),
            "profile_present": "profile" in fragment_sample_fields,
        },
    }

    _write_json(output / "dataset_inventory.json", inventory)
    _write_json(output / "npz_schema.json", npz_schema)
    _write_json(output / "template_inventory.json", {"templates": template_inventory})
    _write_json(output / "warnings.json", {"warnings": warnings})
    fieldnames = list(sample_rows[0]) if sample_rows else ["sample_id"]
    with (output / "sample_index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)
    error_count = sum(warning["severity"] == "error" for warning in warnings)
    markdown = f"""# Dataset inventory: `{root.name}`

- Root: `{root}`
- Scenes / GT frames / fragment samples: **{len(scenes)} / {inventory['num_gt_frames']} / {len(sample_rows)}**
- Inspected referenced visible NPZs: **{len(inspected_visible_npz)} / {len(referenced_visible_npz)}**; orphan NPZs: **{len(orphan_global)}**
- Templates: **{len(templates)}**; missing symmetry sidecars: **{len(missing_sidecar_templates)}**
- Per-fragment observed points: **{min(sample_counts) if sample_counts else 'n/a'}..{max(sample_counts) if sample_counts else 'n/a'}** (variable N)
- Samples below 128 / above 4096: **{len(below_default)} / {sum(count > 4096 for count in sample_counts)}**
- `points_O` → `T_C_from_O` → `points_C` max error: **{max(transform_errors_O, default=float('nan')):.3e} m**
- `points_F` → `T_C_from_F` → `points_C` max error: **{max(transform_errors_F, default=float('nan')):.3e} m**
- Row/index/u-v/mask invariant failures: **{sum(row_invariant_failures.values())}**
- Error-severity audit findings: **{error_count}**
- GT visible pixels omitted from NPZ: **{min(visible_pixel_differences, default='n/a')}..{max(visible_pixel_differences, default='n/a')} per sample** (surface-mask value 0 inside the instance)

The JSON inventory records camera intrinsics, raster shapes/dtypes/encodings,
scene-GT joins, transform checks, mask joins, depth-at-pixel diagnostics and all
missing/corrupt assets. See `warnings.json` for actionable findings.
"""
    (output / "dataset_inventory.md").write_text(markdown, encoding="utf-8")
    return {
        "out_dir": str(output),
        "num_scenes": len(scenes),
        "num_samples": len(sample_rows),
        "point_count_range": [min(sample_counts), max(sample_counts)] if sample_counts else None,
        "transform_max_abs_error_m": max(transform_errors_O, default=None),
        "row_invariant_failures": int(sum(row_invariant_failures.values())),
        "error_findings": error_count,
        "num_warnings": len(warnings),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(inspect_dataset(args.dataset_root, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
