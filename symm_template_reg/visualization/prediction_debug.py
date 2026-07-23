"""Deterministic predicted-pose PLY artifacts for epoch-based debug training."""

from __future__ import annotations

import colorsys
import gc
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from symm_template_reg.datasets.template_repository import load_ply
from symm_template_reg.engine.evaluator import move_to_device
from symm_template_reg.engine.metrics import batch_pose_metric_rows
from symm_template_reg.engine.single_fragment import world_pose_consistency
from symm_template_reg.geometry import closest_points_on_triangle_mesh
from symm_template_reg.models.pose.pose_representation import (
    invert_transform,
    transform_points,
)
from symm_template_reg.models.geometry.point_ops import farthest_point_indices
from symm_template_reg.models.geometry.aux_guided_triangle_candidates import (
    AuxGuidedTriangleCandidateBuilder,
)
from symm_template_reg.models.heads.coordinate_guided_surface_projection import (
    CoordinateGuidedSurfaceProjectionHead,
)
from symm_template_reg.evaluation.correspondence_diagnostics import attention_distribution_metrics
from symm_template_reg.models.symmetry.groups import SO2Group, group_to_dict
from symm_template_reg.models.symmetry.hypothesis_expander import (
    place_fragment_for_hypothesis,
    visualization_equivalent_pose_set,
)
from symm_template_reg.models.symmetry.region_assignment import (
    assign_symmetry_regions,
    effective_group_from_regions,
    region_indices_from_membership,
)
from symm_template_reg.models.symmetry.pose_conditioned_resolver import (
    PoseConditionedSymmetryResolver,
)

from .ply import write_colored_ply
from .symmetry_debug import (
    GeometryBuilder,
    _TriangleSurfaceIndex,
    _add_axis_boundaries_and_origin,
    _project_fragment_shell_to_template,
    _refine_template_at_fragment_boundary,
    _triangle_areas,
)


WARNING_FLAGS = {
    "debug_training_on_test_split": True,
    "train_and_validation_use_same_samples": True,
    "results_are_not_final_evaluation": True,
}

LIGHT_GRAY = np.asarray([205, 205, 205], dtype=np.uint8)
OBSERVED_GREEN = np.asarray([35, 230, 80], dtype=np.uint8)
PREDICTED_BLUE = np.asarray([50, 105, 235], dtype=np.uint8)
AXIS_BLUE = np.asarray([15, 45, 130], dtype=np.uint8)
GT_ORANGE = np.asarray([245, 135, 35], dtype=np.uint8)
REFERENCE_RED = np.asarray([235, 45, 45], dtype=np.uint8)


@dataclass(frozen=True)
class _FragmentProjectionSource:
    mesh_path: Path
    face_labels_path: Path
    points_C: np.ndarray
    faces: np.ndarray
    face_labels: np.ndarray
    shell_faces: np.ndarray


def _fragment_projection_source(sample: Mapping[str, Any]) -> _FragmentProjectionSource:
    """Load the physical fragment mesh and place it in the sample camera frame."""

    mesh_path = Path(str(sample["meta"]["fragment_mesh"]["mesh_path"])).resolve()
    mesh = load_ply(mesh_path)
    faces_value = mesh.get("faces")
    if faces_value is None:
        raise ValueError(f"fragment mesh has no triangular faces: {mesh_path}")
    faces = np.asarray(faces_value, dtype=np.int64)

    annotation_path = mesh_path.parent / "fragment_annotations.json"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    fragment_id = int(sample["fragment_id"])
    entry = next(
        (
            value
            for value in annotations.get("fragments", [])
            if int(value.get("fragment_id", -1)) == fragment_id
        ),
        None,
    )
    if entry is None:
        raise KeyError(f"fragment {fragment_id} is absent from {annotation_path}")
    face_labels_path = (mesh_path.parent.parent / str(entry["face_labels"])).resolve()
    face_labels = np.asarray(np.load(face_labels_path), dtype=np.uint8)
    if face_labels.shape != (len(faces),):
        raise ValueError(
            f"face-label count {len(face_labels)} does not match {len(faces)} faces"
        )
    shell_faces = faces[face_labels == 0]
    if not len(shell_faces):
        raise ValueError(f"fragment has no shell faces: {mesh_path}")

    transform = sample["gt"].get("T_C_from_F")
    if not isinstance(transform, torch.Tensor):
        raise ValueError("predicted footprint visualization requires gt.T_C_from_F")
    transform_np = transform.detach().cpu().numpy().astype(np.float64)
    points_F = np.asarray(mesh["points"], dtype=np.float64)
    points_C = points_F @ transform_np[:3, :3].T + transform_np[:3, 3]
    return _FragmentProjectionSource(
        mesh_path=mesh_path,
        face_labels_path=face_labels_path,
        points_C=points_C.astype(np.float32),
        faces=faces,
        face_labels=face_labels,
        shell_faces=shell_faces,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _bright_colors(count: int) -> np.ndarray:
    return np.asarray(
        [
            [round(255 * value) for value in colorsys.hsv_to_rgb(i / max(count, 1), 0.82, 1.0)]
            for i in range(count)
        ],
        dtype=np.uint8,
    )


def _reference_direction(axis: np.ndarray) -> np.ndarray:
    candidate = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(candidate, axis))) > 0.9:
        candidate = np.asarray([0.0, 0.0, 1.0])
    projected = candidate - float(np.dot(candidate, axis)) * axis
    return projected / max(float(np.linalg.norm(projected)), 1e-12)


def _shell_points(sample: Mapping[str, Any]) -> torch.Tensor:
    points = sample["observed"]["points_C"]
    labels = sample["observed"].get("surface_labels")
    if isinstance(labels, torch.Tensor):
        shell = points[labels.eq(0)]
        if len(shell):
            return shell
    return points


def _axis_points(
    transform: torch.Tensor,
    metadata: Any,
    extent: float,
) -> tuple[np.ndarray, np.ndarray]:
    origin = torch.as_tensor(metadata.axis.origin, dtype=transform.dtype, device=transform.device)
    axis = torch.as_tensor(metadata.axis.direction, dtype=transform.dtype, device=transform.device)
    reference = torch.as_tensor(
        _reference_direction(axis.detach().cpu().numpy()),
        dtype=transform.dtype,
        device=transform.device,
    )
    object_points = torch.stack(
        (origin, origin + axis * extent, origin + reference * extent * 0.65)
    )
    camera_points = transform_points(transform, object_points).detach().cpu().numpy()
    edges = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
    return camera_points, edges


def _camera_frame_ply(
    path: Path,
    sample: Mapping[str, Any],
    predicted_pose: torch.Tensor,
    *,
    include_gt: bool,
) -> None:
    template_points = sample["template"]["points_O"].to(predicted_pose.device)
    faces = sample["template"].get("faces")
    if not isinstance(faces, torch.Tensor):
        raise ValueError("prediction visualization requires template faces")
    faces_np = faces.detach().cpu().numpy().astype(np.int64)
    metadata = sample["template"]["symmetry_metadata"]
    observed = _shell_points(sample).detach().cpu().numpy()
    predicted = transform_points(predicted_pose, template_points).detach().cpu().numpy()
    extent = max(float(np.ptp(template_points.detach().cpu().numpy(), axis=0).max()), 1e-4)
    axis_points, axis_edges = _axis_points(predicted_pose, metadata, extent)
    vertices = [observed, predicted, axis_points]
    colors = [
        np.broadcast_to(OBSERVED_GREEN, (len(observed), 3)),
        np.broadcast_to(PREDICTED_BLUE, (len(predicted), 3)),
        np.asarray([AXIS_BLUE, AXIS_BLUE, REFERENCE_RED], dtype=np.uint8),
    ]
    face_blocks = [faces_np + len(observed)]
    edge_offset = len(observed) + len(predicted)
    edge_blocks = [axis_edges + edge_offset]
    if include_gt:
        gt_pose = sample["gt"]["T_C_from_O"].to(predicted_pose.device)
        gt_points = transform_points(gt_pose, template_points).detach().cpu().numpy()
        gt_axis_points, gt_axis_edges = _axis_points(gt_pose, metadata, extent)
        gt_offset = sum(len(block) for block in vertices)
        vertices.extend([gt_points, gt_axis_points])
        colors.extend(
            [
                np.broadcast_to(GT_ORANGE, (len(gt_points), 3)),
                np.broadcast_to(GT_ORANGE, (len(gt_axis_points), 3)),
            ]
        )
        face_blocks.append(faces_np + gt_offset)
        edge_blocks.append(gt_axis_edges + gt_offset + len(gt_points))
    write_colored_ply(
        path,
        np.concatenate(vertices),
        np.concatenate(colors),
        faces=np.concatenate(face_blocks),
        edges=np.concatenate(edge_blocks),
    )


def _gallery_offsets(count: int, columns: int, spacing: float) -> np.ndarray:
    result = []
    for index in range(count):
        row, column = divmod(index, max(columns, 1))
        result.append([column * spacing, 0.0, -row * spacing])
    return np.asarray(result, dtype=np.float32)


def _projected_template_footprint(
    *,
    template: np.ndarray,
    faces: np.ndarray,
    shell_surface_C: _TriangleSurfaceIndex,
    pose: torch.Tensor,
    boundary_resolution_m: float,
    boundary_max_depth: int,
) -> tuple[Any, dict[str, Any]]:
    pose_np = pose.detach().cpu().numpy()
    vertex_mask, face_mask = _project_fragment_shell_to_template(
        template_points=template,
        template_faces=faces,
        shell_surface=shell_surface_C,
        template_to_fragment_surface=pose_np,
    )
    refined = _refine_template_at_fragment_boundary(
        template_points=template,
        template_faces=faces,
        shell_surface=shell_surface_C,
        template_to_fragment_surface=pose_np,
        boundary_resolution_m=boundary_resolution_m,
        max_depth=boundary_max_depth,
    )
    areas = _triangle_areas(refined.vertices, refined.faces)
    diagnostics = {
        "projected_template_vertices": int(vertex_mask.sum()),
        "projected_template_faces": int(face_mask.sum()),
        "output_template_faces": int(len(refined.faces)),
        "output_projected_face_pieces": int(refined.projected_faces.sum()),
        "boundary_split_source_faces": int(refined.split_source_faces),
        "output_projected_surface_area_m2": float(
            areas[refined.projected_faces].sum()
        ),
    }
    return refined, diagnostics


def _fragment_regions_on_template(
    path: Path,
    sample: Mapping[str, Any],
    pose: torch.Tensor,
    shell_surface_C: _TriangleSurfaceIndex,
    *,
    footprint_color: np.ndarray,
    boundary_resolution_m: float,
    boundary_max_depth: int,
    comments: Sequence[str] | None = None,
) -> dict[str, Any]:
    template = sample["template"]["points_O"].detach().cpu().numpy()
    faces_value = sample["template"].get("faces")
    if not isinstance(faces_value, torch.Tensor):
        raise ValueError("predicted footprint requires template faces")
    faces = faces_value.detach().cpu().numpy().astype(np.int64)
    refined, diagnostics = _projected_template_footprint(
        template=template,
        faces=faces,
        shell_surface_C=shell_surface_C,
        pose=pose,
        boundary_resolution_m=boundary_resolution_m,
        boundary_max_depth=boundary_max_depth,
    )
    colors = np.broadcast_to(LIGHT_GRAY, (len(refined.faces), 3)).copy()
    colors[refined.projected_faces] = footprint_color
    builder = GeometryBuilder()
    builder.add_face_colored_mesh(refined.vertices, refined.faces, colors)
    _add_axis_boundaries_and_origin(
        builder,
        template,
        sample["template"]["symmetry_metadata"],
        include_reference=True,
    )
    builder.write(path, comments=comments)
    return diagnostics


def _registered_fragment_on_template(
    path: Path,
    sample: Mapping[str, Any],
    pose: torch.Tensor,
    fragment_source: _FragmentProjectionSource,
) -> dict[str, Any]:
    """Show the complete physical fragment in the object frame implied by the model."""

    template = sample["template"]["points_O"].detach().cpu().numpy()
    template_faces_value = sample["template"].get("faces")
    if not isinstance(template_faces_value, torch.Tensor):
        raise ValueError("registered-fragment visualization requires template faces")
    template_faces = template_faces_value.detach().cpu().numpy().astype(np.int64)
    fragment_C = torch.as_tensor(
        fragment_source.points_C,
        dtype=pose.dtype,
        device=pose.device,
    )
    fragment_O = (
        transform_points(invert_transform(pose), fragment_C)
        .detach()
        .cpu()
        .numpy()
    )
    fragment_face_colors = np.broadcast_to(
        np.asarray([235, 85, 45], dtype=np.uint8),
        (len(fragment_source.faces), 3),
    ).copy()
    fragment_face_colors[fragment_source.face_labels == 0] = PREDICTED_BLUE
    builder = GeometryBuilder()
    builder.add_mesh(template, template_faces, LIGHT_GRAY)
    builder.add_face_colored_mesh(
        fragment_O,
        fragment_source.faces,
        fragment_face_colors,
    )
    _add_axis_boundaries_and_origin(
        builder,
        template,
        sample["template"]["symmetry_metadata"],
        include_boundaries=False,
        include_reference=True,
    )
    builder.write(path)
    template_center = template.mean(axis=0)
    fragment_center = fragment_O.mean(axis=0)
    return {
        "representation": "full physical fragment transformed by inverse(predicted T_C_from_O) @ dataset T_C_from_F",
        "model_output_used": "top-1 predicted T_C_from_O",
        "dataset_debug_reference_used": "gt.T_C_from_F places the known physical fragment in camera coordinates",
        "shell_color_rgb": PREDICTED_BLUE.tolist(),
        "fracture_color_rgb": [235, 85, 45],
        "fragment_center_distance_to_template_center_mm": float(
            np.linalg.norm(fragment_center - template_center) * 1000.0
        ),
    }


def _predicted_gallery(
    path: Path,
    sample: Mapping[str, Any],
    poses: torch.Tensor,
    metadata: Any,
    shell_surface_C: _TriangleSurfaceIndex,
    *,
    columns: int,
    spacing_scale: float,
    boundary_resolution_m: float,
    boundary_max_depth: int,
    show_progress: bool,
) -> dict[str, Any]:
    template = sample["template"]["points_O"].detach().cpu().numpy()
    faces = sample["template"].get("faces")
    if not isinstance(faces, torch.Tensor):
        raise ValueError("prediction gallery requires template faces")
    faces = faces.detach().cpu().numpy().astype(np.int64)
    extent = max(float(np.ptp(template, axis=0).max()), 1e-4)
    offsets = _gallery_offsets(len(poses), columns, extent * spacing_scale)
    hypothesis_colors = _bright_colors(len(poses))
    builder = GeometryBuilder()
    entries = []
    pose_progress = tqdm(
        poses,
        desc="      symmetry hypotheses",
        unit="hyp",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for index, pose in enumerate(pose_progress):
        offset = offsets[index]
        refined, diagnostics = _projected_template_footprint(
            template=template,
            faces=faces,
            shell_surface_C=shell_surface_C,
            pose=pose,
            boundary_resolution_m=boundary_resolution_m,
            boundary_max_depth=boundary_max_depth,
        )
        colors = np.broadcast_to(LIGHT_GRAY, (len(refined.faces), 3)).copy()
        colors[refined.projected_faces] = hypothesis_colors[index]
        builder.add_face_colored_mesh(refined.vertices + offset, refined.faces, colors)
        _add_axis_boundaries_and_origin(
            builder,
            template,
            metadata,
            offset=offset,
            include_boundaries=False,
            include_reference=True,
        )
        entries.append(
            {
                "index": index,
                "color_rgb": hypothesis_colors[index].tolist(),
                "gallery_offset_m": offset.tolist(),
                "T_C_from_O": pose.detach().cpu().tolist(),
                **diagnostics,
            }
        )
    builder.write(path)
    return {
        "template_copy_count": len(poses),
        "observed_shell_copy_count": 0,
        "colored_template_footprint_copy_count": len(poses),
        "representation": "strict predicted fragment/template surface overlap colored on adaptively split template faces",
        "hypotheses": entries,
    }


def _topk_base_gallery(
    path: Path,
    sample: Mapping[str, Any],
    poses: torch.Tensor,
) -> None:
    template = sample["template"]["points_O"].to(poses.device)
    faces = sample["template"].get("faces")
    if not isinstance(faces, torch.Tensor):
        raise ValueError("top-K gallery requires template faces")
    faces_np = faces.detach().cpu().numpy().astype(np.int64)
    extent = max(float(np.ptp(template.detach().cpu().numpy(), axis=0).max()), 1e-4)
    colors = _bright_colors(len(poses))
    vertices, rgb, face_blocks = [], [], []
    offset = 0
    for index, pose in enumerate(poses):
        placed = transform_points(pose, template).detach().cpu().numpy()
        placed[:, 0] += index * extent * 1.5
        vertices.append(placed)
        rgb.append(np.broadcast_to(colors[index], (len(placed), 3)))
        face_blocks.append(faces_np + offset)
        offset += len(placed)
    write_colored_ply(
        path,
        np.concatenate(vertices),
        np.concatenate(rgb),
        faces=np.concatenate(face_blocks),
    )


def _point_region_colors(region_ids: torch.Tensor, region_count: int) -> np.ndarray:
    palette = _bright_colors(region_count)
    indices = region_ids.detach().cpu().numpy().astype(np.int64)
    colors = np.broadcast_to(REFERENCE_RED, (len(indices), 3)).copy()
    assigned = (indices >= 0) & (indices < region_count)
    colors[assigned] = palette[indices[assigned]]
    return colors


def _visible_points_regions_on_template(
    path: Path,
    sample: Mapping[str, Any],
    points_O: torch.Tensor,
    region_ids: torch.Tensor,
    *,
    comments: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Show only inference-available visible points over the reference template."""

    template = sample["template"]["points_O"].detach().cpu().numpy()
    faces_value = sample["template"].get("faces")
    if not isinstance(faces_value, torch.Tensor):
        raise ValueError("visible-point visualization requires template faces")
    faces = faces_value.detach().cpu().numpy().astype(np.int64)
    region_count = len(sample["template"]["symmetry_metadata"].regions)
    visible = points_O.detach().cpu().numpy()
    builder = GeometryBuilder()
    builder.add_mesh(template, faces, LIGHT_GRAY)
    builder.add_points(visible, _point_region_colors(region_ids, region_count))
    _add_axis_boundaries_and_origin(
        builder,
        template,
        sample["template"]["symmetry_metadata"],
        include_reference=True,
    )
    builder.write(path, comments=comments)
    return {
        "geometry_source": "observed_visible_points_C transformed by inverse(model top-1 pose)",
        "uses_full_fragment_mesh": False,
        "num_visible_points": len(visible),
        "out_of_sidecar_bounds_points": int(region_ids.lt(0).sum()),
    }


def _visible_points_gallery(
    path: Path,
    sample: Mapping[str, Any],
    expanded_poses: torch.Tensor,
    points_O: torch.Tensor,
    region_ids: torch.Tensor,
    *,
    columns: int,
    spacing_scale: float,
    comments: Sequence[str],
) -> dict[str, Any]:
    """Create template copies with the same visible observation on every copy."""

    template = sample["template"]["points_O"].detach().cpu().numpy()
    faces_value = sample["template"].get("faces")
    if not isinstance(faces_value, torch.Tensor):
        raise ValueError("visible gallery requires template faces")
    faces = faces_value.detach().cpu().numpy().astype(np.int64)
    visible = points_O.detach().cpu().numpy()
    region_count = len(sample["template"]["symmetry_metadata"].regions)
    visible_colors = _point_region_colors(region_ids, region_count)
    extent = max(float(np.ptp(template, axis=0).max()), 1e-4)
    offsets = _gallery_offsets(len(expanded_poses), columns, extent * spacing_scale)
    builder = GeometryBuilder()
    entries = []
    for index, (pose, offset) in enumerate(zip(expanded_poses, offsets)):
        builder.add_mesh(template + offset, faces, LIGHT_GRAY)
        builder.add_points(visible + offset, visible_colors)
        _add_axis_boundaries_and_origin(
            builder,
            template,
            sample["template"]["symmetry_metadata"],
            offset=offset,
            include_boundaries=False,
            include_reference=True,
        )
        entries.append(
            {
                "index": index,
                "gallery_offset_m": offset.tolist(),
                "T_C_from_O": pose.detach().cpu().tolist(),
            }
        )
    builder.write(path, comments=comments)
    return {
        "template_copy_count": len(expanded_poses),
        "observed_visible_points_copy_count": len(expanded_poses),
        "points_per_copy": len(visible),
        "uses_full_fragment_mesh": False,
        "hypotheses": entries,
    }


def _ensure_gt_reference(
    reference_dir: Path,
    sample: Mapping[str, Any],
    fragment_source: _FragmentProjectionSource,
    shell_surface_C: _TriangleSurfaceIndex,
    *,
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Write immutable GT artifacts once and return their stable paths."""

    reference_dir.mkdir(parents=True, exist_ok=True)
    footprint_path = reference_dir / "gt_fragment_regions_on_template.ply"
    gallery_path = reference_dir / "gt_hypotheses_gallery.ply"
    summary_path = reference_dir / "gt_summary.json"
    if not summary_path.exists():
        metadata = sample["template"]["symmetry_metadata"]
        gt_pose = sample["gt"]["T_C_from_O"]
        group = sample["gt"]["effective_symmetry_group"]
        expanded = visualization_equivalent_pose_set(
            gt_pose,
            metadata,
            effective_group=group,
            so2_visualization_samples=int(config.get("so2_visualization_samples", 12)),
        )
        footprint = _fragment_regions_on_template(
            footprint_path,
            sample,
            gt_pose,
            shell_surface_C,
            footprint_color=GT_ORANGE,
            boundary_resolution_m=float(
                config.get("template_boundary_resolution_m", 1e-4)
            ),
            boundary_max_depth=int(config.get("template_boundary_max_depth", 2)),
            comments=("gt_reference=true", "written_once=true"),
        )
        gallery = _predicted_gallery(
            gallery_path,
            sample,
            expanded.poses,
            metadata,
            shell_surface_C,
            columns=int(config.get("gallery_columns", 4)),
            spacing_scale=float(config.get("gallery_spacing_scale", 1.5)),
            boundary_resolution_m=float(
                config.get("template_boundary_resolution_m", 1e-4)
            ),
            boundary_max_depth=int(config.get("template_boundary_max_depth", 2)),
            show_progress=False,
        )
        _write_json(
            summary_path,
            {
                **WARNING_FLAGS,
                "sample_id": sample["sample_id"],
                "written_once": True,
                "gt_effective_group": group,
                "num_hypotheses": expanded.num_hypotheses,
                "fragment_footprint": footprint,
                "gallery": gallery,
                "source_fragment_mesh": str(fragment_source.mesh_path),
            },
        )
    return {
        "gt_fragment_regions_on_template": str(footprint_path),
        "gt_hypotheses_gallery": str(gallery_path),
        "gt_summary": str(summary_path),
    }


def select_debug_samples(
    dataset: Any,
    dataset_indices: Sequence[int],
    *,
    count: int,
    seed: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    buckets: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    all_candidates: list[tuple[int, Mapping[str, Any]]] = []
    for index in dataset_indices:
        sample = dataset[int(index)]
        all_candidates.append((int(index), sample))
        group = sample["gt"].get("effective_symmetry_group")
        key = json.dumps(group, sort_keys=True) if group is not None else "none"
        buckets.setdefault(key, []).append((int(index), sample))
    for key, values in buckets.items():
        values.sort(
            key=lambda item: (
                hashlib.sha256(
                    f"{seed}:{item[1]['sample_id']}".encode("utf-8")
                ).hexdigest(),
                item[1]["meta"]["num_observed_points_raw"],
            )
        )
    selected: list[tuple[int, Mapping[str, Any]]] = []
    # Guarantee several scenes before balancing the remaining slots by group.
    scenes = sorted(
        {str(sample["scene_id"]) for _, sample in all_candidates},
        key=lambda scene: hashlib.sha256(f"{seed}:{scene}".encode("utf-8")).hexdigest(),
    )
    for scene in scenes[: min(3, count)]:
        candidates = [
            item for item in all_candidates if str(item[1]["scene_id"]) == scene
        ]
        candidates.sort(
            key=lambda item: hashlib.sha256(
                f"{seed}:{item[1]['sample_id']}".encode("utf-8")
            ).hexdigest()
        )
        chosen = candidates[0]
        selected.append(chosen)
        for values in buckets.values():
            values[:] = [item for item in values if item[0] != chosen[0]]
    keys = sorted(buckets)
    while len(selected) < count and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(0))
    indices = [item[0] for item in selected]
    entries = [
        {
            "dataset_index": index,
            "sample_id": sample["sample_id"],
            "scene_id": sample["scene_id"],
            "fragment_id": sample["fragment_id"],
            "num_observed_points": len(sample["observed"]["points_C"]),
            "num_observed_points_raw": sample["meta"]["num_observed_points_raw"],
            "effective_symmetry_group": sample["gt"].get("effective_symmetry_group"),
            "fragment_num_faces": sample["meta"]["fragment_mesh"]["num_faces"],
        }
        for index, sample in selected
    ]
    return indices, entries


def joint_visualization_epochs(max_epochs: int = 1500, interval: int = 250) -> list[int]:
    """Deterministic epoch schedule, including both endpoints."""
    if interval <= 0 or max_epochs < 0:
        raise ValueError("visualization schedule requires non-negative epochs and positive interval")
    values = list(range(0, max_epochs + 1, interval))
    if values[-1] != max_epochs:
        values.append(max_epochs)
    return values


def _joint_sample_directory(destination: Path, sample: Mapping[str, Any], config: Mapping[str, Any]) -> Path:
    """Return a collision-free directory for single- or multi-fragment layouts."""
    frame = f"frame_{int(sample['frame_id']):06d}"
    if bool(config.get("multifragment_layout", False)):
        return destination / "per_view" / frame / f"fragment_{int(sample['fragment_id']):04d}"
    return destination / "per_view" / frame


@torch.no_grad()
def _export_joint_prediction_visualizations(
    model: torch.nn.Module,
    dataset: Any,
    dataset_indices: Sequence[int],
    collate: Any,
    device: torch.device,
    *,
    epoch: int,
    output_dir: str | Path,
    config: Mapping[str, Any],
) -> list[str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    was_training = model.training
    model.eval()
    written: list[str] = []
    world_pose_rows: list[tuple[int, Tensor, Tensor]] = []
    joint_config = config.get("joint_surface_correspondence_pose_v3", {})
    if not bool(joint_config.get("enabled", False)):
        joint_config = config.get("joint_correspondence_pose", {})
    for dataset_index in dataset_indices:
        sample = dataset[int(dataset_index)]
        sample_dir = _joint_sample_directory(destination, sample, config)
        sample_dir.mkdir(parents=True, exist_ok=False)
        batch = move_to_device(collate([sample]), device)
        prediction = model(batch)
        row = batch_pose_metric_rows(
            prediction, batch, joint_loss_config=joint_config
        )[0]
        pose = prediction.correspondence_pose[0]
        valid = prediction.observed_valid_mask[0]
        q_pred = prediction.correspondence_points_O[0, valid]
        p_observed = batch["observed"].to_padded()["points"][0, valid]
        q_gt_payload = batch["gt"]["points_O_corresponding"]
        q_gt_all = q_gt_payload.to_padded()["points"] if hasattr(q_gt_payload, "to_padded") else q_gt_payload
        q_gt = q_gt_all[0, valid]
        gt_pose = batch["gt"]["T_C_from_O"][0]
        metadata = batch["template_symmetry_metadata"][0]
        group = batch["gt"]["effective_symmetry_group"][0]
        expanded_gt = visualization_equivalent_pose_set(
            gt_pose, metadata, effective_group=group,
            so2_visualization_samples=int(config.get("so2_visualization_samples", 12)),
        )
        selected = int(row["selected_shared_symmetry_element"])
        selected_pose = expanded_gt.poses[selected]
        symmetry = invert_transform(gt_pose) @ selected_pose
        matched_q_gt = transform_points(invert_transform(symmetry), q_gt)

        auxiliary = getattr(prediction, "correspondence_auxiliary", None)
        legacy_auxiliary = (
            auxiliary if auxiliary is not None and "patch_points_O" in auxiliary else None
        )
        coordinate_primary = bool(config.get("coordinate_guided_primary", False))
        q_aux = None
        q_k16 = None
        k16_recall = None
        if coordinate_primary:
            if auxiliary is None or "fine_aux_coordinate_normalized" not in auxiliary:
                raise ValueError("coordinate-guided visualization requires q_aux")
            mesh_vertices = batch["template_mesh_vertices_O"][0].to(device)
            mesh_faces = batch["template_mesh_faces"][0].to(device=device, dtype=torch.long)
            qn = auxiliary["fine_aux_coordinate_normalized"][0, valid]
            extent = (mesh_vertices.amax(0) - mesh_vertices.amin(0)).clamp_min(1e-8)
            q_aux = .5 * (qn + 1.0) * extent + mesh_vertices.amin(0)
            exact = closest_points_on_triangle_mesh(
                q_aux, mesh_vertices, mesh_faces, point_chunk_size=256
            )
            q_pred = exact["points"]
            built_k16 = AuxGuidedTriangleCandidateBuilder(
                mode="aux_guided_global_topk", candidate_k=16,
                projection_chunk_size=256,
            )(
                q_aux[None], [mesh_vertices], [mesh_faces],
                torch.ones((1, len(q_aux)), dtype=torch.bool, device=device),
            )
            projected_k16 = CoordinateGuidedSurfaceProjectionHead()(
                q_aux[None], built_k16["candidate_triangle_ids"],
                [mesh_vertices], [mesh_faces],
                torch.ones((1, len(q_aux)), dtype=torch.bool, device=device),
                built_k16["candidate_triangle_mask"],
            )
            q_k16 = projected_k16["surface_correspondence_points_O"][0]
            k16_recall = float(
                (
                    built_k16["candidate_triangle_ids"][0]
                    == exact["face_ids"][:, None]
                ).any(-1).float().mean()
            )
            solution = model.weighted_procrustes.solve(
                q_pred[None], p_observed[None], q_pred.new_ones((1, len(q_pred))),
                torch.ones((1, len(q_pred)), dtype=torch.bool, device=device),
            )
            pose = solution["transform"][0]

        camera_path = sample_dir / "pred_vs_gt_camera_frame.ply"
        _camera_frame_ply(camera_path, sample, pose, include_gt=True)

        template_points = sample["template"]["points_O"].detach().cpu().numpy()
        template_faces = sample["template"]["faces"].detach().cpu().numpy().astype(np.int64)
        gt_np = matched_q_gt.detach().cpu().numpy()
        pred_np = q_pred.detach().cpu().numpy()
        if coordinate_primary and q_aux is not None:
            q_aux_np = q_aux.detach().cpu().numpy()
            global_path = sample_dir / "q_aux_vs_global_projection.ply"
            k16_path = sample_dir / "q_aux_vs_k16_projection.ply"
            colors = np.concatenate((
                np.broadcast_to(np.asarray([255, 0, 255], dtype=np.uint8), (len(q_aux_np), 3)),
                np.broadcast_to(OBSERVED_GREEN, (len(pred_np), 3)),
            ))
            points = np.concatenate((q_aux_np, pred_np))
            write_colored_ply(global_path, points, colors, comments=("q_aux=magenta; exact_global=green",))
            q_k16_np = q_k16.detach().cpu().numpy()
            k16_points = np.concatenate((q_aux_np, q_k16_np))
            write_colored_ply(
                k16_path, k16_points, colors,
                comments=(
                    "q_aux=magenta; aux_guided_global_k16=green",
                    f"global_triangle_recall={k16_recall:.9f}",
                ),
            )
            written.extend((str(global_path), str(k16_path)))
        corr_error = torch.linalg.vector_norm(q_pred - matched_q_gt, dim=-1)
        pred_colors = np.broadcast_to(REFERENCE_RED, (len(pred_np), 3)).copy()
        pred_colors[corr_error.detach().cpu().numpy() > 0.002] = np.asarray([255, 0, 255], dtype=np.uint8)
        correspondence_path = sample_dir / "correspondences_on_template.ply"
        write_colored_ply(
            correspondence_path,
            np.concatenate((template_points, gt_np, pred_np)),
            np.concatenate((
                np.broadcast_to(LIGHT_GRAY, (len(template_points), 3)),
                np.broadcast_to(OBSERVED_GREEN, (len(gt_np), 3)), pred_colors,
            )),
            faces=template_faces,
            comments=("correspondence_lines=false", "magenta_threshold_mm=2", "shared_symmetry_element=true"),
        )

        logits = prediction.correspondence_logits[0, valid]
        attention = attention_distribution_metrics(logits)
        if legacy_auxiliary is None:
            template_padded = batch["template"].to_padded()
            anchor_ids, anchor_mask = farthest_point_indices(
                template_padded["points"], template_padded["valid_mask"], logits.shape[-1]
            )
            anchors = template_padded["points"][0, anchor_ids[0, anchor_mask[0]]]
        else:
            anchors = legacy_auxiliary["patch_points_O"][0]
        counts = attention["anchor_counts"][: len(anchors)].detach().cpu().numpy()
        frequency = counts / max(int(counts.max()), 1)
        anchor_colors = np.stack(
            (255 * frequency, 150 * (1 - frequency), 255 * (1 - frequency)), axis=1
        ).astype(np.uint8)
        collision_cutoff = max(2, int(np.ceil(len(q_pred) * 0.05)))
        anchor_colors[counts >= collision_cutoff] = np.asarray([255, 0, 0], dtype=np.uint8)
        usage_path = sample_dir / "attention_anchor_usage.ply"
        write_colored_ply(
            usage_path,
            np.concatenate((template_points, anchors.detach().cpu().numpy(), pred_np)),
            np.concatenate((
                np.broadcast_to(LIGHT_GRAY, (len(template_points), 3)),
                anchor_colors,
                pred_colors,
            )),
            faces=template_faces,
            comments=(
                "template=gray; anchor_frequency=cyan_to_red",
                f"high_collision_anchor_threshold={collision_cutoff}",
                "row_error_above_2mm=magenta",
            ),
        )
        extra_paths = [usage_path]
        if legacy_auxiliary is not None:
            patch_points = legacy_auxiliary["patch_points_O"][0]
            patch_ids = legacy_auxiliary["selected_patch_ids"][0, valid]
            patch_counts = torch.bincount(patch_ids, minlength=len(patch_points)).detach().cpu().numpy()
            patch_frequency = patch_counts / max(int(patch_counts.max()), 1)
            patch_colors = np.stack((255 * patch_frequency, 180 * (1 - patch_frequency), 40 * np.ones_like(patch_frequency)), axis=1).astype(np.uint8)
            patches_path = sample_dir / "selected_template_patches.ply"
            write_colored_ply(
                patches_path,
                np.concatenate((template_points, patch_points.detach().cpu().numpy())),
                np.concatenate((np.broadcast_to(LIGHT_GRAY, (len(template_points), 3)), patch_colors)),
                faces=template_faces,
                comments=("template=gray", "patch_centers=color_by_selected_frequency"),
            )
            triangle_ids = legacy_auxiliary["selected_triangle_ids"][0, valid].detach().cpu().numpy()
            selected_vertices = np.unique(template_faces[triangle_ids].reshape(-1))
            triangle_colors = np.broadcast_to(LIGHT_GRAY, (len(template_points), 3)).copy()
            triangle_colors[selected_vertices] = np.asarray([45, 120, 255], dtype=np.uint8)
            triangle_path = sample_dir / "predicted_triangle_correspondences.ply"
            write_colored_ply(
                triangle_path,
                np.concatenate((template_points, pred_np)),
                np.concatenate((triangle_colors, pred_colors)),
                faces=template_faces,
                comments=("selected_triangle_vertices=blue", "predicted_points=red_or_magenta"),
            )
            extra_paths.extend((patches_path, triangle_path))

            if "all_candidate_triangle_ids" in legacy_auxiliary:
                candidate_centroids = torch.as_tensor(
                    template_points, device=matched_q_gt.device
                )[torch.as_tensor(template_faces, device=matched_q_gt.device)[
                    legacy_auxiliary["all_candidate_triangle_ids"][0]
                ]].mean(-2)
                gt_patch_ids = torch.cdist(
                    matched_q_gt.float(), candidate_centroids.reshape(-1, 3).float()
                ).reshape(len(matched_q_gt), len(patch_points), -1).amin(-1).argmin(-1)
            else:
                gt_patch_ids = torch.cdist(
                    matched_q_gt.float(), patch_points.float()
                ).argmin(-1)
            predicted_patch_ids = legacy_auxiliary["coarse_patch_logits"][0, valid].argmax(-1)
            coarse_colors = np.broadcast_to(LIGHT_GRAY, (len(template_points), 3)).copy()
            predicted_patch_vertices = torch.cdist(
                torch.as_tensor(template_points, device=patch_points.device).float(),
                patch_points[torch.unique(predicted_patch_ids)].float(),
            ).amin(-1).lt(0.002).detach().cpu().numpy()
            coarse_colors[predicted_patch_vertices] = np.asarray([255, 90, 40], dtype=np.uint8)
            coarse_path = sample_dir / "coarse_patch_prediction.ply"
            write_colored_ply(coarse_path, template_points, coarse_colors, faces=template_faces,
                              comments=("predicted_coarse_patches=orange",))

            comparison_colors = np.broadcast_to(LIGHT_GRAY, (len(patch_points), 3)).copy()
            comparison_colors[torch.unique(gt_patch_ids).detach().cpu().numpy()] = np.asarray([45, 210, 70], dtype=np.uint8)
            comparison_colors[torch.unique(predicted_patch_ids).detach().cpu().numpy()] = np.asarray([255, 90, 40], dtype=np.uint8)
            both = set(torch.unique(gt_patch_ids).tolist()) & set(torch.unique(predicted_patch_ids).tolist())
            if both:
                comparison_colors[np.asarray(sorted(both), dtype=np.int64)] = np.asarray([255, 220, 0], dtype=np.uint8)
            comparison_path = sample_dir / "GT_patch_vs_predicted_patch.ply"
            write_colored_ply(comparison_path, patch_points.detach().cpu().numpy(), comparison_colors,
                              comments=("GT=green; predicted=orange; overlap=yellow",))

            candidate_ids = legacy_auxiliary["candidate_triangle_ids"][0, valid]
            candidate_vertices = np.unique(template_faces[candidate_ids.detach().cpu().numpy()].reshape(-1))
            candidate_colors = np.broadcast_to(LIGHT_GRAY, (len(template_points), 3)).copy()
            candidate_colors[candidate_vertices] = np.asarray([80, 170, 255], dtype=np.uint8)
            candidate_path = sample_dir / "local_candidate_triangles.ply"
            write_colored_ply(candidate_path, template_points, candidate_colors, faces=template_faces,
                              comments=("local_candidate_triangle_vertices=blue",))

            gt_triangle = closest_points_on_triangle_mesh(
                matched_q_gt, torch.as_tensor(template_points, device=matched_q_gt.device),
                torch.as_tensor(template_faces, device=matched_q_gt.device),
            )["face_ids"]
            gt_in_candidates = candidate_ids.eq(gt_triangle[:, None]).any(-1)
            gt_vertices = np.unique(template_faces[gt_triangle.detach().cpu().numpy()].reshape(-1))
            gt_triangle_colors = np.broadcast_to(LIGHT_GRAY, (len(template_points), 3)).copy()
            gt_triangle_colors[gt_vertices] = np.asarray(
                [45, 210, 70] if bool(gt_in_candidates.all()) else [255, 0, 255],
                dtype=np.uint8,
            )
            gt_triangle_path = sample_dir / "GT_triangle_in_candidate_set.ply"
            write_colored_ply(gt_triangle_path, template_points, gt_triangle_colors, faces=template_faces,
                              comments=(f"candidate_recall={float(gt_in_candidates.float().mean()):.6f}",))

            centered = q_pred - q_pred.mean(0)
            covariance = centered.T @ centered / max(len(q_pred) - 1, 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            center = q_pred.mean(0)
            axes = [center]
            axis_colors = [np.asarray([255, 255, 255], dtype=np.uint8)]
            palette = ([255, 0, 0], [0, 255, 0], [0, 100, 255])
            for axis_index in range(3):
                scale = eigenvalues[axis_index].clamp_min(0).sqrt()
                direction = eigenvectors[:, axis_index] * scale
                axes.extend((center - direction, center + direction))
                axis_colors.extend((np.asarray(palette[axis_index], dtype=np.uint8),) * 2)
            covariance_path = sample_dir / "predicted_covariance_axes.ply"
            write_colored_ply(covariance_path, torch.stack(axes), np.stack(axis_colors),
                              edges=np.asarray([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]], dtype=np.int64),
                              comments=("covariance_principal_axes=endpoints",))
            extra_paths.extend((coarse_path, comparison_path, candidate_path, gt_triangle_path, covariance_path))

        reconstructed = transform_points(pose, q_pred)
        if coordinate_primary and sample.get("gt", {}).get("T_W_from_C") is not None:
            T_W_from_C = torch.as_tensor(sample["gt"]["T_W_from_C"], device=pose.device, dtype=pose.dtype)
            world_pose_rows.append((int(sample["frame_id"]), T_W_from_C @ pose, q_pred.detach()))
        reconstructed_path = sample_dir / "reconstructed_visible_shell.ply"
        write_colored_ply(
            reconstructed_path,
            torch.cat((p_observed, reconstructed)),
            np.concatenate((
                np.broadcast_to(OBSERVED_GREEN, (len(p_observed), 3)),
                np.broadcast_to(REFERENCE_RED, (len(reconstructed), 3)),
            )),
            comments=("observed=green", "reconstructed_from_predicted_correspondences=red"),
        )

        gallery_path = sample_dir / "symmetry_hypotheses_gallery.ply"
        expanded = visualization_equivalent_pose_set(
            pose, metadata, effective_group=group,
            so2_visualization_samples=int(config.get("so2_visualization_samples", 12)),
        )
        visible_by_pose = place_fragment_for_hypothesis(q_pred, pose, expanded.poses)
        extent = max(float(np.ptp(template_points, axis=0).max()), 1e-4)
        offsets = _gallery_offsets(len(expanded.poses), int(config.get("gallery_columns", 4)), extent * float(config.get("gallery_spacing_scale", 1.5)))
        builder = GeometryBuilder()
        for hypothesis_points, offset in zip(visible_by_pose.detach().cpu().numpy(), offsets):
            builder.add_mesh(template_points + offset, template_faces, LIGHT_GRAY)
            builder.add_points(hypothesis_points + offset, OBSERVED_GREEN)
        builder.write(gallery_path, comments=(
            "group_source=gt_training_symmetry_target",
            f"effective_group={group}",
            "production_symmetry_expander=true",
        ))

        mesh_source = Path(str(sample["meta"]["fragment_mesh"]["mesh_path"])).resolve()
        mesh_copy = sample_dir / "source_fragment_mesh.ply"
        shutil.copyfile(mesh_source, mesh_copy)
        summary_path = sample_dir / "visualization_summary.json"
        _write_json(summary_path, {
            **WARNING_FLAGS, "epoch": epoch, "sample_id": sample["sample_id"],
            "selected_shared_symmetry_element": selected,
            "rotation_error_deg": row["rotation_error_deg"],
            "translation_total_mm": row["translation_total_mm"],
            "correspondence_p95_mm": row["correspondence_p95_mm"],
            "visible_alignment_p95_mm": row["visible_alignment_p95_mm"],
            "predicted_to_template_surface_p95_mm": row["predicted_to_template_surface_p95_mm"],
            "group_source": "gt_training_symmetry_target",
            "primary_pose_source": (
                "active/exact_global/uniform_procrustes"
                if coordinate_primary else "legacy_correspondence"
            ),
            "inactive_legacy_pose_is_primary": False if coordinate_primary else None,
            "k16_exact_global_triangle_recall": k16_recall,
            "predicted_q_bbox_m": [pred_np.min(0).tolist(), pred_np.max(0).tolist()],
            "gt_q_bbox_m": [gt_np.min(0).tolist(), gt_np.max(0).tolist()],
            "unique_predicted_patches": row.get("unique_predicted_patches"),
            "unique_predicted_triangles": row.get("unique_predicted_triangles"),
            "most_popular_patch_fraction": row.get("most_popular_patch_fraction"),
            "ply_paths": {
                "pred_vs_gt_camera_frame": str(camera_path),
                "correspondences_on_template": str(correspondence_path),
                "reconstructed_visible_shell": str(reconstructed_path),
                "symmetry_hypotheses_gallery": str(gallery_path),
                "attention_anchor_usage": str(usage_path),
                "source_fragment_mesh": str(mesh_copy),
                **({
                    "q_aux_vs_global_projection": str(global_path),
                    "q_aux_vs_k16_projection": str(k16_path),
                } if coordinate_primary else {}),
                **({
                    "selected_template_patches": str(extra_paths[1]),
                    "predicted_triangle_correspondences": str(extra_paths[2]),
                } if legacy_auxiliary is not None else {}),
            },
        })
        written.extend(str(path) for path in (camera_path, correspondence_path, reconstructed_path, gallery_path, mesh_copy, summary_path, *extra_paths))
    if len(world_pose_rows) in {2, 4, 8, 10}:
        view_label = {
            2: "two_view", 4: "four_view", 8: "eight_view", 10: "ten_view"
        }[len(world_pose_rows)]
        world_path = destination / f"{view_label}_world_pose_comparison.ply"
        world_points = []
        world_colors = []
        palette = (
            np.asarray([40, 210, 80], dtype=np.uint8),
            np.asarray([255, 150, 35], dtype=np.uint8),
            np.asarray([170, 70, 230], dtype=np.uint8),
            np.asarray([70, 130, 255], dtype=np.uint8),
            np.asarray([230, 60, 70], dtype=np.uint8),
            np.asarray([40, 205, 205], dtype=np.uint8),
            np.asarray([245, 215, 55], dtype=np.uint8),
            np.asarray([235, 90, 190], dtype=np.uint8),
            np.asarray([125, 215, 80], dtype=np.uint8),
            np.asarray([120, 100, 245], dtype=np.uint8),
        )
        for index, (_, world_pose, points_O) in enumerate(world_pose_rows):
            placed = transform_points(world_pose[None], points_O[None])[0]
            world_points.append(placed.detach().cpu().numpy())
            world_colors.append(np.broadcast_to(palette[index], (len(placed), 3)))
        write_colored_ply(
            world_path, np.concatenate(world_points), np.concatenate(world_colors),
            comments=("active_pose_source=exact_global_uniform_procrustes",),
        )
        transforms = torch.stack([value[1] for value in world_pose_rows])
        consistency = world_pose_consistency(transforms, metadata, group)
        world_summary = destination / f"{view_label}_world_pose_summary.json"
        _write_json(
            world_summary,
            {
                **consistency,
                "active_pose_source": "exact_global_uniform_procrustes",
                "frame_ids": [value[0] for value in world_pose_rows],
                "loss_weight": 0.0,
            },
        )
        written.extend((str(world_path), str(world_summary)))
    if was_training:
        model.train()
    return written


@torch.no_grad()
def export_prediction_visualizations(
    model: torch.nn.Module,
    dataset: Any,
    dataset_indices: Sequence[int],
    collate: Any,
    device: torch.device,
    *,
    epoch: int,
    output_dir: str | Path,
    config: Mapping[str, Any],
) -> list[str]:
    if bool(getattr(model, "is_joint_uniform_correspondence_model", False)):
        return _export_joint_prediction_visualizations(
            model, dataset, dataset_indices, collate, device,
            epoch=epoch, output_dir=output_dir, config=config,
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    single_fragment_layout = bool(config.get("single_fragment_layout", False))
    default_reference_root = (
        destination / "reference"
        if single_fragment_layout
        else destination.parent / "reference"
    )
    reference_root = Path(
        str(config.get("reference_root", default_reference_root))
    )
    reference_root.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    written: list[str] = []
    threshold = float(config.get("learned_active_region_threshold", 0.5))
    topk_count = int(config.get("debug_num_base_queries", 3))
    projection_distance_m = float(config.get("template_projection_distance_m", 5e-4))
    boundary_resolution_m = float(config.get("template_boundary_resolution_m", 1e-4))
    boundary_max_depth = int(config.get("template_boundary_max_depth", 2))
    show_progress = bool(config.get("progress_bar", True))
    activity = dict(config.get("symmetry_region_activity", {}))
    activity.setdefault(
        "unresolved_group_policy", config.get("unresolved_group_policy", "base_pose_only")
    )
    activity.setdefault("so2_visualization_samples", config.get("so2_visualization_samples", 12))
    resolver = PoseConditionedSymmetryResolver()
    cross_view_rows: list[tuple[dict[str, Any], Any]] = []
    sample_progress = tqdm(
        dataset_indices,
        desc=f"debug-vis epoch {epoch:04d}",
        unit="sample",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    for dataset_index in sample_progress:
        sample = dataset[int(dataset_index)]
        sample_id = str(sample["sample_id"])
        safe_id = sample_id.replace("/", "__")
        sample_progress.set_postfix_str(sample_id, refresh=False)
        sample_dir = (
            destination / "per_view" / f"frame_{int(sample['frame_id']):06d}"
            if single_fragment_layout
            else destination / safe_id
        )
        sample_dir.mkdir(parents=True, exist_ok=False)
        metadata = sample["template"]["symmetry_metadata"]
        region_count = len(metadata.regions)

        # This source is copied for inspection and is used only by explicitly
        # labelled oracle/reference artifacts below.
        fragment_source = _fragment_projection_source(sample)
        copied_fragment_path = sample_dir / fragment_source.mesh_path.name
        shutil.copyfile(fragment_source.mesh_path, copied_fragment_path)
        shell_surface_C = _TriangleSurfaceIndex(
            fragment_source.points_C[fragment_source.shell_faces],
            projection_distance_m,
        )
        reference_key = (
            f"fragment_{int(sample['fragment_id']):04d}"
            if single_fragment_layout
            else safe_id
        )
        reference_paths = _ensure_gt_reference(
            reference_root / reference_key,
            sample,
            fragment_source,
            shell_surface_C,
            config=config,
        )

        batch = move_to_device(collate([sample]), device)
        prediction = model(batch)
        metric_row = batch_pose_metric_rows(
            prediction,
            batch,
            ranking_config=config.get("pose_query_ranking"),
        )[0]
        scores = torch.softmax(prediction.pose_logits[0], dim=-1)
        top_indices = torch.argsort(scores, descending=True)[: max(topk_count, 1)]
        top_index = int(top_indices[0])
        base_pose = prediction.pose_hypotheses[0, top_index]
        observed_points_C = sample["observed"]["points_C"].to(device)
        observed_batch = observed_points_C.unsqueeze(0)
        observed_valid = torch.ones(
            observed_batch.shape[:2], dtype=torch.bool, device=device
        )
        resolution = resolver.resolve(
            observed_batch,
            observed_valid,
            prediction.pose_hypotheses,
            [metadata],
            activity,
        )
        point_region_ids = resolution.point_region_ids_per_pose[0][top_index]
        pose_active = resolution.active_regions_per_pose[0][top_index]
        pose_group = resolution.effective_group_per_pose[0][top_index]
        pose_status = resolution.effective_group_status_per_pose[0][top_index]
        pose_expanded = resolution.expanded_poses_per_base_pose[0][top_index]
        points_O_pred = transform_points(invert_transform(base_pose), observed_points_C)

        top1_path = sample_dir / "pred_top1_camera_frame.ply"
        _camera_frame_ply(top1_path, sample, base_pose, include_gt=False)
        oracle_index = int(metric_row["oracle_query_index"])
        oracle_pose = prediction.pose_hypotheses[0, oracle_index]
        oracle_camera_path = sample_dir / "oracle_best_camera_frame.ply"
        _camera_frame_ply(
            oracle_camera_path, sample, oracle_pose, include_gt=True
        )
        visible_path = sample_dir / "predicted_observed_points_regions_on_template.ply"
        visible_info = _visible_points_regions_on_template(
            visible_path,
            sample,
            points_O_pred,
            point_region_ids,
            comments=(
                "geometry_source=observed_visible_points_C",
                "full_fragment_mesh_used=false",
                "group_source=pose_conditioned_geometry",
            ),
        )

        oracle_path = sample_dir / "oracle_full_fragment_regions_on_template.ply"
        oracle_info = _fragment_regions_on_template(
            oracle_path,
            sample,
            base_pose,
            shell_surface_C,
            footprint_color=PREDICTED_BLUE,
            boundary_resolution_m=boundary_resolution_m,
            boundary_max_depth=boundary_max_depth,
            comments=(
                "oracle_debug_only=true",
                "unavailable_at_inference=true",
                "group_source=none",
            ),
        )
        oracle_info.update(
            {
                "oracle_debug_only": True,
                "unavailable_at_inference": True,
                "used_for_predicted_group": False,
            }
        )

        pose_gallery_path = (
            sample_dir / "predicted_hypotheses_gallery_pose_conditioned.ply"
        )
        pose_gallery = _visible_points_gallery(
            pose_gallery_path,
            sample,
            pose_expanded,
            points_O_pred,
            point_region_ids,
            columns=int(config.get("gallery_columns", 4)),
            spacing_scale=float(config.get("gallery_spacing_scale", 1.5)),
            comments=(
                "group_source=pose_conditioned_geometry",
                "geometry_source=observed_visible_points_C",
                "full_fragment_mesh_used=false",
            ),
        )
        gallery_alias_path = sample_dir / "predicted_hypotheses_gallery.ply"
        shutil.copyfile(pose_gallery_path, gallery_alias_path)

        learned_head_available = (
            prediction.active_region_logits is not None
            and prediction.observed_region_logits is not None
        )
        if learned_head_available:
            active_probabilities = torch.sigmoid(
                prediction.active_region_logits[0, :region_count]
            )
            learned_active = active_probabilities >= threshold
        else:
            active_probabilities = base_pose.new_zeros((region_count,))
            learned_active = torch.zeros(
                region_count, dtype=torch.bool, device=base_pose.device
            )
        if learned_head_available and bool(learned_active.any()):
            learned_group = effective_group_from_regions(metadata, learned_active)
            learned_status = "resolved"
            learned_expanded = visualization_equivalent_pose_set(
                base_pose,
                metadata,
                effective_group=learned_group,
                so2_visualization_samples=int(
                    config.get("so2_visualization_samples", 12)
                ),
            ).poses
        else:
            learned_group = None
            learned_status = (
                "unresolved" if learned_head_available else "head_disabled"
            )
            learned_expanded = base_pose.unsqueeze(0)
        if learned_head_available:
            learned_point_logits = prediction.observed_region_logits[
                0, : len(observed_points_C), :region_count
            ]
            learned_point_ids = learned_point_logits.argmax(dim=-1)
        else:
            learned_point_ids = torch.zeros(
                len(observed_points_C), dtype=torch.long, device=base_pose.device
            )
        learned_gallery_path = (
            sample_dir / "predicted_hypotheses_gallery_learned_regions.ply"
        )
        learned_gallery = _visible_points_gallery(
            learned_gallery_path,
            sample,
            learned_expanded,
            points_O_pred,
            learned_point_ids,
            columns=int(config.get("gallery_columns", 4)),
            spacing_scale=float(config.get("gallery_spacing_scale", 1.5)),
            comments=(
                "group_source=learned_active_region_head",
                "main_gallery=false",
                "geometry_source=observed_visible_points_C",
            ),
        )

        topk_path = sample_dir / "predicted_topK_base_poses.ply"
        _topk_base_gallery(
            topk_path,
            sample,
            prediction.pose_hypotheses[0, top_indices],
        )
        if single_fragment_layout:
            cross_view_rows.append((metric_row, metadata))

        gt_active_mask = sample["gt"]["active_symmetry_regions"].bool()
        gt_group = sample["gt"]["effective_symmetry_group"]
        gt_group_payload = group_to_dict(gt_group)
        pose_group_payload = group_to_dict(pose_group) if pose_group is not None else None
        learned_group_payload = (
            group_to_dict(learned_group) if learned_group is not None else None
        )
        region_names = [region.region_id for region in metadata.regions]
        active_names = lambda mask: [
            name for name, enabled in zip(region_names, mask.tolist()) if enabled
        ]
        warnings = list(resolution.diagnostics[0][top_index]["warnings"])
        if not learned_head_available:
            warnings.append("learned region head disabled in pose-only model")
        elif learned_group is None:
            warnings.append("learned active-region head unresolved; learned gallery uses base pose only")
        group_consistency = {
            "sample_id": sample_id,
            "epoch": epoch,
            "gt_active_regions": active_names(gt_active_mask),
            "gt_effective_group": gt_group_payload,
            "pose_conditioned_active_regions": active_names(pose_active),
            "pose_conditioned_effective_group": pose_group_payload,
            "pose_conditioned_effective_group_status": pose_status,
            "learned_region_probabilities": {
                name: float(probability)
                for name, probability in zip(region_names, active_probabilities)
            },
            "learned_active_regions_at_threshold": active_names(learned_active),
            "learned_effective_group": learned_group_payload,
            "learned_effective_group_status": learned_status,
            "learned_region_head_available": learned_head_available,
            "pose_conditioned_matches_gt": pose_group_payload == gt_group_payload,
            "learned_matches_gt": learned_group_payload == gt_group_payload,
            "out_of_bounds_fraction": resolution.out_of_sidecar_bounds_fraction[0][top_index],
            "warnings": warnings,
        }
        group_consistency_path = sample_dir / "group_consistency.json"
        _write_json(group_consistency_path, group_consistency)

        summary = {
            **WARNING_FLAGS,
            "sample_id": sample_id,
            "epoch": epoch,
            "reference": reference_paths,
            "num_observed_points": len(observed_points_C),
            "top1_base_query_index": top_index,
            "top1_base_query_score": float(scores[top_index]),
            "topK_base_query_indices": top_indices.detach().cpu().tolist(),
            "topK_base_query_scores": scores[top_indices].detach().cpu().tolist(),
            "main_group_source": "pose_conditioned_geometry",
            "main_geometry_source": "observed_visible_points_C",
            "pose_conditioned_effective_group": pose_group_payload,
            "pose_conditioned_effective_group_status": pose_status,
            "pose_conditioned_active_regions": pose_active.detach().cpu().tolist(),
            "num_pose_conditioned_hypotheses": len(pose_expanded),
            "learned_group_source": "learned_active_region_head",
            "learned_region_head_available": learned_head_available,
            "learned_effective_group": learned_group_payload,
            "num_learned_hypotheses": len(learned_expanded),
            "visible_geometry": visible_info,
            "oracle_full_fragment_geometry": oracle_info,
            "pose_conditioned_gallery": pose_gallery,
            "learned_region_gallery": learned_gallery,
            "base_pose_T_C_from_O": base_pose.detach().cpu().tolist(),
            "pose_metrics": metric_row,
            "prediction_generation_contract": {
                "training_pose_group_source": "GT effective group only",
                "main_predicted_group_source": "production pose-conditioned resolver",
                "main_predicted_geometry_source": "observed visible points only",
                "learned_head_is_auxiliary": learned_head_available,
                "full_fragment_mesh_used_for_main_group": False,
                "full_fragment_mesh_used_for_query_score": False,
                "full_fragment_mesh_is_oracle_debug_only": True,
            },
            "ply_paths": {
                "source_fragment_mesh_copy": str(copied_fragment_path),
                "pred_top1_camera_frame": str(top1_path),
                "oracle_best_camera_frame": str(oracle_camera_path),
                "predicted_observed_points_regions_on_template": str(visible_path),
                "oracle_full_fragment_regions_on_template": str(oracle_path),
                "predicted_hypotheses_gallery_pose_conditioned": str(
                    pose_gallery_path
                ),
                "predicted_hypotheses_gallery": str(gallery_alias_path),
                "predicted_hypotheses_gallery_learned_regions": str(
                    learned_gallery_path
                ),
                "predicted_topK_base_poses": str(topk_path),
            },
        }
        summary_path = sample_dir / "prediction_summary.json"
        _write_json(summary_path, summary)
        written.extend(
            str(path)
            for path in (
                copied_fragment_path,
                top1_path,
                oracle_camera_path,
                visible_path,
                oracle_path,
                pose_gallery_path,
                gallery_alias_path,
                learned_gallery_path,
                topk_path,
                summary_path,
                group_consistency_path,
            )
        )
        # Boundary-refined template meshes and triangle indices are relatively
        # large.  Ten-view debug must not retain previous-view CPU/GPU storage.
        del (
            batch,
            prediction,
            resolution,
            fragment_source,
            shell_surface_C,
            observed_batch,
            observed_valid,
            observed_points_C,
            points_O_pred,
            pose_expanded,
            learned_expanded,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if single_fragment_layout and cross_view_rows:
        cross_view_dir = destination / "cross_view"
        cross_view_dir.mkdir(parents=True, exist_ok=False)
        metadata = cross_view_rows[0][1]
        group = cross_view_rows[0][0].get("effective_symmetry_group")
        top1_world = torch.as_tensor(
            [row["top1_T_W_from_O"] for row, _ in cross_view_rows],
            dtype=torch.float64,
        )
        oracle_world = torch.as_tensor(
            [row["oracle_T_W_from_O"] for row, _ in cross_view_rows],
            dtype=torch.float64,
        )
        consistency = {
            **WARNING_FLAGS,
            "num_views": len(cross_view_rows),
            "top1": world_pose_consistency(top1_world, metadata, group),
            "oracle": world_pose_consistency(oracle_world, metadata, group),
        }
        consistency_path = cross_view_dir / "cross_view_consistency.json"
        _write_json(consistency_path, consistency)
        axis_O = torch.as_tensor(metadata.axis.direction, dtype=torch.float64)
        vertices: list[list[float]] = []
        colors: list[list[int]] = []
        edges: list[list[int]] = []
        for transforms, color in (
            (top1_world, PREDICTED_BLUE),
            (oracle_world, GT_ORANGE),
        ):
            for transform in transforms:
                center = transform[:3, 3]
                axis = transform[:3, :3] @ axis_O
                start = len(vertices)
                vertices.extend(
                    [center.tolist(), (center + 0.03 * axis).tolist()]
                )
                colors.extend([color.tolist(), color.tolist()])
                edges.append([start, start + 1])
        world_ply = write_colored_ply(
            cross_view_dir / "predicted_world_centers_and_axes.ply",
            vertices,
            colors,
            edges=edges,
            comments=(
                "blue=top1 world centers and axes",
                "orange=oracle world centers and axes",
            ),
        )
        written.extend([str(consistency_path), str(world_ply)])
    if was_training:
        model.train()
    return written


__all__ = [
    "export_prediction_visualizations",
    "select_debug_samples",
]
