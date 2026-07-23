"""Reusable diagnostics for direct-pose and progressive-view experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from symm_template_reg.engine.single_fragment import manifest_content_sha256
from symm_template_reg.models.pose.metrics import (
    rotation_error_deg,
    symmetry_aware_pose_errors,
)
from symm_template_reg.models.pose.pose_codec import PoseCodec


def subset_view_manifest(
    source: Mapping[str, Any], frame_ids: Sequence[int]
) -> dict[str, Any]:
    """Return a content-addressable single-fragment manifest subset."""

    requested = tuple(int(value) for value in frame_ids)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("frame_ids must be non-empty and unique")
    by_frame = {int(sample["frame_id"]): sample for sample in source["samples"]}
    missing = sorted(set(requested) - set(by_frame))
    if missing:
        raise ValueError(f"source manifest does not contain frames {missing}")
    payload = deepcopy(dict(source))
    samples = [deepcopy(by_frame[frame]) for frame in requested]
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    payload.update(
        {
            "manifest_type": "single_fragment_overfit",
            "accepted_observations": len(samples),
            "train_sample_ids": sample_ids,
            "validation_sample_ids": sample_ids,
            "samples": samples,
            "view_ladder": {
                "source_manifest_sha256": source.get("manifest_sha256"),
                "frame_ids": list(requested),
                "deterministic_point_order": True,
                "augmentation_enabled": False,
            },
        }
    )
    payload["manifest_sha256"] = manifest_content_sha256(payload)
    return payload


def assignment_switch_rate(
    assignments: Sequence[Mapping[int, int]],
) -> tuple[float, int, int]:
    """Fraction of shared frames whose oracle query changes at adjacent evals."""

    changes = 0
    comparisons = 0
    for previous, current in zip(assignments, assignments[1:]):
        for frame in sorted(set(previous) & set(current)):
            comparisons += 1
            changes += int(int(previous[frame]) != int(current[frame]))
    return changes / comparisons if comparisons else 0.0, changes, comparisons


def query_assignment_summary(
    rows: Sequence[Mapping[str, Any]], *, num_queries: int | None = None
) -> dict[str, Any]:
    """Validate and summarize one frame-by-query cost matrix."""

    if not rows:
        raise ValueError("query assignment rows must not be empty")
    costs = [list(map(float, row["query_pose_costs"])) for row in rows]
    inferred = len(costs[0])
    query_count = inferred if num_queries is None else int(num_queries)
    if query_count < 1 or any(len(value) != query_count for value in costs):
        raise ValueError("query cost matrix has inconsistent width")
    occupancy = {str(index): 0 for index in range(query_count)}
    frames = {str(index): [] for index in range(query_count)}
    successes: dict[str, list[int]] = {str(index): [] for index in range(query_count)}
    for row, row_costs in zip(rows, costs):
        frame = int(row["frame_id"])
        assigned = int(row.get("oracle_query_index", min(range(query_count), key=row_costs.__getitem__)))
        occupancy[str(assigned)] += 1
        frames[str(assigned)].append(frame)
        rotations = list(map(float, row.get("query_rotation_error_deg", [])))
        translations = list(map(float, row.get("query_translation_error_mm", [])))
        if rotations and translations:
            for index, (rotation, translation) in enumerate(zip(rotations, translations)):
                if rotation < 2.0 and translation < 2.0:
                    successes[str(index)].append(frame)
    return {
        "num_frames": len(rows),
        "num_queries": query_count,
        "query_occupancy": occupancy,
        "assigned_frames": frames,
        "query_success_frames_2deg_2mm": successes,
        "matrix_shape": [len(rows), query_count],
    }


def query_world_consistency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-query world-center and symmetry-axis spread across available views."""

    if not rows or "query_T_W_from_O" not in rows[0]:
        return {}
    query_count = len(rows[0]["query_T_W_from_O"])
    result: dict[str, Any] = {}
    for query in range(query_count):
        transforms = torch.as_tensor(
            [row["query_T_W_from_O"][query] for row in rows], dtype=torch.float64
        )
        centers = transforms[:, :3, 3]
        center = centers.mean(dim=0)
        distances = torch.cdist(centers, centers)
        axes = []
        for row, transform in zip(rows, transforms):
            axis = torch.as_tensor(row["symmetry_axis_O"], dtype=torch.float64)
            axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
            axes.append(transform[:3, :3] @ axis)
        axis_tensor = torch.stack(axes)
        # A symmetry axis is an unoriented line: +axis and -axis describe the
        # same physical axis and must therefore have zero angular spread.
        dots = torch.abs(axis_tensor @ axis_tensor.T).clamp(-1.0, 1.0)
        axis_spread = torch.rad2deg(torch.acos(dots)).max()
        result[str(query)] = {
            "world_translation_center_std_mm": float(
                torch.sqrt(((centers - center) ** 2).sum(-1).mean()) * 1000.0
            ),
            "world_translation_range_mm": float(distances.max() * 1000.0),
            "world_axis_spread_deg": float(axis_spread),
        }
    return result


def pose_context_change(
    original_poses: Tensor,
    shuffled_poses: Tensor,
    original_normalized: Tensor,
    shuffled_normalized: Tensor,
    *,
    rotation_threshold_deg: float = 0.01,
    normalized_translation_threshold: float = 1e-3,
) -> dict[str, Any]:
    """Measure whether fixed query identities react to swapped observations."""

    if original_poses.shape != shuffled_poses.shape:
        raise ValueError("original and shuffled pose shapes differ")
    if original_normalized.shape != shuffled_normalized.shape:
        raise ValueError("normalized pose parameter shapes differ")
    rotation = rotation_error_deg(original_poses, shuffled_poses)
    translation_delta = torch.linalg.vector_norm(
        original_normalized[..., 6:9] - shuffled_normalized[..., 6:9], dim=-1
    )
    rotation_max = float(rotation.detach().max())
    translation_max = float(translation_delta.detach().max())
    ignores = (
        rotation_max < float(rotation_threshold_deg)
        and translation_max < float(normalized_translation_threshold)
    )
    return {
        "mean_query_rotation_change_deg": float(rotation.detach().mean()),
        "max_query_rotation_change_deg": rotation_max,
        "mean_normalized_translation_change": float(
            translation_delta.detach().mean()
        ),
        "max_normalized_translation_change": translation_max,
        "diagnosis": (
            "pose_queries_ignore_observed_context"
            if ignores
            else "pose_queries_respond_to_observed_context"
        ),
    }


def direct_optimize_pose_parameters(
    *,
    gt_pose: Tensor,
    observed_points_C: Tensor,
    symmetry_metadata: Any,
    effective_group: Any,
    num_starts: int = 16,
    steps: int = 3000,
    learning_rate: float = 0.05,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> list[dict[str, Any]]:
    """Optimize only raw 6D rotation and normalized translation parameters."""

    if num_starts < 1 or steps < 1:
        raise ValueError("num_starts and steps must be positive")
    target_device = torch.device(device)
    dtype = torch.float64
    points = observed_points_C.to(device=target_device, dtype=dtype)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("observed_points_C must have shape [N,3]")
    codec = PoseCodec().to(device=target_device, dtype=dtype)
    context = codec.context(
        points.unsqueeze(0),
        torch.ones((1, len(points)), dtype=torch.bool, device=target_device),
    )
    generator = torch.Generator(device=target_device).manual_seed(int(seed))
    rotation_6d = torch.nn.Parameter(
        torch.randn(
            (num_starts, 6), generator=generator, dtype=dtype, device=target_device
        )
    )
    translation_normalized = torch.nn.Parameter(
        torch.randn(
            (num_starts, 3), generator=generator, dtype=dtype, device=target_device
        )
    )
    optimizer = torch.optim.Adam(
        (rotation_6d, translation_normalized), lr=float(learning_rate)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(steps), eta_min=1e-7
    )
    target = gt_pose.to(device=target_device, dtype=dtype).unsqueeze(0).expand(
        num_starts, -1, -1
    )
    centroid = context.observed_centroid_C.expand(num_starts, -1)
    scale = context.observed_scale.expand(num_starts)
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        poses = codec.decode_transform(
            rotation_6d, translation_normalized, centroid, scale
        )
        errors = symmetry_aware_pose_errors(
            poses,
            target,
            symmetry_metadata,
            effective_group=effective_group,
            translation_weight=10.0,
        )
        errors["combined"].mean().backward()
        optimizer.step()
        scheduler.step()
    with torch.no_grad():
        poses = codec.decode_transform(
            rotation_6d, translation_normalized, centroid, scale
        )
        errors = symmetry_aware_pose_errors(
            poses,
            target,
            symmetry_metadata,
            effective_group=effective_group,
            translation_weight=10.0,
        )
    rows = []
    for index in range(num_starts):
        rotation = float(errors["rotation_deg"][index])
        translation_mm = float(errors["translation_m"][index] * 1000.0)
        rows.append(
            {
                "start": index,
                "rotation_error_deg": rotation,
                "translation_error_mm": translation_mm,
                "pose_cost": float(errors["combined"][index]),
                "success_0p1deg_0p1mm": rotation < 0.1
                and translation_mm < 0.1,
            }
        )
    return rows


def view_scaling_summary(
    rows: Sequence[Mapping[str, Any]], *, num_queries: int
) -> dict[str, float | int]:
    """Aggregate one completed ladder run into the requested curve columns."""

    if not rows:
        raise ValueError("view scaling rows must not be empty")
    rotation = [float(row["oracle_topk_rotation_error_deg"]) for row in rows]
    translation = [float(row["oracle_translation_total_mm"]) for row in rows]
    success = [
        float(str(row["oracle_topk_success_2deg_2mm"]).lower() in {"1", "true"})
        for row in rows
    ]
    return {
        "num_views": len(rows),
        "K": int(num_queries),
        "mean_rotation_deg": sum(rotation) / len(rotation),
        "max_rotation_deg": max(rotation),
        "mean_translation_mm": sum(translation) / len(translation),
        "max_translation_mm": max(translation),
        "pose_success_2deg_2mm": sum(success) / len(success),
    }


__all__ = [
    "assignment_switch_rate",
    "direct_optimize_pose_parameters",
    "pose_context_change",
    "query_assignment_summary",
    "query_world_consistency",
    "subset_view_manifest",
    "view_scaling_summary",
]
