"""Minimal physical evaluation for the sole production coordinate path."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from symm_template_reg.geometry.triangle_surface import closest_points_on_triangle_mesh
from symm_template_reg.models.geometry.aux_guided_triangle_candidates import (
    AuxGuidedTriangleCandidateBuilder,
)
from symm_template_reg.models.geometry.triangle_targets import (
    closest_barycentric_on_triangles,
)
from symm_template_reg.models.heads.coordinate_guided_surface_projection import (
    CoordinateGuidedSurfaceProjectionHead,
)
from symm_template_reg.models.pose.pose_representation import transform_points
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance


def _distance_summary_mm(values_m: Tensor, prefix: str) -> dict[str, float]:
    values = values_m.detach().float()
    return {
        f"{prefix}_rmse_mm": float(values.square().mean().sqrt() * 1000.0),
        f"{prefix}_p50_mm": float(torch.quantile(values, 0.50) * 1000.0),
        f"{prefix}_p95_mm": float(torch.quantile(values, 0.95) * 1000.0),
        f"{prefix}_max_mm": float(values.max() * 1000.0),
    }


def active_values_are_finite(values: Mapping[str, Any]) -> bool:
    return all(
        math.isfinite(float(value))
        for value in values.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _pose_metrics(
    projected_O: Tensor,
    observed_C: Tensor,
    target_pose: Tensor,
    procrustes: Any,
) -> tuple[dict[str, float | int | bool], Tensor]:
    valid = torch.ones((1, len(projected_O)), dtype=torch.bool, device=projected_O.device)
    solution = procrustes.solve(
        projected_O[None].float(),
        observed_C[None].float(),
        projected_O.new_ones((1, len(projected_O))).float(),
        valid,
    )
    pose = solution["transform"][0].to(projected_O)
    reconstructed = transform_points(pose[None], projected_O[None])[0]
    alignment = torch.linalg.vector_norm(reconstructed - observed_C, dim=-1)
    rotation = torch.rad2deg(
        rotation_geodesic_distance(
            pose[:3, :3][None], target_pose[:3, :3][None]
        )
    )[0]
    translation = torch.linalg.vector_norm(
        pose[:3, 3] - target_pose[:3, 3]
    ) * 1000.0
    return {
        **_distance_summary_mm(alignment, "projection_alignment"),
        "projection_rotation_error_deg": float(rotation),
        "projection_translation_error_mm": float(translation),
        "projection_rank": int(solution["rank"][0]),
        "projection_rank_valid": bool(solution["rank_valid"][0]),
    }, pose


@torch.no_grad()
def evaluate_active_sample(
    *,
    q_aux_O: Tensor,
    valid_mask: Tensor,
    target_O: Tensor,
    observed_C: Tensor,
    vertices_O: Tensor,
    faces: Tensor,
    equivalent_pose: Tensor,
    procrustes: Any,
    candidate_k: int = 16,
    projection_chunk_size: int = 256,
) -> dict[str, Any]:
    mask = valid_mask.bool()
    q_valid = q_aux_O[mask]
    target_valid = target_O[mask]
    observed_valid = observed_C[mask]
    if not len(q_valid):
        raise ValueError("evaluation sample has no valid correspondence points")

    if q_aux_O.device.type == "cuda":
        torch.cuda.synchronize(q_aux_O.device)
    started = time.perf_counter()
    exact = closest_points_on_triangle_mesh(
        q_valid, vertices_O, faces.long(), point_chunk_size=projection_chunk_size
    )
    if q_aux_O.device.type == "cuda":
        torch.cuda.synchronize(q_aux_O.device)
    exact_runtime_ms = (time.perf_counter() - started) * 1000.0
    exact_points = exact["points"]
    exact_ids = exact["face_ids"].long()

    if q_aux_O.device.type == "cuda":
        torch.cuda.synchronize(q_aux_O.device)
    started = time.perf_counter()
    built = AuxGuidedTriangleCandidateBuilder(
        mode="aux_guided_global_topk",
        candidate_k=int(candidate_k),
        projection_chunk_size=int(projection_chunk_size),
    ).to(q_aux_O.device)(q_aux_O[None], [vertices_O], [faces], mask[None])
    ids = built["candidate_triangle_ids"]
    candidate_mask = built["candidate_triangle_mask"]
    projected = CoordinateGuidedSurfaceProjectionHead().to(q_aux_O.device)(
        q_aux_O[None], ids, [vertices_O], [faces], mask[None], candidate_mask
    )
    k16_points = projected["surface_correspondence_points_O"][0, mask]
    k16_ids = projected["selected_triangle_ids"][0, mask]
    if q_aux_O.device.type == "cuda":
        torch.cuda.synchronize(q_aux_O.device)
    k16_runtime_ms = (time.perf_counter() - started) * 1000.0
    shortlist = ids[0, mask]
    shortlist_mask = candidate_mask[0, mask]
    recall = ((shortlist == exact_ids[:, None]) & shortlist_mask).any(-1).float().mean()

    modes: dict[str, Any] = {}
    poses: dict[str, Tensor] = {}
    for name, points, selected_ids, runtime_ms in (
        ("exact_global", exact_points, exact_ids, exact_runtime_ms),
        ("k16", k16_points, k16_ids, k16_runtime_ms),
    ):
        pose_values, pose = _pose_metrics(
            points, observed_valid, equivalent_pose, procrustes
        )
        triangles = vertices_O[faces.long()[selected_ids]]
        membership = closest_barycentric_on_triangles(points, triangles)["distances"]
        values = {
            **_distance_summary_mm(
                torch.linalg.vector_norm(points - target_valid, dim=-1),
                "projected_correspondence",
            ),
            **pose_values,
            "surface_membership_p95_mm": float(
                torch.quantile(membership.float(), 0.95) * 1000.0
            ),
            "runtime_ms": runtime_ms,
        }
        values["nonfinite_detected"] = not active_values_are_finite(values)
        modes[name], poses[name] = values, pose
    modes["k16"].update(
        exact_global_triangle_recall=float(recall),
        fallback_fraction=float((~shortlist_mask.any(-1)).float().mean()),
        candidate_count_min=int(shortlist_mask.sum(-1).min()),
        candidate_count_max=int(shortlist_mask.sum(-1).max()),
    )
    return {
        "num_shell_points": int(mask.sum()),
        "raw_q_aux": _distance_summary_mm(
            torch.linalg.vector_norm(q_valid - target_valid, dim=-1),
            "aux_coordinate",
        ),
        "exact_global": modes["exact_global"],
        "k16": modes["k16"],
        "T_C_from_O": poses,
        "projected_points_O": {"exact_global": exact_points, "k16": k16_points},
        "selected_triangle_ids": {"exact_global": exact_ids, "k16": k16_ids},
    }


def active_row(
    result: Mapping[str, Any],
    *,
    sample_id: str,
    frame_id: int,
    T_W_from_C: Tensor,
    target_leakage_detected: bool = False,
) -> dict[str, Any]:
    exact, k16 = result["exact_global"], result["k16"]
    exact_pose = result["T_C_from_O"]["exact_global"]
    k16_pose = result["T_C_from_O"]["k16"]
    values = {
        "sample_id": str(sample_id),
        "frame_id": int(frame_id),
        "num_shell_points": int(result["num_shell_points"]),
        **result["raw_q_aux"],
        **{f"exact_global_{key}": value for key, value in exact.items()},
        **{f"k16_{key}": value for key, value in k16.items()},
        "k16_exact_global_triangle_recall": k16["exact_global_triangle_recall"],
        "k16_fallback_fraction": k16["fallback_fraction"],
        "exact_global_T_C_from_O": exact_pose.detach().cpu().tolist(),
        "k16_T_C_from_O": k16_pose.detach().cpu().tolist(),
        "exact_global_T_W_from_O": (T_W_from_C @ exact_pose).detach().cpu().tolist(),
        "k16_T_W_from_O": (T_W_from_C @ k16_pose).detach().cpu().tolist(),
        "target_leakage_detected": bool(target_leakage_detected),
    }
    values["active_nonfinite_detected"] = not active_values_are_finite(values)
    return values


__all__ = ["active_row", "active_values_are_finite", "evaluate_active_sample"]
