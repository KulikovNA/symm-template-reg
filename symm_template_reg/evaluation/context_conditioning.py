"""Metrics that verify pose outputs actually respond to view-specific inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from symm_template_reg.models.pose.metrics import rotation_error_deg


def _pairwise_pose(poses: Tensor) -> tuple[Tensor, Tensor]:
    rotations = []
    translations = []
    for left in range(len(poses)):
        for right in range(left + 1, len(poses)):
            rotations.append(rotation_error_deg(poses[left], poses[right]))
            translations.append(
                torch.linalg.vector_norm(
                    poses[left, :3, 3] - poses[right, :3, 3]
                )
                * 1000.0
            )
    if not rotations:
        zero = poses.new_zeros((0,))
        return zero, zero
    return torch.stack(rotations), torch.stack(translations)


def _pose_matrices(poses: Tensor) -> tuple[Tensor, Tensor]:
    count = len(poses)
    rotation = poses.new_zeros((count, count))
    translation = poses.new_zeros((count, count))
    for left in range(count):
        for right in range(left + 1, count):
            r = rotation_error_deg(poses[left], poses[right])
            t = torch.linalg.vector_norm(
                poses[left, :3, 3] - poses[right, :3, 3]
            ) * 1000.0
            rotation[left, right] = rotation[right, left] = r
            translation[left, right] = translation[right, left] = t
    return rotation, translation


def _correlation(left: Tensor, right: Tensor, *, ranks: bool = False) -> float:
    if left.numel() < 2:
        return 1.0
    if ranks:
        left = torch.argsort(torch.argsort(left)).to(torch.float64)
        right = torch.argsort(torch.argsort(right)).to(torch.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 1e-12:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def context_conditioning_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        len(rows) >= 2
        and "correspondence_prediction_summary" in rows[0]
        and "sample_context" not in rows[0]
    ):
        summaries = torch.as_tensor(
            [row["correspondence_prediction_summary"] for row in rows],
            dtype=torch.float64,
        )
        ranks = [row.get("fine_feature_effective_rank") for row in rows]
        variances = [row.get("fine_feature_variance") for row in rows]
        return {
            "conditioning_source": "clean_q_aux_summary",
            "q_aux_summary_pairwise_distance_matrix": torch.cdist(
                summaries, summaries
            ).tolist(),
            "q_aux_summary_variance_per_dimension": summaries.var(
                dim=0, unbiased=False
            ).tolist(),
            "q_aux_summary_pairwise_distance_mean": float(
                torch.pdist(summaries).mean()
            ),
            "fine_feature_effective_rank_per_sample": ranks,
            "fine_feature_variance_per_sample": variances,
            "input_conditioning_collapsed": bool(
                float(torch.pdist(summaries).mean()) <= 1e-8
            ),
        }
    if (
        len(rows) < 2
        or "base_T_C_from_O" not in rows[0]
        or "sample_context" not in rows[0]
    ):
        return {}
    base = torch.as_tensor([row["base_T_C_from_O"] for row in rows], dtype=torch.float64)
    gt = torch.as_tensor([row["gt_T_C_from_O"] for row in rows], dtype=torch.float64)
    result: dict[str, Any] = {}
    for name in ("sample_context", "rotation_context", "translation_context"):
        if name in rows[0]:
            value = torch.as_tensor([row[name] for row in rows], dtype=torch.float64)
            result[f"{name}_pairwise_distance_matrix"] = torch.cdist(value, value).tolist()
            result[f"{name}_variance_per_dimension"] = value.var(dim=0, unbiased=False).tolist()
    gt_rotation, gt_translation = _pose_matrices(gt)
    base_rotation, base_translation = _pose_matrices(base)
    result.update(
        {
            "gt_pose_pairwise_rotation_deg_matrix": gt_rotation.tolist(),
            "gt_pose_pairwise_translation_mm_matrix": gt_translation.tolist(),
            "predicted_base_pose_pairwise_rotation_deg_matrix": base_rotation.tolist(),
            "predicted_base_pose_pairwise_translation_mm_matrix": base_translation.tolist(),
        }
    )
    return result


def input_permutation_equivariance_error(
    original_base_poses: Tensor,
    permuted_observed_base_poses: Tensor,
    permutation: Tensor | Sequence[int],
) -> dict[str, float]:
    order = torch.as_tensor(
        permutation, dtype=torch.long, device=original_base_poses.device
    )
    expected = original_base_poses[order]
    rotation = rotation_error_deg(permuted_observed_base_poses, expected)
    translation = torch.linalg.vector_norm(
        permuted_observed_base_poses[..., :3, 3] - expected[..., :3, 3], dim=-1
    ) * 1000.0
    return {
        "input_permutation_equivariance_rotation_error_deg": float(rotation.mean()),
        "input_permutation_equivariance_translation_error_mm": float(
            translation.mean()
        ),
        "input_permutation_equivariance_error": float(
            rotation.mean() + translation.mean()
        ),
    }


def context_conditioning_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    significant_gt_rotation_deg: float = 5.0,
    static_rotation_deg: float = 0.1,
) -> dict[str, float]:
    if (
        len(rows) < 2
        or "base_T_C_from_O" not in rows[0]
        or "sample_context" not in rows[0]
    ):
        return {}
    base = torch.as_tensor(
        [row["base_T_C_from_O"] for row in rows], dtype=torch.float64
    )
    gt = torch.as_tensor([row["gt_T_C_from_O"] for row in rows], dtype=torch.float64)
    context = torch.as_tensor(
        [row["sample_context"] for row in rows], dtype=torch.float64
    )
    base_rotation, base_translation = _pairwise_pose(base)
    gt_rotation, gt_translation = _pairwise_pose(gt)
    context_distance = torch.pdist(context)
    rotation_context = torch.as_tensor(
        [row.get("rotation_context", row["sample_context"]) for row in rows],
        dtype=torch.float64,
    )
    translation_context = torch.as_tensor(
        [row.get("translation_context", row["sample_context"]) for row in rows],
        dtype=torch.float64,
    )
    rotation_context_distance = torch.pdist(rotation_context)
    translation_context_distance = torch.pdist(translation_context)
    variance = context.var(dim=0, unbiased=False)
    eligible = gt_rotation >= significant_gt_rotation_deg
    static = (base_rotation <= static_rotation_deg) & eligible
    denominator = eligible.sum().clamp_min(1)
    result = {
        "context_pairwise_distance": float(context_distance.mean()),
        "rotation_context_pairwise_distance": float(rotation_context_distance.mean()),
        "translation_context_pairwise_distance": float(translation_context_distance.mean()),
        "context_pose_distance_correlation": _correlation(
            context_distance, gt_rotation
        ),
        "rotation_context_gt_rotation_spearman": _correlation(
            rotation_context_distance, gt_rotation, ranks=True
        ),
        "context_variance_per_dimension": float(variance.mean()),
        "collapsed_context_dimension_fraction": float((variance <= 1e-8).double().mean()),
        "base_pose_pairwise_rotation_deg": float(base_rotation.mean()),
        "base_pose_pairwise_translation_mm": float(base_translation.mean()),
        "gt_pose_pairwise_rotation_deg": float(gt_rotation.mean()),
        "gt_pose_pairwise_translation_mm": float(gt_translation.mean()),
        "rotation_response_ratio": float(
            base_rotation.mean() / gt_rotation.mean().clamp_min(1e-8)
        ),
        "translation_response_ratio": float(
            base_translation.mean() / gt_translation.mean().clamp_min(1e-8)
        ),
        "base_pose_static_fraction": float(static.sum() / denominator),
    }
    if "query_T_C_from_O" in rows[0]:
        query = torch.as_tensor(
            [row["query_T_C_from_O"] for row in rows], dtype=torch.float64
        )
        query_static = []
        for index in range(query.shape[1]):
            query_rotation, _ = _pairwise_pose(query[:, index])
            query_static.append(
                ((query_rotation <= static_rotation_deg) & eligible).sum()
                / denominator
            )
        result["query_static_codebook_score"] = float(
            torch.stack(query_static).mean()
        )
    if "residual_T_camera" in rows[0]:
        residual = torch.as_tensor(
            [row["residual_T_camera"] for row in rows], dtype=torch.float64
        )
        residual_static = []
        for index in range(residual.shape[1]):
            residual_rotation, _ = _pairwise_pose(residual[:, index])
            residual_static.append(
                ((residual_rotation <= static_rotation_deg) & eligible).sum()
                / denominator
            )
        result["residual_query_static_fraction"] = float(
            torch.stack(residual_static).mean()
        )
    if "hybrid_correction_T" in rows[0]:
        hybrid = torch.as_tensor(
            [row["hybrid_correction_T"] for row in rows], dtype=torch.float64
        )
        hybrid_rotation, hybrid_translation = _pairwise_pose(hybrid)
        result["hybrid_residual_static_fraction"] = float(
            ((hybrid_rotation <= static_rotation_deg) & eligible).sum() / denominator
        )
        result["hybrid_residual_pairwise_rotation_deg"] = float(
            hybrid_rotation.mean()
        )
        result["hybrid_residual_pairwise_translation_mm"] = float(
            hybrid_translation.mean()
        )
    return result


__all__ = [
    "context_conditioning_diagnostics",
    "context_conditioning_metrics",
    "input_permutation_equivariance_error",
]
