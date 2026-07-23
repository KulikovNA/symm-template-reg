"""Differentiable batched weighted rigid alignment implemented in pure PyTorch."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.models.pose.pose_representation import make_transform
from symm_template_reg.registry import POSE_MODULES


@POSE_MODULES.register_module()
class WeightedProcrustes(nn.Module):
    """Estimate the transform mapping corresponding object points O to camera C."""

    def __init__(
        self,
        eps: float = 1e-8,
        minimum_effective_points: float = 3.0,
        rank_tolerance: float = 1e-7,
        fail_on_degenerate: bool = False,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.minimum_effective_points = float(minimum_effective_points)
        self.rank_tolerance = float(rank_tolerance)
        self.fail_on_degenerate = bool(fail_on_degenerate)

    def forward(
        self,
        predicted_points_O: Tensor,
        observed_points_C: Tensor,
        correspondence_weights: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        return self.solve(
            predicted_points_O,
            observed_points_C,
            correspondence_weights,
            valid_mask,
        )["transform"]

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def solve(
        self,
        predicted_points_O: Tensor,
        observed_points_C: Tensor,
        correspondence_weights: Tensor,
        valid_mask: Tensor,
        *,
        fail_on_degenerate: bool | None = None,
    ) -> dict[str, Tensor]:
        if predicted_points_O.shape != observed_points_C.shape:
            raise ValueError("corresponding point tensors must have equal shape")
        if predicted_points_O.ndim != 3 or predicted_points_O.shape[-1] != 3:
            raise ValueError("points must have shape [B,N,3]")
        if correspondence_weights.shape != predicted_points_O.shape[:2]:
            raise ValueError("correspondence_weights must have shape [B,N]")
        if valid_mask.shape != predicted_points_O.shape[:2]:
            raise ValueError("valid_mask must have shape [B,N]")
        weights = correspondence_weights.to(predicted_points_O).clamp_min(0.0)
        weights = weights * valid_mask.to(weights.dtype)
        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        normalized = weights / weight_sum
        effective_count = 1.0 / normalized.square().sum(dim=1).clamp_min(self.eps)
        source_center = torch.sum(
            normalized.unsqueeze(-1) * predicted_points_O, dim=1
        )
        target_center = torch.sum(
            normalized.unsqueeze(-1) * observed_points_C, dim=1
        )
        source = predicted_points_O - source_center.unsqueeze(1)
        target = observed_points_C - target_center.unsqueeze(1)
        covariance = source.transpose(1, 2) @ (normalized.unsqueeze(-1) * target)
        u, singular_values, vh = torch.linalg.svd(covariance, full_matrices=False)
        source_rank = torch.linalg.matrix_rank(
            source * normalized.sqrt().unsqueeze(-1), tol=self.rank_tolerance
        )
        target_rank = torch.linalg.matrix_rank(
            target * normalized.sqrt().unsqueeze(-1), tol=self.rank_tolerance
        )
        # V4 contract: a 3-D pose is reported as valid only when both point
        # clouds span all three dimensions.  The SVD transform is still
        # returned for diagnostics, but downstream pose losses/metrics mask it.
        rank_valid = (source_rank >= 3) & (target_rank >= 3)
        effective_valid = effective_count >= self.minimum_effective_points
        valid_solution = rank_valid & effective_valid & (weights.sum(dim=1) > self.eps)
        strict = self.fail_on_degenerate if fail_on_degenerate is None else bool(fail_on_degenerate)
        if strict and not bool(valid_solution.all()):
            invalid = torch.nonzero(~valid_solution, as_tuple=False).flatten().tolist()
            raise ValueError(
                "WeightedProcrustes has insufficient rank/effective correspondences "
                f"for batch indices {invalid}; source_rank={source_rank.tolist()}, "
                f"target_rank={target_rank.tolist()}, effective_count={effective_count.tolist()}"
            )
        raw_rotation = vh.transpose(-2, -1) @ u.transpose(-2, -1)
        sign = torch.where(
            torch.linalg.det(raw_rotation) < 0,
            raw_rotation.new_tensor(-1.0),
            raw_rotation.new_tensor(1.0),
        )
        correction = torch.eye(
            3, dtype=raw_rotation.dtype, device=raw_rotation.device
        ).expand(len(raw_rotation), 3, 3).clone()
        correction[:, -1, -1] = sign
        rotation = vh.transpose(-2, -1) @ correction @ u.transpose(-2, -1)
        translation = target_center - torch.einsum(
            "bij,bj->bi", rotation, source_center
        )
        return {
            "transform": make_transform(rotation, translation),
            "normalized_weights": normalized,
            "effective_correspondence_count": effective_count,
            "source_rank": source_rank,
            "target_rank": target_rank,
            "singular_values": singular_values,
            "rank_valid": rank_valid,
            "valid_solution": valid_solution,
            "determinant": torch.linalg.det(rotation),
            "orthogonality_error": torch.linalg.matrix_norm(
                rotation.transpose(-2, -1) @ rotation
                - torch.eye(3, dtype=rotation.dtype, device=rotation.device),
                ord="fro",
                dim=(-2, -1),
            ),
            "reflection_corrected": sign < 0,
            "valid_point_count": valid_mask.sum(dim=1),
            "rank": torch.minimum(source_rank, target_rank),
        }


__all__ = ["WeightedProcrustes"]
