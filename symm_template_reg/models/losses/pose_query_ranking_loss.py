"""Quality-aware supervision for ranking direct pose queries."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn

from symm_template_reg.models.pose.metrics import symmetry_aware_pose_errors
from symm_template_reg.registry import LOSSES


def symmetry_aware_pose_costs(
    pose_hypotheses: Tensor,
    gt_poses: Tensor,
    symmetry_metadata: Sequence[Any],
    effective_symmetry_groups: Sequence[Any],
    *,
    translation_weight: float = 10.0,
    rotation_weight: float = 1.0,
) -> Tensor:
    """Return ``[B,K]`` costs using exactly the GT symmetry group."""

    if pose_hypotheses.ndim != 4 or pose_hypotheses.shape[-2:] != (4, 4):
        raise ValueError("pose_hypotheses must have shape [B,K,4,4]")
    if gt_poses.shape != (len(pose_hypotheses), 4, 4):
        raise ValueError("gt_poses must have shape [B,4,4]")
    if len(symmetry_metadata) != len(pose_hypotheses) or len(
        effective_symmetry_groups
    ) != len(pose_hypotheses):
        raise ValueError("symmetry metadata/groups must match batch size")
    rows = []
    for index in range(len(pose_hypotheses)):
        errors = symmetry_aware_pose_errors(
            pose_hypotheses[index],
            gt_poses[index],
            symmetry_metadata[index],
            effective_group=effective_symmetry_groups[index],
            translation_weight=float(translation_weight)
            / max(float(rotation_weight), 1e-12),
        )
        rows.append(
            float(translation_weight) * errors["translation_m"]
            + float(rotation_weight) * errors["rotation_rad"]
        )
    return torch.stack(rows)


@LOSSES.register_module()
class PoseQueryRankingLoss(nn.Module):
    """Train query scores using binary, categorical, or soft-quality targets."""

    def __init__(
        self,
        type: str = "soft_quality",
        temperature: float = 0.25,
        cost_normalization: str = "minmax",
        detach_pose_cost: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if type not in {"legacy_binary", "matched_categorical", "soft_quality"}:
            raise ValueError(f"unsupported pose-query ranking type: {type!r}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if cost_normalization not in {"minmax", "none"}:
            raise ValueError("cost_normalization must be minmax or none")
        self.ranking_type = type
        self.temperature = float(temperature)
        self.cost_normalization = cost_normalization
        self.detach_pose_cost = bool(detach_pose_cost)
        self.eps = float(eps)

    def target_distribution(self, pose_costs: Tensor) -> Tensor:
        costs = pose_costs.detach() if self.detach_pose_cost else pose_costs
        if self.cost_normalization == "minmax":
            minimum = costs.amin(dim=-1, keepdim=True)
            maximum = costs.amax(dim=-1, keepdim=True)
            costs = (costs - minimum) / (maximum - minimum + self.eps)
        return torch.softmax(-costs / self.temperature, dim=-1)

    def forward(self, pose_logits: Tensor, pose_costs: Tensor) -> dict[str, Tensor]:
        if pose_logits.shape != pose_costs.shape or pose_logits.ndim != 2:
            raise ValueError("pose_logits and pose_costs must both have shape [B,K]")
        matched = torch.argmin(pose_costs.detach(), dim=-1)
        if self.ranking_type == "legacy_binary":
            target = torch.zeros_like(pose_logits)
            target.scatter_(1, matched.unsqueeze(-1), 1.0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pose_logits, target
            )
            distribution = target
        elif self.ranking_type == "matched_categorical":
            loss = torch.nn.functional.cross_entropy(pose_logits, matched)
            distribution = torch.nn.functional.one_hot(
                matched, num_classes=pose_logits.shape[-1]
            ).to(pose_logits.dtype)
        else:
            distribution = self.target_distribution(pose_costs)
            loss = -(distribution * torch.log_softmax(pose_logits, dim=-1)).sum(
                dim=-1
            ).mean()
        return {
            "loss_pose_query_ranking": loss,
            "pose_query_target_distribution": distribution,
            "pose_query_costs": pose_costs,
            "oracle_query_indices": matched,
        }


__all__ = ["PoseQueryRankingLoss", "symmetry_aware_pose_costs"]
