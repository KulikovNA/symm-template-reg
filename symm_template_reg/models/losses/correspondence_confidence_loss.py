"""Prevent correspondence weights from collapsing to zero or one point."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.registry import LOSSES


def correspondence_confidence_diagnostics(
    confidence: Tensor, valid_mask: Tensor, *, eps: float = 1e-8
) -> dict[str, Tensor]:
    weights = confidence.clamp_min(0.0) * valid_mask.to(confidence.dtype)
    sums = weights.sum(dim=-1, keepdim=True)
    uniform = valid_mask.to(confidence.dtype)
    uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
    normalized = torch.where(sums > eps, weights / sums.clamp_min(eps), uniform)
    entropy = -(normalized * normalized.clamp_min(eps).log()).sum(dim=-1)
    effective_count = 1.0 / normalized.square().sum(dim=-1).clamp_min(eps)
    return {
        "normalized_weights": normalized,
        "weight_sum": sums.squeeze(-1),
        "entropy": entropy,
        "effective_count": effective_count,
        "max_normalized_weight": normalized.max(dim=-1).values,
    }


@LOSSES.register_module()
class CorrespondenceConfidenceRegularizationLoss(nn.Module):
    def __init__(
        self,
        minimum_effective_point_count: float = 16.0,
        minimum_weight_sum: float = 1e-3,
    ) -> None:
        super().__init__()
        self.minimum_effective_point_count = float(minimum_effective_point_count)
        self.minimum_weight_sum = float(minimum_weight_sum)

    def forward(self, confidence: Tensor, valid_mask: Tensor) -> dict[str, Tensor]:
        diagnostics = correspondence_confidence_diagnostics(confidence, valid_mask)
        effective_penalty = torch.relu(
            confidence.new_tensor(self.minimum_effective_point_count)
            - diagnostics["effective_count"]
        ) / max(self.minimum_effective_point_count, 1.0)
        sum_penalty = torch.relu(
            confidence.new_tensor(self.minimum_weight_sum)
            - diagnostics["weight_sum"]
        ) / max(self.minimum_weight_sum, 1e-8)
        return {
            "loss_confidence_regularization": (effective_penalty + sum_penalty).mean(),
            "confidence_entropy": diagnostics["entropy"].mean(),
            "effective_correspondence_count": diagnostics["effective_count"].mean(),
            "minimum_effective_correspondence_count": diagnostics["effective_count"].min(),
            "maximum_normalized_correspondence_weight": diagnostics["max_normalized_weight"].max(),
            "confidence_weight_sum": diagnostics["weight_sum"].mean(),
        }


__all__ = [
    "CorrespondenceConfidenceRegularizationLoss",
    "correspondence_confidence_diagnostics",
]
