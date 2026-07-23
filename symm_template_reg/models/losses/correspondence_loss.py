from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.registry import LOSSES


def masked_average(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


@LOSSES.register_module()
class CorrespondenceLoss(nn.Module):
    def __init__(self, beta: float = 0.01) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, prediction: Tensor, target: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        error = torch.nn.functional.smooth_l1_loss(
            prediction, target, beta=self.beta, reduction="none"
        ).sum(-1)
        return masked_average(error, valid_mask)


@LOSSES.register_module()
class PointConfidenceLoss(nn.Module):
    def forward(self, logits: Tensor, labels: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.to(logits.dtype), reduction="none"
        )
        return masked_average(loss, valid_mask)

