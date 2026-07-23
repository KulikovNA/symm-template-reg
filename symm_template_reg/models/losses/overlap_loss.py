from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.registry import LOSSES

from .correspondence_loss import masked_average


@LOSSES.register_module()
class OverlapLoss(nn.Module):
    def forward(self, logits: Tensor, labels: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.to(logits.dtype), reduction="none"
        )
        return masked_average(loss, valid_mask)


@LOSSES.register_module()
class InsufficientInformationLoss(nn.Module):
    """Binary classification loss for samples that cannot support a pose."""

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.to(logits.dtype)
        )
