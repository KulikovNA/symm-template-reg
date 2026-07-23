from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.registry import LOSSES

from .correspondence_loss import masked_average


def masked_point_region_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    valid_mask: Tensor,
    region_valid_mask: Tensor,
    *,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Categorical per-point loss excluding padding, OOB points and spare slots."""

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("point region logits must be [B,N,R] and labels [B,N]")
    if valid_mask.shape != labels.shape:
        raise ValueError("point region valid_mask must match labels")
    if region_valid_mask.ndim != 2 or region_valid_mask.shape[0] != logits.shape[0]:
        raise ValueError("region_valid_mask must have shape [B,R_valid]")
    capacity = logits.shape[-1]
    slot_valid = torch.zeros(
        (logits.shape[0], capacity), dtype=torch.bool, device=logits.device
    )
    width = min(capacity, region_valid_mask.shape[-1])
    slot_valid[:, :width] = region_valid_mask[:, :width]
    masked_logits = logits.masked_fill(~slot_valid[:, None, :], -1e9)
    usable = valid_mask & labels.ge(0) & labels.lt(capacity)
    if bool(usable.any()):
        label_slot_valid = torch.gather(
            slot_valid, 1, labels.clamp(0, capacity - 1)
        )
        usable = usable & label_slot_valid
    if not bool(usable.any()):
        return logits.sum() * 0.0
    weights = None
    if class_weights is not None:
        weights = torch.as_tensor(
            class_weights, dtype=logits.dtype, device=logits.device
        )
        if weights.ndim != 1 or weights.numel() > capacity:
            raise ValueError("point-region class_weights must fit region capacity")
        if weights.numel() < capacity:
            weights = torch.cat([weights, weights.new_zeros(capacity - weights.numel())])
    raw = torch.nn.functional.cross_entropy(
        masked_logits.transpose(1, 2),
        labels.clamp_min(0),
        weight=weights,
        reduction="none",
    )
    return (raw * usable.to(raw.dtype)).sum() / usable.sum()


def active_region_binary_loss(
    logits: Tensor,
    labels: Tensor,
    valid_mask: Tensor,
    *,
    loss_type: str = "bce",
    focal_gamma: float = 2.0,
    pos_weight: Tensor | None = None,
) -> Tensor:
    width = min(logits.shape[-1], labels.shape[-1], valid_mask.shape[-1])
    selected_logits = logits[:, :width]
    selected_labels = labels[:, :width].to(selected_logits.dtype)
    selected_valid = valid_mask[:, :width]
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        selected_logits,
        selected_labels,
        pos_weight=(
            torch.as_tensor(pos_weight, dtype=selected_logits.dtype, device=logits.device)[
                :width
            ]
            if pos_weight is not None
            else None
        ),
        reduction="none",
    )
    if loss_type == "focal_bce":
        probability = torch.sigmoid(selected_logits)
        pt = torch.where(selected_labels.bool(), probability, 1.0 - probability)
        raw = raw * (1.0 - pt).pow(float(focal_gamma))
    elif loss_type != "bce":
        raise ValueError("active-region loss type must be bce or focal_bce")
    return (raw * selected_valid.to(raw.dtype)).sum() / selected_valid.sum().clamp_min(1)


def aggregate_point_region_activity(
    point_logits: Tensor,
    point_valid_mask: Tensor,
    region_valid_mask: Tensor,
    *,
    aggregation: str = "topk_mean",
    topk: int = 16,
) -> Tensor:
    """Differentiably aggregate categorical point probabilities to region activity."""

    capacity = point_logits.shape[-1]
    slot_valid = torch.zeros(
        (point_logits.shape[0], capacity), dtype=torch.bool, device=point_logits.device
    )
    width = min(capacity, region_valid_mask.shape[-1])
    slot_valid[:, :width] = region_valid_mask[:, :width]
    logits = point_logits.masked_fill(~slot_valid[:, None, :], -1e9)
    probabilities = torch.softmax(logits, dim=-1)
    rows = []
    for index in range(len(probabilities)):
        selected = probabilities[index, point_valid_mask[index]]
        if len(selected) == 0:
            rows.append(probabilities[index].sum(dim=0) * 0.0)
            continue
        if aggregation == "topk_mean":
            count = min(max(int(topk), 1), len(selected))
            aggregate = selected.topk(count, dim=0).values.mean(dim=0)
        elif aggregation == "logsumexp":
            aggregate = torch.sigmoid(
                torch.logsumexp(torch.logit(selected.clamp(1e-6, 1 - 1e-6)), dim=0)
                - torch.log(selected.new_tensor(float(len(selected))))
            )
        elif aggregation == "noisy_or":
            aggregate = 1.0 - torch.prod(1.0 - selected.clamp(0.0, 1.0), dim=0)
        else:
            raise ValueError("region aggregation must be topk_mean, logsumexp, or noisy_or")
        rows.append(aggregate * slot_valid[index].to(aggregate.dtype))
    return torch.stack(rows)


@LOSSES.register_module()
class RegionLoss(nn.Module):
    def forward(self, logits: Tensor, labels: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.to(logits.dtype), reduction="none"
        )
        return masked_average(loss, valid_mask)
