"""Set-valued targets for overlapping template patch candidate regions."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


PATCH_TARGET_MODES = {"single_owner", "multi_valid_patch_set"}


def valid_patch_mask(
    gt_triangle_ids: Tensor, all_candidate_triangle_ids: Tensor
) -> Tensor:
    """Return patches whose local candidate list contains each GT triangle.

    Args:
        gt_triangle_ids: ``[..., N]`` triangle ids.
        all_candidate_triangle_ids: ``[..., P, C]`` patch candidate ids.  The
            leading dimensions must broadcast with those of ``gt_triangle_ids``.
    Returns:
        Boolean tensor ``[..., N, P]``.
    """

    return all_candidate_triangle_ids.unsqueeze(-3).eq(
        gt_triangle_ids.unsqueeze(-1).unsqueeze(-1)
    ).any(-1)


def single_owner_patch_ids(gt_triangle_ids: Tensor, face_owner_patch_ids: Tensor) -> Tensor:
    """Map exact GT triangle ids to their unique Voronoi/FPS owner patch."""

    return face_owner_patch_ids.gather(-1, gt_triangle_ids.long())


def multi_positive_softmax_loss(
    logits: Tensor,
    valid_targets: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Stable ``-log(sum(P(valid class)))`` loss.

    This is a categorical multi-positive objective.  It deliberately is not
    independent BCE: probability mass still competes across all patches.
    """

    if logits.shape != valid_targets.shape:
        raise ValueError(
            f"logits and valid_targets must have identical shape, got "
            f"{tuple(logits.shape)} and {tuple(valid_targets.shape)}"
        )
    valid_targets = valid_targets.bool()
    if not bool(valid_targets.any(-1).all()):
        raise ValueError("every row must contain at least one valid target")
    log_probability = F.log_softmax(logits, dim=-1)
    selected = log_probability.masked_fill(~valid_targets, float("-inf"))
    loss = -torch.logsumexp(selected, dim=-1)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unsupported reduction: {reduction}")


def valid_set_topk_hits(
    logits_or_topk: Tensor,
    valid_targets: Tensor,
    k: int,
    *,
    already_topk: bool = False,
) -> Tensor:
    """Return one boolean hit per point for a set-valued top-k target."""

    if k < 1:
        raise ValueError("k must be positive")
    topk = (
        logits_or_topk[..., : min(k, logits_or_topk.shape[-1])]
        if already_topk
        else logits_or_topk.topk(min(k, logits_or_topk.shape[-1]), dim=-1).indices
    )
    return valid_targets.gather(-1, topk.long()).any(-1)


__all__ = [
    "PATCH_TARGET_MODES",
    "multi_positive_softmax_loss",
    "single_owner_patch_ids",
    "valid_patch_mask",
    "valid_set_topk_hits",
]
