"""Pure-PyTorch point-cloud primitives; no compiled extensions are required."""

from __future__ import annotations

import torch
from torch import Tensor


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def masked_max(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    fill = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask.unsqueeze(-1), fill)
    result = masked.max(dim=dim).values
    any_valid = mask.any(dim=dim, keepdim=False).unsqueeze(-1)
    return torch.where(any_valid, result, torch.zeros_like(result))


def batched_gather(values: Tensor, indices: Tensor) -> Tensor:
    """Gather ``[B,N,C]`` values with ``[B,...]`` indices."""

    if values.ndim != 3 or indices.ndim < 2 or values.shape[0] != indices.shape[0]:
        raise ValueError("expected values [B,N,C] and indices [B,...]")
    batch_shape = [values.shape[0]] + [1] * (indices.ndim - 1)
    batch = torch.arange(values.shape[0], device=values.device).view(batch_shape)
    batch = batch.expand_as(indices)
    return values[batch, indices]


@torch.no_grad()
def knn_indices(
    query: Tensor,
    support: Tensor,
    support_mask: Tensor,
    k: int,
    *,
    chunk_size: int = 512,
) -> Tensor:
    """Return nearest support indices, chunking queries to bound temporary memory."""

    if query.ndim != 3 or support.ndim != 3:
        raise ValueError("query and support must be [B,N,3]")
    k = min(max(int(k), 1), support.shape[1])
    output = []
    for start in range(0, query.shape[1], chunk_size):
        distance = torch.cdist(query[:, start : start + chunk_size].float(), support.float())
        distance.masked_fill_(~support_mask[:, None, :], float("inf"))
        values, indices = distance.topk(k, dim=-1, largest=False)
        # topk returns arbitrary padded indices when a cloud has fewer than k
        # valid supports. Repeat its closest valid point instead, so padded
        # coordinates can never affect a valid query's local descriptor.
        fallback = distance.argmin(dim=-1, keepdim=True).expand_as(indices)
        indices = torch.where(torch.isfinite(values), indices, fallback)
        output.append(indices)
    return torch.cat(output, dim=1)


@torch.no_grad()
def farthest_point_indices(
    points: Tensor,
    valid_mask: Tensor,
    max_points: int,
) -> tuple[Tensor, Tensor]:
    """Deterministic batched farthest-point sampling for bounded attention tokens."""

    batch = points.shape[0]
    k = min(max(int(max_points), 1), points.shape[1])
    selected = torch.zeros((batch, k), dtype=torch.long, device=points.device)
    selected_mask = torch.zeros((batch, k), dtype=torch.bool, device=points.device)
    for b in range(batch):
        valid = torch.nonzero(valid_mask[b], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        count = min(k, int(valid.numel()))
        cloud = points[b, valid].float()
        centroid = cloud.mean(dim=0, keepdim=True)
        current = torch.linalg.vector_norm(cloud - centroid, dim=-1).argmax()
        min_distance = torch.full((cloud.shape[0],), float("inf"), device=points.device)
        local_selection = []
        for _ in range(count):
            local_selection.append(current)
            distance = ((cloud - cloud[current]) ** 2).sum(-1)
            min_distance = torch.minimum(min_distance, distance)
            current = min_distance.argmax()
        chosen = valid[torch.stack(local_selection)]
        selected[b, :count] = chosen
        selected_mask[b, :count] = True
    return selected, selected_mask


def select_tokens(
    points: Tensor,
    features: Tensor,
    valid_mask: Tensor,
    max_tokens: int,
    precomputed_indices: Tensor | None = None,
    precomputed_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if precomputed_indices is None:
        indices, token_mask = farthest_point_indices(points, valid_mask, max_tokens)
    else:
        indices = precomputed_indices
        if precomputed_mask is None:
            raise ValueError("precomputed FPS indices require their validity mask")
        token_mask = precomputed_mask
    return (
        batched_gather(points, indices),
        batched_gather(features, indices),
        token_mask,
        indices,
    )


def nearest_interpolate(
    query_points: Tensor,
    support_points: Tensor,
    support_features: Tensor,
    support_mask: Tensor,
    precomputed_indices: Tensor | None = None,
) -> Tensor:
    indices = (
        knn_indices(query_points, support_points, support_mask, 1).squeeze(-1)
        if precomputed_indices is None else precomputed_indices
    )
    return batched_gather(support_features, indices)


@torch.no_grad()
def nearest_grouped_point_ids(
    query: Tensor,
    grouped_support: Tensor,
    *,
    chunk_size: int = 256,
) -> Tensor:
    """Find the support-group nearest to every query with bounded memory.

    ``query`` is ``[N,D]`` and ``grouped_support`` is ``[G,K,D]``.  The
    unchunked distance matrix can be very large for dense shell points and
    face candidates, while only the winning group id is needed.
    """
    if query.ndim != 2 or grouped_support.ndim != 3:
        raise ValueError("expected query [N,D] and grouped_support [G,K,D]")
    if query.shape[-1] != grouped_support.shape[-1]:
        raise ValueError("query and grouped support dimensions disagree")
    flat = grouped_support.reshape(-1, grouped_support.shape[-1]).float()
    result = []
    for start in range(0, len(query), int(chunk_size)):
        distance = torch.cdist(query[start : start + int(chunk_size)].float(), flat)
        grouped = distance.reshape(len(distance), len(grouped_support), -1)
        result.append(grouped.amin(-1).argmin(-1))
    return torch.cat(result)
