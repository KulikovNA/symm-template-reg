"""Evaluation loop for accepted debug samples."""

from __future__ import annotations

from typing import Any

import torch

from .metrics import aggregate_metric_rows, batch_pose_metric_rows


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if hasattr(value, "to") and callable(value.to):
        return value.to(device)
    return value


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        moved = move_to_device(batch, device)
        prediction = model(moved)
        rows.extend(batch_pose_metric_rows(prediction, moved))
    if was_training:
        model.train()
    return aggregate_metric_rows(rows), rows


__all__ = ["evaluate_model", "move_to_device"]
