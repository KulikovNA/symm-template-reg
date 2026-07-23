"""Small exact rectangular assignment for K pose queries, without SciPy.

Architectural reference: DETR (https://github.com/facebookresearch/detr), commit
29901c51d7fe8712168b8d0d64351170bc0f83e0, paths ``models/matcher.py`` and
``models/detr.py`` (Apache-2.0). No source text was copied. Changes: project-local
rectangular polynomial-time assignment over an arbitrary pose cost matrix; no
box terms and no SciPy runtime dependency.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.registry import MATCHERS


def _rectangular_hungarian(cost: Tensor) -> list[int]:
    """Assign every row to a unique column in O(n*m^2), requiring n <= m."""

    rows, columns = cost.shape
    if rows > columns:
        raise ValueError("rectangular Hungarian helper requires rows <= columns")
    values = cost.detach().to(dtype=torch.float64, device="cpu").tolist()
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    matched_row = [0] * (columns + 1)
    previous_column = [0] * (columns + 1)
    for row in range(1, rows + 1):
        matched_row[0] = row
        minimum = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = matched_row[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                reduced = values[row0 - 1][column - 1] - u[row0] - v[column]
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    previous_column[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = previous_column[column0]
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * rows
    for column in range(1, columns + 1):
        if matched_row[column] != 0:
            assignment[matched_row[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise RuntimeError("Hungarian assignment did not cover every row")
    return assignment


@MATCHERS.register_module(name="HungarianPoseAssigner")
class HungarianPoseAssigner(nn.Module):
    @torch.no_grad()
    def forward(self, cost: Tensor) -> tuple[Tensor, Tensor]:
        if cost.ndim != 2:
            raise ValueError("cost must have shape [num_predictions,num_targets]")
        if cost.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=cost.device)
            return empty, empty
        if cost.shape[0] <= cost.shape[1]:
            pred = list(range(cost.shape[0]))
            target = _rectangular_hungarian(cost)
        else:
            target = list(range(cost.shape[1]))
            pred = _rectangular_hungarian(cost.transpose(0, 1))
        return (
            torch.tensor(pred, dtype=torch.long, device=cost.device),
            torch.tensor(target, dtype=torch.long, device=cost.device),
        )
