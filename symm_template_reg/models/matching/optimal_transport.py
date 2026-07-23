"""Dependency-free log-domain Sinkhorn normalization with an optional dustbin.

Architectural reference: GeoTransformer (https://github.com/qinzheng93/GeoTransformer), commit
e7a135af4c318ff3b8d7f6c963df094d7e4ea540,
``geotransformer/modules/sinkhorn/learnable_sinkhorn.py`` (MIT). No source text
was copied. Changes: input-derived device/dtype, explicit masks, no package or
compiled-extension imports.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.registry import MATCHERS


@MATCHERS.register_module()
class LogOptimalTransport(nn.Module):
    def __init__(self, num_iterations: int = 20, dustbin: bool = True) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        self.use_dustbin = dustbin
        self.dustbin_score = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        scores: Tensor,
        row_mask: Tensor | None = None,
        column_mask: Tensor | None = None,
    ) -> Tensor:
        batch, rows, columns = scores.shape
        if row_mask is None:
            row_mask = torch.ones((batch, rows), dtype=torch.bool, device=scores.device)
        if column_mask is None:
            column_mask = torch.ones((batch, columns), dtype=torch.bool, device=scores.device)
        if self.use_dustbin:
            dustbin = self.dustbin_score.to(dtype=scores.dtype, device=scores.device)
            augmented = torch.cat(
                (
                    torch.cat((scores, dustbin.expand(batch, rows, 1)), dim=-1),
                    dustbin.expand(batch, 1, columns + 1),
                ),
                dim=-2,
            )
            row_mask = torch.cat((row_mask, torch.ones_like(row_mask[:, :1])), -1)
            column_mask = torch.cat((column_mask, torch.ones_like(column_mask[:, :1])), -1)
        else:
            augmented = scores
        valid = row_mask[:, :, None] & column_mask[:, None, :]
        log_assignment = augmented.masked_fill(~valid, -1e4)
        for _ in range(self.num_iterations):
            log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=-1, keepdim=True)
            log_assignment = log_assignment.masked_fill(~valid, -1e4)
            log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=-2, keepdim=True)
            log_assignment = log_assignment.masked_fill(~valid, -1e4)
        return log_assignment[:, :rows, :columns]
