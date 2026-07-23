"""Clean-room coarse feature matching.

Architectural reference: GeoTransformer (https://github.com/qinzheng93/GeoTransformer), commit
e7a135af4c318ff3b8d7f6c963df094d7e4ea540, paths
``geotransformer/modules/geotransformer/superpoint_matching.py`` and
``geotransformer/modules/ops/pairwise_distance.py`` (MIT). No source text was
copied. Changes: masked batched dual-softmax and differentiable dense scores.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from symm_template_reg.registry import MATCHERS


@MATCHERS.register_module()
class CoarseMatching(nn.Module):
    """Masked dual-softmax matching over coarse point tokens."""

    def __init__(self, temperature: float = 0.1, mutual: bool = True) -> None:
        super().__init__()
        self.temperature = temperature
        self.mutual = mutual

    def forward(
        self,
        observed: Tensor,
        template: Tensor,
        observed_mask: Tensor,
        template_mask: Tensor,
    ) -> Tensor:
        observed = torch.nn.functional.normalize(observed, dim=-1)
        template = torch.nn.functional.normalize(template, dim=-1)
        similarity = torch.matmul(observed, template.transpose(-2, -1)) / max(self.temperature, 1e-6)
        valid = observed_mask[:, :, None] & template_mask[:, None, :]
        similarity = similarity.masked_fill(~valid, -1e4)
        row = similarity.softmax(-1)
        if not self.mutual:
            return row * valid
        column = similarity.softmax(-2)
        return row * column * valid
