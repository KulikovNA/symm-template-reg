from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.registry import HEADS


@HEADS.register_module()
class SymmetryRegionHead(nn.Module):
    def __init__(self, embed_dim: int = 256, max_regions: int = 16) -> None:
        super().__init__()
        self.max_regions = max_regions
        self.point_classifier = nn.Linear(embed_dim, max_regions)
        self.active_classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, max_regions)
        )

    def forward(
        self,
        observed_features: Tensor,
        observed_global: Tensor,
        template_global: Tensor,
        observed_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        point_logits = self.point_classifier(observed_features)
        point_logits = point_logits.masked_fill(~observed_mask.unsqueeze(-1), 0.0)
        active_logits = self.active_classifier(torch.cat((observed_global, template_global), -1))
        return point_logits, active_logits
