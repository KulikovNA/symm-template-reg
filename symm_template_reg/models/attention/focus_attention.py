"""Audited clean-room focus-attention interface, not enabled by the baseline.

Architectural reference: DFAT (https://github.com/fukexue/DFAT), commit
884149656199c734e2fceff1eda7d7d3b8ebf8c6, paths
``geotransformer/modules/transformer/spotguided_transformer.py``,
``geotransformer/modules/lineartransformer/linear_attention.py``,
``geotransformer/modules/lineartransformer/transformer.py`` and
``experiments/3DMatch/model.py`` (MIT). No source text was copied. Changes: only
a learned saliency/top-k interface; no fine-scale integration or native ops.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.registry import ATTENTION


@ATTENTION.register_module()
class FocusAttention(nn.Module):
    """Gate tokens by learned saliency; provided for second-queue experiments."""

    def __init__(self, embed_dim: int = 256, keep_ratio: float = 0.5) -> None:
        super().__init__()
        self.keep_ratio = keep_ratio
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, features: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.score(features).squeeze(-1).masked_fill(~valid_mask, float("-inf"))
        keep = max(1, int(features.shape[1] * self.keep_ratio))
        keep = min(keep, int(valid_mask.sum(-1).min().item()))
        indices = scores.topk(min(keep, features.shape[1]), dim=-1).indices
        batch = torch.arange(features.shape[0], device=features.device)[:, None]
        return features[batch, indices], indices
