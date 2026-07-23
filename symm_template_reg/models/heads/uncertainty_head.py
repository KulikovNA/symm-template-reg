from __future__ import annotations

from torch import Tensor, nn

from symm_template_reg.registry import HEADS


@HEADS.register_module()
class UncertaintyHead(nn.Module):
    def __init__(self, embed_dim: int = 256, output_dim: int = 6) -> None:
        super().__init__()
        self.projection = nn.Linear(embed_dim, output_dim)

    def forward(self, features: Tensor) -> Tensor:
        return self.projection(features)

