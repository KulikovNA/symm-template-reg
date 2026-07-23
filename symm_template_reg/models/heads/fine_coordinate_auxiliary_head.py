"""Training-only normalized canonical-coordinate control head."""

from __future__ import annotations

from torch import Tensor, nn

from symm_template_reg.registry import HEADS


@HEADS.register_module()
class FineCanonicalCoordinateAuxiliaryHead(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )

    def forward(self, fine_point_features: Tensor) -> Tensor:
        return self.network(fine_point_features)


__all__ = ["FineCanonicalCoordinateAuxiliaryHead"]
