"""Rotation-invariant geometric bias adapter.

Architectural reference: RoITr (https://github.com/haoyu94/RoITr), commit
393539d6709c55b2465231cccb7b951f736a5c72, paths ``model/model.py``,
``model/transformer/ppftransformer.py`` and ``model/transformer/attention.py``
(MIT). No source text was copied. Changes: caller-provided PPF, learned pair
bias, common registry contract, and no pointops/Open3D dependencies.
"""

from __future__ import annotations

from torch import Tensor, nn

from symm_template_reg.registry import ATTENTION

from .geometric_attention import GeometricAttention


@ATTENTION.register_module()
class RotationInvariantAttention(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        self.ppf_bias = nn.Sequential(nn.Linear(4, embed_dim // 4), nn.GELU(), nn.Linear(embed_dim // 4, 1))
        self.attention = GeometricAttention(embed_dim, num_heads)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_valid_mask: Tensor,
        pair_features: Tensor,
    ) -> Tensor:
        return self.attention(query, key_value, key_valid_mask, self.ppf_bias(pair_features).squeeze(-1))
