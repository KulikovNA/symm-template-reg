"""Clean-room bidirectional interaction pattern.

Architectural reference: RegTR (https://github.com/yewzijian/RegTR), commit
0edee25cda6b1ac1c2b0ac686dcdf2593abf25ba, paths
``src/models/transformer/transformers.py`` and ``src/models/regtr.py`` (MIT).
No source text was copied. Changes: batch-first tensors, explicit validity masks,
no KPConv/package coupling, and no numerical pose solver in inference.
"""

from __future__ import annotations

from copy import deepcopy

from torch import Tensor, nn

from symm_template_reg.registry import ATTENTION


class _ResidualAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_valid_mask: Tensor,
    ) -> Tensor:
        update = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=~key_valid_mask,
            need_weights=False,
        )[0]
        return self.norm(query + self.dropout(update))


class BidirectionalInteractionLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.observed_self = _ResidualAttention(embed_dim, num_heads, dropout)
        self.template_self = _ResidualAttention(embed_dim, num_heads, dropout)
        self.observed_cross = _ResidualAttention(embed_dim, num_heads, dropout)
        self.template_cross = _ResidualAttention(embed_dim, num_heads, dropout)
        feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.observed_ffn = deepcopy(feedforward)
        self.template_ffn = deepcopy(feedforward)
        self.observed_norm = nn.LayerNorm(embed_dim)
        self.template_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        observed: Tensor,
        template: Tensor,
        observed_mask: Tensor,
        template_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        observed = self.observed_self(observed, observed, observed_mask)
        template = self.template_self(template, template, template_mask)
        observed_before, template_before = observed, template
        observed = self.observed_cross(observed_before, template_before, template_mask)
        template = self.template_cross(template_before, observed_before, observed_mask)
        observed = self.observed_norm(observed + self.observed_ffn(observed))
        template = self.template_norm(template + self.template_ffn(template))
        return (
            observed * observed_mask.unsqueeze(-1),
            template * template_mask.unsqueeze(-1),
        )


@ATTENTION.register_module(name="RegTRInteractionTransformer")
class RegTRInteractionTransformer(nn.Module):
    """Stack of symmetric self/cross point-token interaction blocks."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.0,
        return_intermediate: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BidirectionalInteractionLayer(
                    embed_dim, num_heads, feedforward_dim, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.return_intermediate = return_intermediate

    def forward(
        self,
        observed: Tensor,
        template: Tensor,
        observed_mask: Tensor,
        template_mask: Tensor,
        observed_position: Tensor | None = None,
        template_position: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        if observed_position is not None:
            observed = observed + observed_position
        if template_position is not None:
            template = template + template_position
        intermediates = []
        for layer in self.layers:
            observed, template = layer(observed, template, observed_mask, template_mask)
            if self.return_intermediate:
                intermediates.append((observed, template))
        return observed, template, intermediates
