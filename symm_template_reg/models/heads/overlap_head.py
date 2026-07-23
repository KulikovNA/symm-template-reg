"""Clean-room overlap head.

Architectural reference: RegTR (https://github.com/yewzijian/RegTR), commit
0edee25cda6b1ac1c2b0ac686dcdf2593abf25ba,
``src/models/regtr.py`` (MIT). No source text was copied. Changes: standalone
registry module, batch-first padded masks, and a small project-local MLP.
"""

from __future__ import annotations

from torch import Tensor, nn

from symm_template_reg.registry import HEADS


@HEADS.register_module()
class OverlapHead(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        logits = self.network(features).squeeze(-1)
        return logits if valid_mask is None else logits.masked_fill(~valid_mask, 0.0)
