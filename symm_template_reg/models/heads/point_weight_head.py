"""Clean-room per-point confidence head.

Architectural reference: TAX-Pose (https://github.com/r-pad/taxpose), commit
0c4298fa0486fd09e63bf24d618a579b66ba0f18, paths
``taxpose/nets/transformer_flow_pm.py``, ``taxpose/nets/transformer_flow.py`` and
``taxpose/models/taxpose.py`` (MIT). No source text was copied. Changes: small
masked MLP returning logits, without flow, solver, PyTorch3D, or PyG coupling.
"""

from __future__ import annotations

from torch import Tensor, nn

from symm_template_reg.registry import HEADS


@HEADS.register_module()
class PointWeightHead(nn.Module):
    """Per-point confidence logit following TAX-Pose's importance-head idea."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, features: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        logits = self.network(features).squeeze(-1)
        return logits if valid_mask is None else logits.masked_fill(~valid_mask, 0.0)
