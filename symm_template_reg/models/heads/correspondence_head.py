"""Soft feature matching with a learned residual correspondence correction.

Architectural reference: RegTR (https://github.com/yewzijian/RegTR), commit
0edee25cda6b1ac1c2b0ac686dcdf2593abf25ba,
``src/models/regtr.py`` (MIT). No source text was copied. Changes: soft template
matching, bounded residual coordinates, explicit masks, and no pose solver.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from symm_template_reg.registry import HEADS


@HEADS.register_module()
class CorrespondenceHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        residual_scale_m: float = 0.02,
        output_mode: str = "soft_with_bounded_residual",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if output_mode not in {
            "soft_with_bounded_residual",
            "soft_template_surface_matching",
        }:
            raise ValueError(f"unsupported correspondence output_mode: {output_mode}")
        self.output_mode = output_mode
        if float(temperature) <= 0:
            raise ValueError("correspondence temperature must be positive")
        self.temperature = float(temperature)
        self.residual_scale_m = residual_scale_m
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.residual = (
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 3),
                nn.Tanh(),
            )
            if output_mode == "soft_with_bounded_residual"
            else None
        )

    def forward(
        self,
        observed_features: Tensor,
        template_features: Tensor,
        template_points: Tensor,
        observed_mask: Tensor,
        template_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits = torch.matmul(
            self.query(observed_features), self.key(template_features).transpose(-2, -1)
        ) / (math.sqrt(observed_features.shape[-1]) * self.temperature)
        logits = logits.masked_fill(~template_mask[:, None, :], float("-inf"))
        weights = logits.softmax(-1)
        if self.output_mode == "soft_template_surface_matching":
            # Convex soft matching over real template surface samples, with no
            # free coordinate residual.  Unlike a hard argmax this retains
            # small geometric differences required by differentiable SVD;
            # the explicit surface loss measures and penalizes any chordal
            # interpolation that falls inside the mesh rather than on it.
            points = torch.matmul(weights, template_points)
        else:
            points = torch.matmul(weights, template_points)
            assert self.residual is not None
            points = points + self.residual(observed_features) * self.residual_scale_m
        confidence = weights.max(-1).values
        points = points * observed_mask.unsqueeze(-1)
        confidence = confidence * observed_mask
        return points, confidence, logits
