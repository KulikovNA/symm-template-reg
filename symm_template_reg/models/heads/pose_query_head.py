"""Clean-room DETR-style learnable pose queries for direct pose prediction.

Architectural reference: DETR (https://github.com/facebookresearch/detr), commit
29901c51d7fe8712168b8d0d64351170bc0f83e0, paths ``models/detr.py`` and
``models/transformer.py`` (Apache-2.0). No source text was copied. Changes:
batch-first point memory, 6D rotation + translation + uncertainty outputs, and
intermediate decoder predictions for auxiliary pose losses.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any
import warnings

import torch
from torch import Tensor, nn

from symm_template_reg.models.pose.pose_representation import PoseRepresentation  # noqa: F401
from symm_template_reg.registry import HEADS, POSE_MODULES, build_from_cfg


@HEADS.register_module()
class LegacyAbsolutePoseQueryHead(nn.Module):
    """Legacy absolute learned-query predictor retained for reproducibility."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_queries: int = 8,
        num_decoder_layers: int = 3,
        feedforward_dim: int = 512,
        uncertainty_dim: int = 6,
        dropout: float = 0.0,
        pose_representation: Mapping[str, Any] | None = None,
        pose_codec: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        warnings.warn(
            "This architecture may degenerate into an input-independent pose codebook.",
            UserWarning,
            stacklevel=2,
        )
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_decoder_layers)])
        self.query_embedding = nn.Embedding(num_queries, embed_dim)
        self.query_content = nn.Parameter(torch.zeros(num_queries, embed_dim))
        self.pose_projection = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 9)
        )
        self.logit_projection = nn.Linear(embed_dim, 1)
        self.uncertainty_projection = nn.Linear(embed_dim, uncertainty_dim)
        self.pose_representation = build_from_cfg(
            pose_representation or dict(type="PoseRepresentation"), POSE_MODULES
        )
        self.pose_codec = (
            build_from_cfg(pose_codec, POSE_MODULES)
            if pose_codec is not None
            else None
        )

    def _decode(
        self,
        features: Tensor,
        observed_centroid_C: Tensor | None,
        observed_scale: Tensor | None,
    ) -> dict[str, Tensor]:
        raw_pose = self.pose_projection(features)
        if self.pose_codec is None:
            transform = self.pose_representation(raw_pose[..., :6], raw_pose[..., 6:9])
        else:
            if observed_centroid_C is None or observed_scale is None:
                raise ValueError("pose codec requires observed centroid and scale")
            transform = self.pose_codec.decode_transform(
                raw_pose[..., :6],
                raw_pose[..., 6:9],
                observed_centroid_C,
                observed_scale,
            )
        return {
            "pose_hypotheses": transform,
            "pose_parameters_normalized": raw_pose,
            "pose_logits": self.logit_projection(features).squeeze(-1),
            "pose_uncertainty": self.uncertainty_projection(features),
        }

    def forward(
        self,
        memory: Tensor,
        memory_valid_mask: Tensor,
        observed_centroid_C: Tensor | None = None,
        observed_scale: Tensor | None = None,
    ) -> dict[str, object]:
        batch = memory.shape[0]
        position = self.query_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
        target = self.query_content.unsqueeze(0).expand(batch, -1, -1) + position
        intermediate = []
        for layer in self.layers:
            target = layer(target, memory, memory_key_padding_mask=~memory_valid_mask)
            intermediate.append(
                self._decode(target, observed_centroid_C, observed_scale)
            )
        output = dict(intermediate[-1])
        output["auxiliary_outputs"] = intermediate[:-1]
        return output


@HEADS.register_module()
class PoseQueryHead(LegacyAbsolutePoseQueryHead):
    """Backward-compatible registry/import alias for the legacy architecture."""
