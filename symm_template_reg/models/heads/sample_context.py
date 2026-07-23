"""Masked sample context built from cross-conditioned observed/template tokens."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.models.geometry.point_ops import masked_max, masked_mean
from symm_template_reg.registry import HEADS


class _MaskedAttentionPool(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, tokens: Tensor, valid_mask: Tensor) -> Tensor:
        logits = self.score(tokens).squeeze(-1)
        logits = logits.masked_fill(~valid_mask, -torch.inf)
        has_valid = valid_mask.any(dim=1, keepdim=True)
        logits = torch.where(has_valid, logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=1) * valid_mask.to(tokens.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)


@HEADS.register_module()
class SampleConditionedContextAggregator(nn.Module):
    """Aggregate a view-specific context; masks are authoritative everywhere."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int | None = None,
        aggregation: str = "masked_attention_pooling",
        split_rotation_translation: bool = False,
    ) -> None:
        super().__init__()
        if aggregation not in {
            "masked_attention_pooling",
            "masked_mean_max_pooling",
        }:
            raise ValueError(
                "aggregation must be masked_attention_pooling or "
                "masked_mean_max_pooling"
            )
        self.aggregation = aggregation
        self.split_rotation_translation = bool(split_rotation_translation)
        self.observed_attention = _MaskedAttentionPool(embed_dim)
        self.template_attention = _MaskedAttentionPool(embed_dim)
        if aggregation == "masked_mean_max_pooling":
            self.observed_reduce = nn.Linear(embed_dim * 2, embed_dim)
            self.template_reduce = nn.Linear(embed_dim * 2, embed_dim)
        else:
            self.observed_reduce = nn.Identity()
            self.template_reduce = nn.Identity()
        hidden = int(hidden_dim or embed_dim * 2)
        if self.split_rotation_translation:
            self.geometric_pair_mlp = nn.Sequential(
                nn.Linear(embed_dim * 4, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            self.translation_context_mlp = nn.Sequential(
                nn.Linear(embed_dim + 4, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.pair_mlp = nn.Sequential(
                nn.Linear(embed_dim * 4 + 4, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, embed_dim),
                nn.LayerNorm(embed_dim),
            )

    def _pool(self, tokens: Tensor, mask: Tensor, observed: bool) -> Tensor:
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
            raise ValueError("tokens must be [B,N,D] with a matching [B,N] mask")
        if self.aggregation == "masked_attention_pooling":
            module = self.observed_attention if observed else self.template_attention
            return module(tokens, mask)
        pooled = torch.cat(
            (masked_mean(tokens, mask, 1), masked_max(tokens, mask, 1)), dim=-1
        )
        module = self.observed_reduce if observed else self.template_reduce
        return module(pooled)

    def forward(
        self,
        observed_tokens: Tensor,
        template_tokens: Tensor,
        observed_valid_mask: Tensor,
        template_valid_mask: Tensor,
        observed_centroid_C: Tensor,
        observed_scale: Tensor,
    ) -> dict[str, Tensor]:
        observed_context = self._pool(
            observed_tokens, observed_valid_mask, observed=True
        )
        template_context = self._pool(
            template_tokens, template_valid_mask, observed=False
        )
        scale = observed_scale.to(observed_context).clamp_min(1e-8)
        centroid = observed_centroid_C.to(observed_context)
        pose_codec_features = torch.cat(
            (centroid / scale.unsqueeze(-1), torch.log(scale).unsqueeze(-1)), dim=-1
        )
        geometric_features = torch.cat(
            (
                observed_context,
                template_context,
                observed_context - template_context,
                observed_context * template_context,
            ),
            dim=-1,
        )
        if self.split_rotation_translation:
            rotation_context = self.geometric_pair_mlp(geometric_features)
            translation_context = self.translation_context_mlp(
                torch.cat((rotation_context, pose_codec_features), dim=-1)
            )
            sample_context = rotation_context
        else:
            sample_context = self.pair_mlp(
                torch.cat((geometric_features, pose_codec_features), dim=-1)
            )
            rotation_context = sample_context
            translation_context = sample_context
        return {
            "observed_context": observed_context,
            "template_context": template_context,
            "sample_context": sample_context,
            "rotation_context": rotation_context,
            "translation_context": translation_context,
            "observed_context_norm": torch.linalg.vector_norm(
                observed_context, dim=-1
            ),
            "template_context_norm": torch.linalg.vector_norm(
                template_context, dim=-1
            ),
            "sample_context_norm": torch.linalg.vector_norm(sample_context, dim=-1),
            "rotation_context_norm": torch.linalg.vector_norm(
                rotation_context, dim=-1
            ),
            "translation_context_norm": torch.linalg.vector_norm(
                translation_context, dim=-1
            ),
        }


__all__ = ["SampleConditionedContextAggregator"]
