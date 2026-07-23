"""Modular direct multi-hypothesis fragment-to-template registration model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from symm_template_reg.models.geometry.point_ops import (
    batched_gather,
    masked_mean,
    nearest_interpolate,
    select_tokens,
)
from symm_template_reg.models.structures import RegistrationPrediction
from symm_template_reg.registry import (
    ATTENTION,
    BACKBONES,
    GEOMETRY_MODULES,
    HEADS,
    MATCHERS,
    MODELS,
    SYMMETRY_MODULES,
    build_from_cfg,
)


def _symmetry_availability(meta: Any, batch_size: int, device: torch.device) -> Tensor:
    if meta is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    if isinstance(meta, Tensor):
        if meta.numel() != batch_size:
            raise ValueError(
                f"symmetry availability has {meta.numel()} entries for batch size {batch_size}"
            )
        return meta.to(device=device, dtype=torch.bool).reshape(batch_size)
    if isinstance(meta, Mapping):
        value = meta.get("symmetry_available", False)
        if isinstance(value, Tensor):
            if value.numel() != batch_size:
                raise ValueError(
                    f"symmetry availability has {value.numel()} entries for batch size {batch_size}"
                )
            return value.to(device=device, dtype=torch.bool).reshape(batch_size)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != batch_size:
                raise ValueError(
                    f"symmetry availability has {len(value)} entries for batch size {batch_size}"
                )
            return torch.tensor([bool(item) for item in value], dtype=torch.bool, device=device)
        return torch.full((batch_size,), bool(value), dtype=torch.bool, device=device)
    if isinstance(meta, Sequence):
        values = [
            bool(item.get("symmetry_available", False))
            if isinstance(item, Mapping)
            else bool(item)
            if isinstance(item, (bool, int))
            else item is not None
            for item in meta
        ]
        if len(values) != batch_size:
            raise ValueError(
                f"symmetry metadata has {len(values)} entries for batch size {batch_size}"
            )
        return torch.tensor(values, dtype=torch.bool, device=device)
    return torch.full((batch_size,), bool(meta), dtype=torch.bool, device=device)


@MODELS.register_module()
class SymmTemplateReg(nn.Module):
    """Direct K-pose baseline; numerical pose solving is not part of inference."""

    def __init__(
        self,
        observed_encoder: Mapping[str, Any],
        template_encoder: Mapping[str, Any],
        interaction_transformer: Mapping[str, Any],
        correspondence_head: Mapping[str, Any],
        overlap_head: Mapping[str, Any],
        pose_head: Mapping[str, Any],
        point_weight_head: Mapping[str, Any],
        symmetry_head: Mapping[str, Any] | None = None,
        symmetry_expander: Mapping[str, Any] | None = None,
        geometric_embedding: Mapping[str, Any] | None = None,
        coarse_matcher: Mapping[str, Any] | None = None,
        template_visibility_head: Mapping[str, Any] | None = None,
        embed_dim: int = 256,
        max_observed_tokens: int = 256,
        max_template_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.observed_encoder = build_from_cfg(observed_encoder, BACKBONES)
        self.template_encoder = build_from_cfg(template_encoder, BACKBONES)
        self.interaction_transformer = build_from_cfg(interaction_transformer, ATTENTION)
        self.correspondence_head = build_from_cfg(correspondence_head, HEADS)
        self.observed_overlap_head = build_from_cfg(overlap_head, HEADS)
        self.template_visibility_head = build_from_cfg(
            template_visibility_head or overlap_head, HEADS
        )
        self.pose_head = build_from_cfg(pose_head, HEADS)
        self.point_weight_head = build_from_cfg(point_weight_head, HEADS)
        self.symmetry_head = (
            build_from_cfg(symmetry_head, HEADS) if symmetry_head is not None else None
        )
        self.symmetry_expander = (
            build_from_cfg(symmetry_expander, SYMMETRY_MODULES)
            if symmetry_expander is not None
            else None
        )
        self.geometric_embedding = (
            build_from_cfg(geometric_embedding, GEOMETRY_MODULES)
            if geometric_embedding is not None
            else None
        )
        self.coarse_matcher = (
            build_from_cfg(coarse_matcher, MATCHERS) if coarse_matcher is not None else None
        )
        self.max_observed_tokens = max_observed_tokens
        self.max_template_tokens = max_template_tokens
        self.cloud_type_embedding = nn.Parameter(torch.zeros(2, embed_dim))
        self.insufficient_information = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1)
        )

    def expand_gt_pose_set(self, T_C_from_O: Tensor, symmetry: Any, **kwargs: Any) -> Any:
        """Expand GT through the configured symmetry contract for loss/assignment code."""

        if self.symmetry_expander is None:
            raise RuntimeError("no symmetry_expander is configured")
        return self.symmetry_expander(T_C_from_O, symmetry, **kwargs)

    @staticmethod
    def build_symmetry_targets(
        points_O: Tensor, T_C_from_O: Tensor, symmetry: Any, **kwargs: Any
    ) -> Any:
        """Expose the same production target builder used by Dataset/debug."""

        from symm_template_reg.models.symmetry.targets import build_fragment_symmetry_targets

        return build_fragment_symmetry_targets(
            points_O, symmetry, base_pose=T_C_from_O, **kwargs
        )

    def forward(
        self,
        batch: Mapping[str, Any] | None = None,
        *,
        observed: Any | None = None,
        template: Any | None = None,
        symmetry_metadata: Any | None = None,
    ) -> RegistrationPrediction:
        context_override = None
        if batch is not None:
            observed = batch["observed"]
            template = batch["template"]
            symmetry_metadata = batch.get(
                "template_symmetry_metadata",
                batch.get("symmetry_metadata", batch.get("meta", symmetry_metadata)),
            )
            context_override = batch.get("pose_codec_context_override")
        if observed is None or template is None:
            raise ValueError("both observed and template point clouds are required")

        observed_encoded = self.observed_encoder(observed)
        template_encoded = self.template_encoder(template)
        observed_features = observed_encoded.point_features
        template_features = template_encoded.point_features
        if self.geometric_embedding is not None:
            observed_features = observed_features + self.geometric_embedding(
                observed_encoded.points, observed_encoded.valid_mask
            )
            template_features = template_features + self.geometric_embedding(
                template_encoded.points, template_encoded.valid_mask
            )

        observed_token_points, observed_tokens, observed_token_mask, observed_indices = select_tokens(
            observed_encoded.points,
            observed_features,
            observed_encoded.valid_mask,
            self.max_observed_tokens,
        )
        template_token_points, template_tokens, template_token_mask, template_indices = select_tokens(
            template_encoded.points,
            template_features,
            template_encoded.valid_mask,
            self.max_template_tokens,
        )
        observed_tokens, template_tokens, _ = self.interaction_transformer(
            observed_tokens,
            template_tokens,
            observed_token_mask,
            template_token_mask,
        )
        if self.coarse_matcher is not None:
            coarse_scores = self.coarse_matcher(
                observed_tokens,
                template_tokens,
                observed_token_mask,
                template_token_mask,
            )
            observed_update = torch.matmul(
                coarse_scores / coarse_scores.sum(-1, keepdim=True).clamp_min(1e-8),
                template_tokens,
            )
            reverse_scores = coarse_scores.transpose(-2, -1)
            template_update = torch.matmul(
                reverse_scores / reverse_scores.sum(-1, keepdim=True).clamp_min(1e-8),
                observed_tokens,
            )
            observed_tokens = (observed_tokens + observed_update) * observed_token_mask.unsqueeze(-1)
            template_tokens = (template_tokens + template_update) * template_token_mask.unsqueeze(-1)
        observed_dense = observed_features + nearest_interpolate(
            observed_encoded.points,
            observed_token_points,
            observed_tokens,
            observed_token_mask,
        )
        template_dense = template_features + nearest_interpolate(
            template_encoded.points,
            template_token_points,
            template_tokens,
            template_token_mask,
        )
        observed_dense = observed_dense * observed_encoded.valid_mask.unsqueeze(-1)
        template_dense = template_dense * template_encoded.valid_mask.unsqueeze(-1)
        observed_global = masked_mean(observed_dense, observed_encoded.valid_mask, 1)
        template_global = masked_mean(template_dense, template_encoded.valid_mask, 1)

        correspondence_points, soft_confidence, _ = self.correspondence_head(
            observed_dense,
            template_tokens,
            template_token_points,
            observed_encoded.valid_mask,
            template_token_mask,
        )
        observed_overlap = self.observed_overlap_head(
            observed_dense, observed_encoded.valid_mask
        )
        template_visibility = self.template_visibility_head(
            template_dense, template_encoded.valid_mask
        )
        weight_logits = self.point_weight_head(observed_dense, observed_encoded.valid_mask)
        correspondence_confidence = soft_confidence * weight_logits.sigmoid()
        if self.symmetry_head is None:
            observed_region, active_region = None, None
        else:
            observed_region, active_region = self.symmetry_head(
                observed_dense,
                observed_global,
                template_global,
                observed_encoded.valid_mask,
            )

        memory = torch.cat(
            (
                observed_tokens + self.cloud_type_embedding[0],
                template_tokens + self.cloud_type_embedding[1],
            ),
            dim=1,
        )
        memory_mask = torch.cat((observed_token_mask, template_token_mask), dim=1)
        pose_codec = getattr(self.pose_head, "pose_codec", None)
        if pose_codec is None:
            pose_context = None
            pose_output = self.pose_head(memory, memory_mask)
        else:
            pose_context = pose_codec.context(
                observed_encoded.points, observed_encoded.valid_mask
            )
            if isinstance(context_override, Mapping):
                pose_context = type(pose_context)(
                    context_override["observed_centroid_C"].to(
                        pose_context.observed_centroid_C
                    ),
                    context_override["observed_scale"].to(
                        pose_context.observed_scale
                    ),
                )
            pose_output = self.pose_head(
                memory,
                memory_mask,
                pose_context.observed_centroid_C,
                pose_context.observed_scale,
            )
        insufficient = self.insufficient_information(
            torch.cat((observed_global, template_global), dim=-1)
        )
        symmetry_available = _symmetry_availability(
            symmetry_metadata, observed_encoded.points.shape[0], observed_encoded.points.device
        )
        prediction = RegistrationPrediction(
            pose_hypotheses=pose_output["pose_hypotheses"],
            pose_logits=pose_output["pose_logits"],
            pose_uncertainty=pose_output["pose_uncertainty"],
            observed_overlap_logits=observed_overlap,
            template_visibility_logits=template_visibility,
            correspondence_points_O=correspondence_points,
            correspondence_confidence=correspondence_confidence,
            observed_region_logits=observed_region,
            active_region_logits=active_region,
            insufficient_information_logit=insufficient,
            observed_valid_mask=observed_encoded.valid_mask,
            template_valid_mask=template_encoded.valid_mask,
            auxiliary_outputs=pose_output["auxiliary_outputs"],
            symmetry_available=symmetry_available,
            observed_centroid_C=(
                pose_context.observed_centroid_C if pose_context is not None else None
            ),
            observed_scale=(pose_context.observed_scale if pose_context is not None else None),
        )
        prediction.validate()
        return prediction
