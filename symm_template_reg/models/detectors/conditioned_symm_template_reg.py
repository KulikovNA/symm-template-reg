"""Sample-conditioned base pose with camera-frame residual hypotheses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from symm_template_reg.models.detectors.symm_template_reg import (
    _symmetry_availability,
)
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
    MODELS,
    POSE_MODULES,
    build_from_cfg,
)


def _extract_padded_feature(value: Any, name: str) -> Tensor | None:
    """Read an optional aligned feature without making it a model target."""

    if hasattr(value, "to_padded"):
        padded = value.to_padded()
        features = padded.get("features", {}) if isinstance(padded, Mapping) else {}
        result = features.get(name)
    elif isinstance(value, Mapping):
        result = value.get(name)
        if result is None and isinstance(value.get("features"), Mapping):
            result = value["features"].get(name)
    else:
        result = None
    if result is not None and result.ndim == 2:
        result = result.unsqueeze(0)
    return result


@MODELS.register_module()
class ConditionedSymmTemplateReg(nn.Module):
    """A direct conditioned pose path; residual modes cannot output absolute poses."""

    def __init__(
        self,
        observed_encoder: Mapping[str, Any],
        template_encoder: Mapping[str, Any],
        interaction_transformer: Mapping[str, Any],
        dual_stream_geometry_encoder: Mapping[str, Any],
        sample_context_aggregator: Mapping[str, Any],
        base_pose_head: Mapping[str, Any],
        residual_pose_head: Mapping[str, Any],
        correspondence_head: Mapping[str, Any],
        overlap_head: Mapping[str, Any],
        point_weight_head: Mapping[str, Any],
        template_visibility_head: Mapping[str, Any] | None = None,
        symmetry_head: Mapping[str, Any] | None = None,
        weighted_procrustes: Mapping[str, Any] | None = None,
        embed_dim: int = 256,
        max_observed_tokens: int = 256,
        max_template_tokens: int = 256,
        base_pose_source: str = "direct_context",
        correspondence_only: bool = False,
    ) -> None:
        super().__init__()
        self.observed_encoder = build_from_cfg(observed_encoder, BACKBONES)
        self.template_encoder = build_from_cfg(template_encoder, BACKBONES)
        self.interaction_transformer = build_from_cfg(
            interaction_transformer, ATTENTION
        )
        self.dual_stream_geometry_encoder = build_from_cfg(
            dual_stream_geometry_encoder, GEOMETRY_MODULES
        )
        self.sample_context_aggregator = build_from_cfg(
            sample_context_aggregator, HEADS
        )
        self.base_pose_head = build_from_cfg(base_pose_head, HEADS)
        self.residual_pose_head = build_from_cfg(residual_pose_head, HEADS)
        self.correspondence_head = build_from_cfg(correspondence_head, HEADS)
        self.observed_overlap_head = build_from_cfg(overlap_head, HEADS)
        self.template_visibility_head = build_from_cfg(
            template_visibility_head or overlap_head, HEADS
        )
        self.point_weight_head = build_from_cfg(point_weight_head, HEADS)
        self.symmetry_head = (
            build_from_cfg(symmetry_head, HEADS) if symmetry_head is not None else None
        )
        self.weighted_procrustes = (
            build_from_cfg(weighted_procrustes, POSE_MODULES)
            if weighted_procrustes is not None
            else None
        )
        if base_pose_source not in {
            "direct_context",
            "weighted_procrustes",
            "procrustes_plus_direct_residual",
        }:
            raise ValueError(f"unsupported base_pose_source: {base_pose_source}")
        if base_pose_source != "direct_context" and self.weighted_procrustes is None:
            raise ValueError(f"{base_pose_source} requires weighted_procrustes")
        self.base_pose_source = base_pose_source
        self.correspondence_only = bool(correspondence_only)
        if self.correspondence_only and base_pose_source != "weighted_procrustes":
            raise ValueError("correspondence_only requires weighted_procrustes base source")
        if getattr(self.residual_pose_head, "num_hypotheses", 0) < 1:
            raise ValueError("conditioned model requires residual num_hypotheses >= 1")
        if hasattr(self.residual_pose_head, "absolute_pose_projection"):
            raise ValueError("residual head must not expose absolute_pose_projection")
        self.max_observed_tokens = int(max_observed_tokens)
        self.max_template_tokens = int(max_template_tokens)
        self.cloud_type_embedding = nn.Parameter(torch.zeros(2, embed_dim))
        self.insufficient_information = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1)
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
        observed_points, observed_tokens, observed_mask, observed_indices = select_tokens(
            observed_encoded.points,
            observed_encoded.point_features,
            observed_encoded.valid_mask,
            self.max_observed_tokens,
        )
        template_points, template_tokens, template_mask, template_indices = select_tokens(
            template_encoded.points,
            template_encoded.point_features,
            template_encoded.valid_mask,
            self.max_template_tokens,
        )
        observed_tokens, template_tokens, _ = self.interaction_transformer(
            observed_tokens,
            template_tokens,
            observed_mask,
            template_mask,
        )
        observed_pose_tokens = self.dual_stream_geometry_encoder.pose_features(
            observed_tokens, observed_mask
        )
        template_pose_tokens = self.dual_stream_geometry_encoder.pose_features(
            template_tokens, template_mask
        )
        matching_geometric_only = bool(
            self.dual_stream_geometry_encoder.matching_geometric_only
        )
        if matching_geometric_only:
            observed_normals = _extract_padded_feature(observed, "normals_C")
            template_normals = _extract_padded_feature(template, "normals_O")
            if observed_normals is not None:
                observed_normals = batched_gather(
                    observed_normals.to(observed_points), observed_indices
                )
            if template_normals is not None:
                template_normals = batched_gather(
                    template_normals.to(template_points), template_indices
                )
            observed_geometry = self.dual_stream_geometry_encoder.matching_geometry(
                observed_points, observed_mask, observed_normals
            )
            template_geometry = self.dual_stream_geometry_encoder.matching_geometry(
                template_points, template_mask, template_normals
            )
            observed_cross_matching, template_cross_matching, _ = (
                self.interaction_transformer(
                    observed_geometry,
                    template_geometry,
                    observed_mask,
                    template_mask,
                )
            )
            observed_matching_tokens = (
                self.dual_stream_geometry_encoder.finalize_matching(
                    observed_cross_matching, observed_geometry, observed_mask
                )
            )
            template_matching_tokens = (
                self.dual_stream_geometry_encoder.finalize_matching(
                    template_cross_matching, template_geometry, template_mask
                )
            )
        else:
            observed_streams = self.dual_stream_geometry_encoder(
                observed_tokens, observed_points, observed_mask
            )
            template_streams = self.dual_stream_geometry_encoder(
                template_tokens, template_points, template_mask
            )
            observed_matching_tokens = observed_streams["matching_features"]
            template_matching_tokens = template_streams["matching_features"]
        pose_codec = self.base_pose_head.pose_codec
        codec_context = pose_codec.context(
            observed_encoded.points, observed_encoded.valid_mask
        )
        centroid = codec_context.observed_centroid_C
        scale = codec_context.observed_scale
        if isinstance(context_override, Mapping):
            centroid = context_override["observed_centroid_C"].to(centroid)
            scale = context_override["observed_scale"].to(scale)
        context = self.sample_context_aggregator(
            observed_pose_tokens,
            template_pose_tokens,
            observed_mask,
            template_mask,
            centroid,
            scale,
        )
        observed_matching_dense = nearest_interpolate(
            observed_encoded.points,
            observed_points,
            observed_matching_tokens,
            observed_mask,
        )
        template_matching_dense = nearest_interpolate(
            template_encoded.points,
            template_points,
            template_matching_tokens,
            template_mask,
        )
        observed_dense = (
            observed_matching_dense
            if matching_geometric_only
            else observed_encoded.point_features + observed_matching_dense
        )
        template_dense = (
            template_matching_dense
            if matching_geometric_only
            else template_encoded.point_features + template_matching_dense
        )
        observed_dense = observed_dense * observed_encoded.valid_mask.unsqueeze(-1)
        template_dense = template_dense * template_encoded.valid_mask.unsqueeze(-1)
        correspondence_points, soft_confidence, _ = self.correspondence_head(
            observed_dense,
            template_matching_tokens,
            template_points,
            observed_encoded.valid_mask,
            template_mask,
        )
        weight_logits = self.point_weight_head(
            observed_dense, observed_encoded.valid_mask
        )
        correspondence_confidence = soft_confidence * weight_logits.sigmoid()
        correspondence_solution = (
            self.weighted_procrustes.solve(
                correspondence_points,
                observed_encoded.points,
                correspondence_confidence,
                observed_encoded.valid_mask,
            )
            if self.weighted_procrustes is not None
            else None
        )
        correspondence_pose = (
            correspondence_solution["transform"]
            if correspondence_solution is not None
            else None
        )
        head_kwargs = {
            "rotation_context": context["rotation_context"],
            "translation_context": context["translation_context"],
        }
        if self.base_pose_source == "direct_context":
            base = self.base_pose_head(
                context["sample_context"], centroid, scale, **head_kwargs
            )
        elif self.base_pose_source == "weighted_procrustes":
            assert correspondence_pose is not None
            base = {
                "base_T_C_from_O": correspondence_pose,
                "base_pose_parameters_normalized": self.base_pose_head.pose_codec.encode_transform(
                    correspondence_pose, centroid, scale
                ),
                "base_uncertainty": None,
                "base_correction_transform": None,
            }
        else:
            assert correspondence_pose is not None
            base = self.base_pose_head(
                context["sample_context"], centroid, scale,
                reference_pose=correspondence_pose, **head_kwargs,
            )
        memory = torch.cat(
            (
                observed_pose_tokens + self.cloud_type_embedding[0],
                template_pose_tokens + self.cloud_type_embedding[1],
            ),
            dim=1,
        )
        memory_mask = torch.cat((observed_mask, template_mask), dim=1)
        residual = self.residual_pose_head(
            context["sample_context"],
            memory,
            memory_mask,
            base["base_T_C_from_O"],
            scale,
        )
        observed_overlap = self.observed_overlap_head(
            observed_dense, observed_encoded.valid_mask
        )
        template_visibility = self.template_visibility_head(
            template_dense, template_encoded.valid_mask
        )
        observed_global = masked_mean(
            observed_dense, observed_encoded.valid_mask, 1
        )
        template_global = masked_mean(
            template_dense, template_encoded.valid_mask, 1
        )
        if self.symmetry_head is None:
            observed_region, active_region = None, None
        else:
            observed_region, active_region = self.symmetry_head(
                observed_dense,
                observed_global,
                template_global,
                observed_encoded.valid_mask,
            )
        prediction = RegistrationPrediction(
            pose_hypotheses=residual["pose_hypotheses"],
            pose_logits=residual["pose_logits"],
            pose_uncertainty=residual["pose_uncertainty"],
            observed_overlap_logits=observed_overlap,
            template_visibility_logits=template_visibility,
            correspondence_points_O=correspondence_points,
            correspondence_confidence=correspondence_confidence,
            observed_region_logits=observed_region,
            active_region_logits=active_region,
            insufficient_information_logit=self.insufficient_information(
                torch.cat((observed_global, template_global), dim=-1)
            ),
            observed_valid_mask=observed_encoded.valid_mask,
            template_valid_mask=template_encoded.valid_mask,
            auxiliary_outputs=residual["auxiliary_outputs"],
            symmetry_available=_symmetry_availability(
                symmetry_metadata,
                observed_encoded.points.shape[0],
                observed_encoded.points.device,
            ),
            observed_centroid_C=centroid,
            observed_scale=scale,
            base_pose=base["base_T_C_from_O"],
            base_pose_parameters_normalized=base[
                "base_pose_parameters_normalized"
            ],
            base_uncertainty=base["base_uncertainty"],
            base_correction_transform=base.get("base_correction_transform"),
            residual_pose_parameters=residual["residual_pose_parameters"],
            residual_transforms=residual["residual_transforms"],
            correspondence_pose=correspondence_pose,
            correspondence_pose_diagnostics=correspondence_solution,
            context_diagnostics=context,
            base_pose_source=self.base_pose_source,
            pose_hypotheses_enabled=not self.correspondence_only,
        )
        prediction.validate()
        return prediction


__all__ = ["ConditionedSymmTemplateReg"]
