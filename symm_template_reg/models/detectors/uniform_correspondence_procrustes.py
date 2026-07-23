"""Single-pose registration from surface correspondences and uniform Procrustes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from symm_template_reg.models.detectors.conditioned_symm_template_reg import (
    _extract_padded_feature,
)
from symm_template_reg.models.detectors.symm_template_reg import _symmetry_availability
from symm_template_reg.models.geometry.point_ops import (
    batched_gather,
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


@MODELS.register_module()
class UniformCorrespondenceProcrustesReg(nn.Module):
    """No pose/query/confidence heads: one pose comes only from Procrustes."""

    is_joint_uniform_correspondence_model = True
    deprecated_for_new_configs = True

    def __init__(
        self,
        observed_encoder: Mapping[str, Any],
        template_encoder: Mapping[str, Any],
        interaction_transformer: Mapping[str, Any],
        dual_stream_geometry_encoder: Mapping[str, Any],
        correspondence_head: Mapping[str, Any],
        weighted_procrustes: Mapping[str, Any],
        max_observed_tokens: int = 256,
        max_template_tokens: int = 512,
        point_weight_head: Mapping[str, Any] | None = None,
        weighting_mode: str = "uniform",
    ) -> None:
        super().__init__()
        if weighting_mode not in {"uniform", "learned_confidence_ablation"}:
            raise ValueError(f"unsupported weighting_mode: {weighting_mode}")
        if weighting_mode == "uniform" and point_weight_head is not None:
            raise ValueError("uniform baseline forbids point_weight_head")
        mesh_head_types = {
            "SurfaceConstrainedCorrespondenceHeadV2",
            "SoftCoarseLocalSurfaceCorrespondenceHead",
            "CanonicalCoordinateRegressionControl",
        }
        if correspondence_head.get("type") not in mesh_head_types and str(correspondence_head.get("output_mode")) != "soft_template_surface_matching":
            raise ValueError("joint baseline requires soft_template_surface_matching")
        self.observed_encoder = build_from_cfg(observed_encoder, BACKBONES)
        self.template_encoder = build_from_cfg(template_encoder, BACKBONES)
        self.interaction_transformer = build_from_cfg(interaction_transformer, ATTENTION)
        self.dual_stream_geometry_encoder = build_from_cfg(
            dual_stream_geometry_encoder, GEOMETRY_MODULES
        )
        self.correspondence_head = build_from_cfg(correspondence_head, HEADS)
        self.dense_observed_fine_projection: nn.Module
        self.fine_template_projection: nn.Module
        if bool(getattr(self.correspondence_head, "uses_separate_fine_features", False)):
            embed_dim = int(getattr(self.correspondence_head.fine_feature_adapter, "embed_dim"))
            self.dense_observed_fine_projection = nn.Linear(embed_dim, embed_dim)
            self.fine_template_projection = nn.Linear(embed_dim, embed_dim)
            nn.init.eye_(self.dense_observed_fine_projection.weight)
            nn.init.zeros_(self.dense_observed_fine_projection.bias)
            nn.init.eye_(self.fine_template_projection.weight)
            nn.init.zeros_(self.fine_template_projection.bias)
        else:
            self.dense_observed_fine_projection = nn.Identity()
            self.fine_template_projection = nn.Identity()
        self.weighted_procrustes = build_from_cfg(weighted_procrustes, POSE_MODULES)
        self.point_weight_head = (
            build_from_cfg(point_weight_head, HEADS)
            if point_weight_head is not None
            else None
        )
        self.weighting_mode = weighting_mode
        self.max_observed_tokens = int(max_observed_tokens)
        self.max_template_tokens = int(max_template_tokens)

    def forward(
        self,
        batch: Mapping[str, Any] | None = None,
        *,
        observed: Any | None = None,
        template: Any | None = None,
        symmetry_metadata: Any | None = None,
    ) -> RegistrationPrediction:
        if batch is not None:
            observed = batch["observed"]
            template = batch["template"]
            symmetry_metadata = batch.get("template_symmetry_metadata", symmetry_metadata)
        if observed is None or template is None:
            raise ValueError("observed and template inputs are required")
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
            observed_tokens, template_tokens, observed_mask, template_mask
        )
        observed_normals = _extract_padded_feature(observed, "normals_C")
        template_normals = _extract_padded_feature(template, "normals_O")
        observed_normals_dense = (
            None
            if observed_normals is None
            else observed_normals.to(observed_encoded.points)
        )
        if observed_normals is not None:
            observed_normals = batched_gather(
                observed_normals_dense, observed_indices
            )
        if template_normals is not None:
            template_normals = batched_gather(
                template_normals.to(template_points), template_indices
            )
        observed_matching = self.dual_stream_geometry_encoder(
            observed_tokens, observed_points, observed_mask, observed_normals
        )["matching_features"]
        template_matching = self.dual_stream_geometry_encoder(
            template_tokens, template_points, template_mask, template_normals
        )["matching_features"]
        observed_interaction_dense = nearest_interpolate(
            observed_encoded.points,
            observed_points,
            observed_matching,
            observed_mask,
        )
        observed_dense = observed_encoded.point_features + observed_interaction_dense
        observed_dense = observed_dense * observed_encoded.valid_mask.unsqueeze(-1)
        if bool(getattr(self.correspondence_head, "requires_template_mesh", False)):
            if batch is None:
                raise ValueError("SurfaceConstrainedCorrespondenceHeadV2 requires collated template mesh")
            teacher_target = None
            teacher_forcing_enabled = (
                self.training
                or bool(
                    getattr(
                        self.correspondence_head,
                        "teacher_forcing_during_evaluation",
                        False,
                    )
                )
            ) and float(
                getattr(self.correspondence_head, "teacher_forcing_probability", 0.0)
            ) > 0.0
            if teacher_forcing_enabled:
                target_payload = batch["gt"].get("points_O_corresponding")
                if target_payload is not None:
                    teacher_target = (
                        target_payload.to_padded()["points"]
                        if hasattr(target_payload, "to_padded")
                        else target_payload
                    )
            head_kwargs = dict(
                template_mesh_vertices_O=batch["template_mesh_vertices_O"],
                template_mesh_faces=batch["template_mesh_faces"],
                teacher_forcing_target_points_O=teacher_target,
            )
            if bool(
                getattr(self.correspondence_head, "is_surface_constrained_v2", False)
            ):
                head_kwargs.update(
                    teacher_forcing_symmetry_metadata=batch.get(
                        "template_symmetry_metadata"
                    ),
                    teacher_forcing_effective_symmetry_groups=batch["gt"].get(
                        "effective_symmetry_group"
                    ),
                )
                if bool(
                    getattr(
                        self.correspondence_head,
                        "uses_separate_fine_features",
                        False,
                    )
                ):
                    head_kwargs.update(
                        original_dense_observed_features=(
                            self.dense_observed_fine_projection(
                                observed_encoded.point_features
                            )
                        ),
                        interpolated_observed_interaction_features=(
                            observed_interaction_dense
                        ),
                        observed_points_C=observed_encoded.points,
                        observed_normals_C=observed_normals_dense,
                        fine_template_interaction_features=(
                            self.fine_template_projection(template_matching)
                        ),
                    )
            head_output = self.correspondence_head(
                observed_dense, template_matching, template_points,
                observed_encoded.valid_mask, template_mask,
                **head_kwargs,
            )
            correspondence_points = head_output["points_O"]
            matching_confidence = head_output["confidence"]
            logits = head_output["logits"]
            correspondence_auxiliary = head_output["auxiliary"]
        else:
            correspondence_points, matching_confidence, logits = self.correspondence_head(
                observed_dense, template_matching, template_points,
                observed_encoded.valid_mask, template_mask,
            )
            correspondence_auxiliary = None
        valid = observed_encoded.valid_mask
        uniform = valid.to(correspondence_points.dtype)
        uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
        if self.weighting_mode == "uniform":
            weights = uniform
        else:
            assert self.point_weight_head is not None
            learned = matching_confidence * self.point_weight_head(
                observed_dense, valid
            ).sigmoid()
            learned = learned * valid
            weights = learned / learned.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # This research baseline explicitly solves in float32 even when a
        # caller accidentally supplies another floating dtype.
        solution = self.weighted_procrustes.solve(
            correspondence_points.float(),
            observed_encoded.points.float(),
            weights.float(),
            valid,
        )
        pose = solution["transform"].to(correspondence_points)
        batch_size = pose.shape[0]
        zeros_observed = correspondence_points.new_zeros(valid.shape)
        zeros_template = correspondence_points.new_zeros(template_encoded.valid_mask.shape)
        prediction = RegistrationPrediction(
            pose_hypotheses=pose.unsqueeze(1),
            pose_logits=pose.new_zeros((batch_size, 1)),
            pose_uncertainty=pose.new_zeros((batch_size, 1, 0)),
            observed_overlap_logits=zeros_observed,
            template_visibility_logits=zeros_template,
            correspondence_points_O=correspondence_points,
            correspondence_confidence=weights,
            observed_region_logits=None,
            active_region_logits=None,
            insufficient_information_logit=pose.new_zeros((batch_size, 1)),
            observed_valid_mask=valid,
            template_valid_mask=template_encoded.valid_mask,
            symmetry_available=_symmetry_availability(
                symmetry_metadata, batch_size, pose.device
            ),
            base_pose=pose,
            correspondence_pose=pose,
            correspondence_pose_diagnostics=solution,
            base_pose_source="uniform_weighted_procrustes",
            pose_hypotheses_enabled=False,
            weighting_mode=self.weighting_mode,
            correspondence_logits=logits,
            correspondence_auxiliary=correspondence_auxiliary,
            correspondence_feature_path={
                "observed_encoder_features": observed_encoded.point_features,
                "sampled_interaction_tokens": observed_matching,
                "sampled_interaction_points_C": observed_points,
                "upsampled_per_point_features": observed_dense,
                "coarse_patch_features": correspondence_auxiliary.get(
                    "coarse_patch_features"
                ) if correspondence_auxiliary is not None else None,
                "fine_point_features": correspondence_auxiliary.get(
                    "fine_point_features"
                ) if correspondence_auxiliary is not None else None,
                "fine_triangle_features": correspondence_auxiliary.get(
                    "fine_triangle_features"
                ) if correspondence_auxiliary is not None else None,
                "observed_token_indices": observed_indices,
            },
        )
        prediction.validate()
        return prediction


__all__ = ["UniformCorrespondenceProcrustesReg"]
