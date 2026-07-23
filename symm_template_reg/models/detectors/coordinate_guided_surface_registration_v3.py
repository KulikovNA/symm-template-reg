"""Clean active-only coordinate registration model for scratch training."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from symm_template_reg.models.detectors.conditioned_symm_template_reg import (
    _extract_padded_feature,
)
from symm_template_reg.engine.static_geometry_cache import StaticGeometryCache
from symm_template_reg.models.backbones.simple_point_encoder import as_padded_points
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


LEGACY_MODULE_TOKENS = (
    "pose_query", "ranking", "patch", "triangle", "barycentric",
    "region", "overlap", "visibility", "confidence", "insufficient",
)


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@MODELS.register_module()
class CoordinateGuidedSurfaceRegistrationV3(nn.Module):
    """Only the trainable raw-q_aux graph plus parameter-free Procrustes."""

    is_joint_uniform_correspondence_model = True
    is_clean_coordinate_guided_v3 = True

    def __init__(
        self,
        observed_encoder: Mapping[str, Any],
        template_encoder: Mapping[str, Any],
        interaction_transformer: Mapping[str, Any],
        dual_stream_geometry_encoder: Mapping[str, Any],
        fine_feature_adapter: Mapping[str, Any],
        canonical_coordinate_head: Mapping[str, Any],
        weighted_procrustes: Mapping[str, Any],
        embed_dim: int = 256,
        max_observed_tokens: int = 256,
        max_template_tokens: int = 512,
        final_coordinate_initialization_std: float = 1e-3,
        static_geometry_cache: Mapping[str, Any] | bool = False,
        shared_template_encoding: bool = False,
    ) -> None:
        super().__init__()
        self.observed_encoder = build_from_cfg(observed_encoder, BACKBONES)
        self.template_encoder = build_from_cfg(template_encoder, BACKBONES)
        self.interaction_transformer = build_from_cfg(interaction_transformer, ATTENTION)
        self.dual_stream_geometry_encoder = build_from_cfg(
            dual_stream_geometry_encoder, GEOMETRY_MODULES
        )
        self.dense_observed_fine_projection = nn.Linear(embed_dim, embed_dim)
        self.fine_template_projection = nn.Linear(embed_dim, embed_dim)
        self.template_context_projection = nn.Linear(embed_dim, embed_dim)
        self.fine_feature_adapter = build_from_cfg(
            fine_feature_adapter, GEOMETRY_MODULES
        )
        self.canonical_coordinate_head = build_from_cfg(
            canonical_coordinate_head, HEADS
        )
        self.weighted_procrustes = build_from_cfg(weighted_procrustes, POSE_MODULES)
        self.max_observed_tokens = int(max_observed_tokens)
        self.max_template_tokens = int(max_template_tokens)
        self.final_coordinate_initialization_std = float(
            final_coordinate_initialization_std
        )
        self.shared_template_encoding = bool(shared_template_encoding)
        cache_config = {} if static_geometry_cache is True else static_geometry_cache
        self.static_geometry_cache_enabled = bool(static_geometry_cache)
        self._static_geometry_cache = (
            StaticGeometryCache(dict(cache_config))
            if self.static_geometry_cache_enabled else None
        )
        self.reset_scratch_parameters()
        legacy = [name for name, _ in self.named_modules() if any(
            token in name.lower() for token in LEGACY_MODULE_TOKENS
        )]
        if legacy:
            raise AssertionError(f"clean V3 instantiated legacy modules: {legacy}")

    def reset_scratch_parameters(self) -> None:
        """Deterministic caller-seeded Xavier initialization for scratch runs."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        final = self.canonical_coordinate_head.network[-2]
        if not isinstance(final, nn.Linear):
            raise TypeError("canonical coordinate head must end with Linear + Tanh")
        nn.init.normal_(final.weight, mean=0.0, std=self.final_coordinate_initialization_std)
        nn.init.zeros_(final.bias)

    @staticmethod
    def checkpoint_forbidden_tokens() -> tuple[str, ...]:
        return LEGACY_MODULE_TOKENS

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
        observed_points_input, observed_mask_input = as_padded_points(observed)
        template_points_input, template_mask_input = as_padded_points(template)
        static = None
        if self._static_geometry_cache is not None:
            static = self._static_geometry_cache.get_or_build(
                observed_points_input, observed_mask_input,
                template_points_input, template_mask_input,
                batch_identity=(
                    "|".join(map(str, batch.get("sample_id", ())))
                    if batch is not None else "explicit-input"
                ),
            )
        observed_encoded = self.observed_encoder(
            {"points": observed_points_input, "valid_mask": observed_mask_input},
            precomputed_indices=(None if static is None else static["observed_encoder_knn"]),
        )
        batch_size = observed_points_input.shape[0]
        if self.shared_template_encoding:
            if not (
                torch.equal(template_mask_input, template_mask_input[:1].expand_as(template_mask_input))
                and torch.equal(
                    template_points_input,
                    template_points_input[:1].expand_as(template_points_input),
                )
            ):
                raise ValueError("shared_template_encoding requires an identical padded template")
            template_encoded_one = self.template_encoder(
                {
                    "points": template_points_input[:1],
                    "valid_mask": template_mask_input[:1],
                },
                precomputed_indices=(
                    None if static is None else static["template_encoder_knn"][:1]
                ),
                samplewise_learned_ops=True,
            )
            template_encoded = type(template_encoded_one)(
                points=template_encoded_one.points.expand(batch_size, -1, -1),
                point_features=template_encoded_one.point_features.expand(batch_size, -1, -1),
                global_feature=template_encoded_one.global_feature.expand(batch_size, -1),
                valid_mask=template_encoded_one.valid_mask.expand(batch_size, -1),
            )
        else:
            template_encoded = self.template_encoder(
                {"points": template_points_input, "valid_mask": template_mask_input},
                precomputed_indices=(None if static is None else static["template_encoder_knn"]),
                samplewise_learned_ops=True,
            )
        observed_points, observed_tokens, observed_mask, observed_indices = select_tokens(
            observed_encoded.points, observed_encoded.point_features,
            observed_encoded.valid_mask, self.max_observed_tokens,
            None if static is None else static["observed_fps_indices"],
            None if static is None else static["observed_token_mask"],
        )
        template_points, template_tokens, template_mask, template_indices = select_tokens(
            template_encoded.points, template_encoded.point_features,
            template_encoded.valid_mask, self.max_template_tokens,
            None if static is None else static["template_fps_indices"],
            None if static is None else static["template_token_mask"],
        )
        observed_tokens, template_tokens, _ = self.interaction_transformer(
            observed_tokens, template_tokens, observed_mask, template_mask
        )
        observed_normals_dense = _extract_padded_feature(observed, "normals_C")
        template_normals = _extract_padded_feature(template, "normals_O")
        if observed_normals_dense is not None:
            observed_normals_dense = observed_normals_dense.to(observed_encoded.points)
            observed_normals = batched_gather(observed_normals_dense, observed_indices)
        else:
            observed_normals = None
        if template_normals is not None:
            template_normals = batched_gather(
                template_normals.to(template_points), template_indices
            )
        observed_matching = self.dual_stream_geometry_encoder(
            observed_tokens, observed_points, observed_mask, observed_normals,
            None if static is None else static["observed_geometry_knn"],
        )["matching_features"]
        template_matching = self.dual_stream_geometry_encoder(
            template_tokens, template_points, template_mask, template_normals,
            None if static is None else static["template_geometry_knn"],
        )["matching_features"]
        observed_interaction_dense = nearest_interpolate(
            observed_encoded.points, observed_points, observed_matching, observed_mask,
            None if static is None else static["dense_to_observed_token_indices"],
        )
        fine_template = self.fine_template_projection(template_matching)
        template_context = (
            fine_template * template_mask.unsqueeze(-1)
        ).sum(1) / template_mask.sum(1, keepdim=True).clamp_min(1)
        template_context = self.template_context_projection(
            template_context + template_encoded.global_feature
        )
        observed_interaction_dense = (
            observed_interaction_dense + template_context[:, None]
            + observed_encoded.global_feature[:, None]
        )
        dense_observed = self.dense_observed_fine_projection(
            observed_encoded.point_features
        )
        fine = self.fine_feature_adapter(
            dense_observed, observed_interaction_dense, observed_encoded.points,
            observed_encoded.valid_mask, observed_normals_dense,
            None if static is None else static["fine_adapter_knn"],
        )
        # Coordinates, symmetry transforms and the numerical pose solver stay
        # fp32 even in the optional AMP ablation.
        q_normalized = self.canonical_coordinate_head(
            fine["fine_point_features"]
        ).float()
        q_normalized = q_normalized * observed_encoded.valid_mask.unsqueeze(-1)
        if batch is None:
            template_min = template_encoded.points.amin(1)
            template_max = template_encoded.points.amax(1)
        else:
            template_min = torch.stack([
                vertices.to(q_normalized).amin(0)
                for vertices in batch["template_mesh_vertices_O"]
            ])
            template_max = torch.stack([
                vertices.to(q_normalized).amax(0)
                for vertices in batch["template_mesh_vertices_O"]
            ])
        extent = (template_max - template_min).clamp_min(1e-8)
        q_aux = 0.5 * (q_normalized + 1.0) * extent[:, None] + template_min[:, None]
        valid = observed_encoded.valid_mask
        weights = valid.to(q_aux.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0)
        solution = self.weighted_procrustes.solve(
            q_aux.float(), observed_encoded.points.float(), weights.float(), valid
        )
        pose = solution["transform"].to(q_aux)
        batch_size = len(q_aux)
        zeros_observed = q_aux.new_zeros(valid.shape)
        zeros_template = q_aux.new_zeros(template_encoded.valid_mask.shape)
        active_similarity_logits = (
            fine["fine_point_features"] @ fine_template.transpose(-2, -1)
        ) / (fine_template.shape[-1] ** 0.5)
        active_similarity_logits = active_similarity_logits.masked_fill(
            ~template_mask[:, None], float("-1e4")
        )
        prediction = RegistrationPrediction(
            pose_hypotheses=pose[:, None], pose_logits=pose.new_zeros((batch_size, 1)),
            pose_uncertainty=pose.new_zeros((batch_size, 1, 0)),
            observed_overlap_logits=zeros_observed,
            template_visibility_logits=zeros_template,
            correspondence_points_O=q_aux, correspondence_confidence=weights,
            observed_region_logits=None, active_region_logits=None,
            insufficient_information_logit=pose.new_zeros((batch_size, 1)),
            observed_valid_mask=valid, template_valid_mask=template_encoded.valid_mask,
            symmetry_available=_symmetry_availability(
                symmetry_metadata, batch_size, pose.device
            ),
            base_pose=pose, correspondence_pose=pose,
            correspondence_pose_diagnostics=solution,
            base_pose_source="raw_q_aux_uniform_procrustes",
            pose_hypotheses_enabled=False, weighting_mode="uniform",
            correspondence_logits=active_similarity_logits,
            correspondence_auxiliary={
                "fine_aux_coordinate_normalized": q_normalized,
                "fine_feature_variance": fine["fine_feature_variance"],
                "fine_feature_effective_rank": fine["fine_feature_effective_rank"],
                "fine_feature_pairwise_distance": fine["fine_feature_pairwise_distance"],
                "fine_feature_collision_fraction": fine["fine_feature_collision_fraction"],
            },
            correspondence_feature_path={
                "observed_encoder_features": observed_encoded.point_features,
                "template_encoder_features": template_encoded.point_features,
                "sampled_observed_interaction_tokens": observed_matching,
                "sampled_template_interaction_tokens": template_matching,
                "fine_point_features": fine["fine_point_features"],
                "observed_token_indices": observed_indices,
                "template_token_indices": template_indices,
            },
        )
        prediction.validate()
        return prediction


__all__ = [
    "CoordinateGuidedSurfaceRegistrationV3",
    "LEGACY_MODULE_TOKENS",
    "state_dict_sha256",
]
