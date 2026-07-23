"""Direct sample-conditioned base pose without an independent learned query."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import math

import torch
from torch import Tensor, nn

from symm_template_reg.models.pose.pose_representation import make_transform
from symm_template_reg.models.pose.rotation import axis_angle_to_matrix
from symm_template_reg.registry import HEADS, POSE_MODULES, build_from_cfg


@HEADS.register_module()
class ConditionedBasePoseHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        uncertainty_dim: int = 6,
        pose_codec: Mapping[str, Any] | None = None,
        predict_uncertainty: bool = True,
        split_rotation_translation: bool = False,
        rotation_uses_centroid: bool = False,
        translation_uses_centroid: bool = True,
        output_mode: str = "absolute",
        max_rotation_correction_deg: float = 15.0,
        max_translation_correction_m: float = 0.01,
    ) -> None:
        super().__init__()
        self.pose_codec = build_from_cfg(
            pose_codec or {"type": "PoseCodec"}, POSE_MODULES
        )
        self.split_rotation_translation = bool(split_rotation_translation)
        self.rotation_uses_centroid = bool(rotation_uses_centroid)
        self.translation_uses_centroid = bool(translation_uses_centroid)
        if self.rotation_uses_centroid:
            raise ValueError("v2 rotation branch must not consume centroid directly")
        if output_mode not in {"absolute", "bounded_correction"}:
            raise ValueError("output_mode must be absolute or bounded_correction")
        if output_mode == "bounded_correction" and not self.split_rotation_translation:
            raise ValueError("bounded_correction requires split_rotation_translation=True")
        self.output_mode = output_mode
        self.max_rotation_correction_rad = math.radians(
            float(max_rotation_correction_deg)
        )
        self.max_translation_correction_m = float(max_translation_correction_m)
        if self.split_rotation_translation:
            self.rotation_projection = nn.Sequential(
                nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 3 if output_mode == "bounded_correction" else 6),
            )
            self.translation_projection = nn.Sequential(
                nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )
            if output_mode == "bounded_correction":
                nn.init.zeros_(self.rotation_projection[-1].weight)
                nn.init.zeros_(self.rotation_projection[-1].bias)
                nn.init.zeros_(self.translation_projection[-1].weight)
                nn.init.zeros_(self.translation_projection[-1].bias)
        else:
            self.pose_projection = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 9),
            )
        self.uncertainty_projection = (
            nn.Linear(embed_dim, uncertainty_dim) if predict_uncertainty else None
        )

    def forward(
        self,
        sample_context: Tensor,
        observed_centroid_C: Tensor,
        observed_scale: Tensor,
        *,
        rotation_context: Tensor | None = None,
        translation_context: Tensor | None = None,
        reference_pose: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        if sample_context.ndim != 2:
            raise ValueError("sample_context must have shape [B,D]")
        rotation_context = sample_context if rotation_context is None else rotation_context
        translation_context = (
            sample_context if translation_context is None else translation_context
        )
        if self.split_rotation_translation:
            rotation_parameters = self.rotation_projection(rotation_context)
            translation_parameters = self.translation_projection(translation_context)
            if self.output_mode == "absolute":
                transform = self.pose_codec.decode_transform(
                    rotation_parameters,
                    translation_parameters,
                    observed_centroid_C,
                    observed_scale,
                )
                parameters = torch.cat(
                    (rotation_parameters, translation_parameters), dim=-1
                )
                correction_transform = None
            else:
                if reference_pose is None:
                    raise ValueError("bounded_correction requires reference_pose")
                vector = torch.tanh(rotation_parameters)
                norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
                vector = vector * torch.clamp(
                    self.max_rotation_correction_rad / norm.clamp_min(1e-12),
                    max=1.0,
                )
                delta_rotation = axis_angle_to_matrix(vector)
                delta_translation = torch.tanh(translation_parameters)
                translation_norm = torch.linalg.vector_norm(
                    delta_translation, dim=-1, keepdim=True
                )
                delta_translation = delta_translation * torch.clamp(
                    self.max_translation_correction_m
                    / translation_norm.clamp_min(1e-12),
                    max=1.0,
                )
                rotation = delta_rotation @ reference_pose[..., :3, :3]
                translation = reference_pose[..., :3, 3] + delta_translation
                transform = make_transform(rotation, translation)
                correction_transform = make_transform(
                    delta_rotation, delta_translation
                )
                parameters = self.pose_codec.encode_transform(
                    transform, observed_centroid_C, observed_scale
                )
        else:
            parameters = self.pose_projection(sample_context)
            transform = self.pose_codec.decode_transform(
                parameters[..., :6], parameters[..., 6:9],
                observed_centroid_C, observed_scale,
            )
            correction_transform = None
        return {
            "base_rotation_6d": parameters[..., :6],
            "base_translation_normalized": parameters[..., 6:9],
            "base_pose_parameters_normalized": parameters,
            "base_T_C_from_O": transform,
            "base_uncertainty": (
                self.uncertainty_projection(rotation_context)
                if self.uncertainty_projection is not None
                else None
            ),
            "base_correction_transform": correction_transform,
        }


__all__ = ["ConditionedBasePoseHead"]
