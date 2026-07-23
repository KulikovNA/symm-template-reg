"""Sample-conditioned residual hypotheses composed on a direct base pose.

Convention: ``R_k = R_delta_camera_k @ R_base`` and
``t_k = t_base + observed_scale * delta_translation_normalized_k``.  Both
residual terms are expressed in the camera frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

import torch
from torch import Tensor, nn

from symm_template_reg.models.pose.pose_representation import make_transform
from symm_template_reg.models.pose.rotation import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from symm_template_reg.registry import HEADS


def compose_camera_residual(
    base_pose: Tensor,
    delta_rotation_6d: Tensor,
    delta_translation_normalized: Tensor,
    observed_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    delta_rotation = rotation_6d_to_matrix(delta_rotation_6d)
    base_rotation = base_pose[..., :3, :3]
    base_translation = base_pose[..., :3, 3]
    while base_rotation.ndim < delta_rotation.ndim:
        base_rotation = base_rotation.unsqueeze(-3)
        base_translation = base_translation.unsqueeze(-2)
        observed_scale = observed_scale.unsqueeze(-1)
    rotation = delta_rotation @ base_rotation
    translation = (
        base_translation
        + observed_scale.unsqueeze(-1) * delta_translation_normalized
    )
    return make_transform(rotation, translation), make_transform(
        delta_rotation,
        observed_scale.unsqueeze(-1) * delta_translation_normalized,
    )


@HEADS.register_module()
class ResidualPoseHypothesisHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_hypotheses: int = 8,
        num_decoder_layers: int = 3,
        feedforward_dim: int = 512,
        uncertainty_dim: int = 6,
        dropout: float = 0.0,
        query_conditioning: Mapping[str, Any] | None = None,
        residual_bounds: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if num_hypotheses < 1:
            raise ValueError("num_hypotheses must be positive")
        conditioning = dict(
            query_conditioning
            or {
                "type": "film",
                "apply_each_decoder_layer": True,
                "allow_unconditioned_bypass": False,
            }
        )
        if conditioning.get("type") != "film":
            raise ValueError("only query_conditioning.type='film' is supported")
        if bool(conditioning.get("allow_unconditioned_bypass", False)):
            raise ValueError(
                "conditioned residual architecture forbids an unconditioned bypass"
            )
        self.num_hypotheses = int(num_hypotheses)
        self.residual_bounds = dict(residual_bounds or {})
        self.max_rotation_rad = math.radians(
            float(self.residual_bounds.get("max_rotation_deg", 0.0))
        )
        self.max_translation_m = float(
            self.residual_bounds.get("max_translation_m", 0.0)
        )
        self.bounded = bool(self.residual_bounds)
        if self.bounded and (
            self.max_rotation_rad <= 0.0 or self.max_translation_m <= 0.0
        ):
            raise ValueError("residual bounds must both be positive")
        self.apply_each_decoder_layer = bool(
            conditioning.get("apply_each_decoder_layer", True)
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
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_decoder_layers)]
        )
        self.mode_embedding = nn.Embedding(num_hypotheses, embed_dim)
        self.context_projection = nn.Linear(embed_dim, embed_dim)
        self.film = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim * 2) for _ in range(num_decoder_layers)]
        )
        self.residual_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 9),
        )
        self.logit_projection = nn.Linear(embed_dim, 1)
        self.uncertainty_projection = nn.Linear(embed_dim, uncertainty_dim)
        final = self.residual_projection[-1]
        assert isinstance(final, nn.Linear)
        # Tiny non-zero weights keep the initial residual close to identity
        # while allowing first-step gradients to reach the conditioned memory.
        nn.init.normal_(final.weight, mean=0.0, std=1e-4)
        with torch.no_grad():
            if self.bounded:
                final.bias.zero_()
            else:
                final.bias.copy_(
                    torch.tensor(
                        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
                    )
                )

    def _condition(self, target: Tensor, context: Tensor, layer: int) -> Tensor:
        gamma, beta = self.film[layer](context).chunk(2, dim=-1)
        return (1.0 + gamma.unsqueeze(1)) * target + beta.unsqueeze(1)

    def _decode(
        self, target: Tensor, base_pose: Tensor, observed_scale: Tensor
    ) -> dict[str, Tensor]:
        parameters = self.residual_projection(target)
        if self.bounded:
            rotation_vector = torch.tanh(parameters[..., :3])
            norm = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
            rotation_vector = rotation_vector * torch.clamp(
                self.max_rotation_rad / norm.clamp_min(1e-12), max=1.0
            )
            delta_rotation = axis_angle_to_matrix(rotation_vector)
            delta_translation_m = (
                torch.tanh(parameters[..., 6:9]) * self.max_translation_m
            )
            base_rotation = base_pose[..., :3, :3].unsqueeze(1)
            base_translation = base_pose[..., :3, 3].unsqueeze(1)
            hypotheses = make_transform(
                delta_rotation @ base_rotation,
                base_translation + delta_translation_m,
            )
            residual_transforms = make_transform(
                delta_rotation, delta_translation_m
            )
            canonical_parameters = torch.cat(
                (
                    matrix_to_rotation_6d(delta_rotation),
                    delta_translation_m / observed_scale[:, None, None].clamp_min(1e-8),
                ),
                dim=-1,
            )
        else:
            hypotheses, residual_transforms = compose_camera_residual(
                base_pose,
                parameters[..., :6],
                parameters[..., 6:9],
                observed_scale,
            )
            canonical_parameters = parameters
        return {
            "pose_hypotheses": hypotheses,
            "pose_logits": self.logit_projection(target).squeeze(-1),
            "pose_uncertainty": self.uncertainty_projection(target),
            "residual_pose_parameters": canonical_parameters,
            "residual_transforms": residual_transforms,
        }

    def forward(
        self,
        sample_context: Tensor,
        memory: Tensor,
        memory_valid_mask: Tensor,
        base_pose: Tensor,
        observed_scale: Tensor,
    ) -> dict[str, object]:
        if sample_context is None:
            raise ValueError("sample_context is mandatory for residual hypotheses")
        batch = sample_context.shape[0]
        if self.num_hypotheses == 1:
            identity = torch.eye(4, dtype=base_pose.dtype, device=base_pose.device)
            identity = identity.expand(batch, 1, 4, 4).clone()
            parameters = base_pose.new_tensor(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            ).expand(batch, 1, 9)
            return {
                "pose_hypotheses": base_pose.unsqueeze(1),
                "pose_logits": base_pose.new_zeros((batch, 1)),
                "pose_uncertainty": base_pose.new_zeros((batch, 1, 6)),
                "residual_pose_parameters": parameters,
                "residual_transforms": identity,
                "auxiliary_outputs": [],
            }
        modes = self.mode_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
        target = modes + self.context_projection(sample_context).unsqueeze(1)
        intermediate = []
        for index, layer in enumerate(self.layers):
            if self.apply_each_decoder_layer or index == 0:
                target = self._condition(target, sample_context, index)
            target = layer(
                target, memory, memory_key_padding_mask=~memory_valid_mask
            )
            intermediate.append(self._decode(target, base_pose, observed_scale))
        output = dict(intermediate[-1])
        output["auxiliary_outputs"] = intermediate[:-1]
        return output


__all__ = ["ResidualPoseHypothesisHead", "compose_camera_residual"]
