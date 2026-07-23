"""Shared observed-centred translation codec for pose heads and targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from symm_template_reg.registry import POSE_MODULES

from .pose_representation import make_transform, split_transform
from .rotation import matrix_to_rotation_6d, rotation_6d_to_matrix


@dataclass(frozen=True)
class PoseCodecContext:
    observed_centroid_C: Tensor
    observed_scale: Tensor


@POSE_MODULES.register_module()
class PoseCodec(nn.Module):
    """Encode translation relative to an observed cloud centroid and scale."""

    def __init__(self, min_scale_m: float = 1e-6) -> None:
        super().__init__()
        if min_scale_m <= 0:
            raise ValueError("min_scale_m must be positive")
        self.min_scale_m = float(min_scale_m)

    def context(self, points_C: Tensor, valid_mask: Tensor) -> PoseCodecContext:
        points = torch.as_tensor(points_C)
        valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=points.device)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points_C must have shape [B,N,3]")
        if valid.shape != points.shape[:2]:
            raise ValueError("valid_mask must have shape [B,N]")
        weights = valid.to(points.dtype).unsqueeze(-1)
        centroid = (points * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        radius = torch.linalg.vector_norm(points - centroid[:, None], dim=-1)
        radius = radius.masked_fill(~valid, 0.0)
        scale = radius.max(dim=1).values.clamp_min(self.min_scale_m)
        return PoseCodecContext(centroid, scale)

    def encode_translation(
        self,
        translation_C: Tensor,
        observed_centroid_C: Tensor,
        observed_scale: Tensor,
    ) -> Tensor:
        scale = torch.as_tensor(
            observed_scale,
            dtype=translation_C.dtype,
            device=translation_C.device,
        )
        centroid = torch.as_tensor(
            observed_centroid_C,
            dtype=translation_C.dtype,
            device=translation_C.device,
        )
        while centroid.ndim < translation_C.ndim:
            centroid = centroid.unsqueeze(-2)
            scale = scale.unsqueeze(-1)
        return (translation_C - centroid) / scale.unsqueeze(-1)

    def decode_translation(
        self,
        translation_normalized: Tensor,
        observed_centroid_C: Tensor,
        observed_scale: Tensor,
    ) -> Tensor:
        centroid = torch.as_tensor(
            observed_centroid_C,
            dtype=translation_normalized.dtype,
            device=translation_normalized.device,
        )
        scale = torch.as_tensor(
            observed_scale,
            dtype=translation_normalized.dtype,
            device=translation_normalized.device,
        )
        while centroid.ndim < translation_normalized.ndim:
            centroid = centroid.unsqueeze(-2)
            scale = scale.unsqueeze(-1)
        return centroid + scale.unsqueeze(-1) * translation_normalized

    def encode_transform(
        self,
        transform: Tensor,
        observed_centroid_C: Tensor,
        observed_scale: Tensor,
    ) -> Tensor:
        rotation, translation = split_transform(transform)
        return torch.cat(
            (
                matrix_to_rotation_6d(rotation),
                self.encode_translation(
                    translation, observed_centroid_C, observed_scale
                ),
            ),
            dim=-1,
        )

    def decode_transform(
        self,
        rotation_6d: Tensor,
        translation_normalized: Tensor,
        observed_centroid_C: Tensor,
        observed_scale: Tensor,
    ) -> Tensor:
        translation = self.decode_translation(
            translation_normalized, observed_centroid_C, observed_scale
        )
        return make_transform(rotation_6d_to_matrix(rotation_6d), translation)


__all__ = ["PoseCodec", "PoseCodecContext"]
