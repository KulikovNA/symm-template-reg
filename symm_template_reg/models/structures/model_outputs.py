"""Typed runtime output contract for direct registration predictions."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor


@dataclass
class RegistrationPrediction:
    """All dense outputs use padded layout and are accompanied by valid masks.

    Shapes:
        pose_hypotheses: ``[B, K, 4, 4]``
        pose_logits: ``[B, K]``
        pose_uncertainty: ``[B, K, U]``
        observed_overlap_logits: ``[B, No]``
        template_visibility_logits: ``[B, Nt]``
        correspondence_points_O: ``[B, No, 3]``
        correspondence_confidence: ``[B, No]``
        observed_region_logits: ``[B, No, R]`` or ``None``
        active_region_logits: ``[B, R]`` or ``None``
        insufficient_information_logit: ``[B, 1]``
    """

    pose_hypotheses: Tensor
    pose_logits: Tensor
    pose_uncertainty: Tensor
    observed_overlap_logits: Tensor
    template_visibility_logits: Tensor
    correspondence_points_O: Tensor
    correspondence_confidence: Tensor
    observed_region_logits: Tensor | None
    active_region_logits: Tensor | None
    insufficient_information_logit: Tensor
    observed_valid_mask: Tensor
    template_valid_mask: Tensor
    auxiliary_outputs: list[dict[str, Tensor]] | None = None
    symmetry_available: Tensor | None = None
    observed_centroid_C: Tensor | None = None
    observed_scale: Tensor | None = None
    base_pose: Tensor | None = None
    base_pose_parameters_normalized: Tensor | None = None
    base_uncertainty: Tensor | None = None
    base_correction_transform: Tensor | None = None
    residual_pose_parameters: Tensor | None = None
    residual_transforms: Tensor | None = None
    correspondence_pose: Tensor | None = None
    correspondence_pose_diagnostics: dict[str, Tensor] | None = None
    context_diagnostics: dict[str, Tensor] | None = None
    base_pose_source: str | None = None
    pose_hypotheses_enabled: bool = True
    weighting_mode: str | None = None
    correspondence_logits: Tensor | None = None
    correspondence_auxiliary: dict[str, Tensor] | None = None
    correspondence_feature_path: dict[str, Tensor] | None = None

    def to(self, *args: Any, **kwargs: Any) -> "RegistrationPrediction":
        moved_pose = self.pose_hypotheses.to(*args, **kwargs)
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                value = (
                    value.to(*args, **kwargs)
                    if value.is_floating_point() or value.is_complex()
                    else value.to(device=moved_pose.device)
                )
            elif isinstance(value, list):
                value = [
                    {
                        key: (
                            item.to(*args, **kwargs)
                            if isinstance(item, Tensor) and (item.is_floating_point() or item.is_complex())
                            else item.to(device=moved_pose.device)
                            if isinstance(item, Tensor)
                            else item
                        )
                        for key, item in layer.items()
                    }
                    for layer in value
                ]
            elif isinstance(value, dict):
                value = {
                    key: (
                        item.to(*args, **kwargs)
                        if isinstance(item, Tensor)
                        and (item.is_floating_point() or item.is_complex())
                        else item.to(device=moved_pose.device)
                        if isinstance(item, Tensor)
                        else item
                    )
                    for key, item in value.items()
                }
            values[field.name] = value
        return type(self)(**values)

    def validate(self) -> None:
        if self.pose_hypotheses.ndim != 4 or self.pose_hypotheses.shape[-2:] != (4, 4):
            raise ValueError("pose_hypotheses must have shape [B,K,4,4]")
        batch, queries = self.pose_hypotheses.shape[:2]
        if self.pose_logits.shape != (batch, queries):
            raise ValueError("pose_logits shape disagrees with pose_hypotheses")
        if self.pose_uncertainty.ndim != 3 or self.pose_uncertainty.shape[:2] != (batch, queries):
            raise ValueError("pose_uncertainty shape disagrees with pose_hypotheses")
        observed_shape = self.observed_valid_mask.shape
        template_shape = self.template_valid_mask.shape
        if len(observed_shape) != 2 or observed_shape[0] != batch:
            raise ValueError("observed_valid_mask must have shape [B,No]")
        if len(template_shape) != 2 or template_shape[0] != batch:
            raise ValueError("template_valid_mask must have shape [B,Nt]")
        if self.observed_valid_mask.dtype != torch.bool or self.template_valid_mask.dtype != torch.bool:
            raise TypeError("point validity masks must be boolean")
        if self.observed_overlap_logits.shape != observed_shape:
            raise ValueError("observed overlap/mask shapes disagree")
        if self.correspondence_points_O.shape[:2] != self.observed_valid_mask.shape:
            raise ValueError("correspondence/observed-mask shapes disagree")
        if self.correspondence_points_O.shape[-1] != 3:
            raise ValueError("correspondence_points_O must end in xyz")
        if self.correspondence_confidence.shape != observed_shape:
            raise ValueError("correspondence confidence/mask shapes disagree")
        if self.template_visibility_logits.shape != self.template_valid_mask.shape:
            raise ValueError("template visibility/mask shapes disagree")
        if self.insufficient_information_logit.shape != (batch, 1):
            raise ValueError("insufficient_information_logit must have shape [B,1]")
        if self.observed_region_logits is not None:
            if self.observed_region_logits.ndim != 3 or self.observed_region_logits.shape[:2] != observed_shape:
                raise ValueError("observed region/mask shapes disagree")
        if self.active_region_logits is not None:
            if self.active_region_logits.ndim != 2 or self.active_region_logits.shape[0] != batch:
                raise ValueError("active region batch shape disagrees")
        if (self.observed_region_logits is None) != (self.active_region_logits is None):
            raise ValueError("observed and active region logits must both be present or both be None")
        if (
            self.observed_region_logits is not None
            and self.active_region_logits is not None
            and self.observed_region_logits.shape[-1] != self.active_region_logits.shape[-1]
        ):
            raise ValueError("observed and active region capacities disagree")
        if self.symmetry_available is not None:
            if self.symmetry_available.shape != (batch,) or self.symmetry_available.dtype != torch.bool:
                raise ValueError("symmetry_available must be bool [B]")
        if self.observed_centroid_C is not None:
            if self.observed_centroid_C.shape != (batch, 3):
                raise ValueError("observed_centroid_C must have shape [B,3]")
        if self.observed_scale is not None:
            if self.observed_scale.shape != (batch,):
                raise ValueError("observed_scale must have shape [B]")
        if self.base_pose is not None and self.base_pose.shape != (batch, 4, 4):
            raise ValueError("base_pose must have shape [B,4,4]")
        if (
            self.base_pose_parameters_normalized is not None
            and self.base_pose_parameters_normalized.shape != (batch, 9)
        ):
            raise ValueError("base_pose_parameters_normalized must have shape [B,9]")
        if (
            self.residual_pose_parameters is not None
            and self.residual_pose_parameters.shape != (batch, queries, 9)
        ):
            raise ValueError("residual_pose_parameters must have shape [B,K,9]")
        if self.base_correction_transform is not None and self.base_correction_transform.shape != (batch, 4, 4):
            raise ValueError("base_correction_transform must have shape [B,4,4]")
        if (
            self.residual_transforms is not None
            and self.residual_transforms.shape != (batch, queries, 4, 4)
        ):
            raise ValueError("residual_transforms must have shape [B,K,4,4]")
        if self.correspondence_pose is not None and self.correspondence_pose.shape != (batch, 4, 4):
            raise ValueError("correspondence_pose must have shape [B,4,4]")
        tensor_values = [value for value in self.as_dict().values() if isinstance(value, Tensor)]
        if self.context_diagnostics is not None:
            tensor_values.extend(self.context_diagnostics.values())
        if self.correspondence_pose_diagnostics is not None:
            tensor_values.extend(
                value for value in self.correspondence_pose_diagnostics.values()
                if isinstance(value, Tensor)
            )
        if self.auxiliary_outputs is not None:
            for layer in self.auxiliary_outputs:
                required = {"pose_hypotheses", "pose_logits", "pose_uncertainty"}
                if not required.issubset(layer):
                    raise ValueError("each auxiliary output must contain pose, logit and uncertainty tensors")
                if layer["pose_hypotheses"].shape != self.pose_hypotheses.shape:
                    raise ValueError("auxiliary pose_hypotheses shape disagrees")
                if layer["pose_logits"].shape != self.pose_logits.shape:
                    raise ValueError("auxiliary pose_logits shape disagrees")
                if layer["pose_uncertainty"].shape != self.pose_uncertainty.shape:
                    raise ValueError("auxiliary pose_uncertainty shape disagrees")
                tensor_values.extend(
                    value for value in layer.values() if isinstance(value, Tensor)
                )
        devices = {value.device for value in tensor_values}
        if len(devices) != 1:
            raise ValueError("prediction tensors must share one device")
        if not all(torch.isfinite(value).all() for value in tensor_values if value.is_floating_point()):
            raise ValueError("prediction tensors contain NaN/Inf")

    def as_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}
