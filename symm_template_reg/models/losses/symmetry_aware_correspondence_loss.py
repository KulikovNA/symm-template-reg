"""Row-aligned correspondence loss with one shared symmetry element per sample."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from symm_template_reg.models.symmetry.groups import (
    CyclicGroup,
    SO2Group,
    parse_rotation_group,
    rotation_group_matrices,
)
from symm_template_reg.registry import LOSSES


def _axis(metadata: Any, reference: Tensor) -> tuple[Tensor, Tensor]:
    return (
        torch.as_tensor(metadata.axis.direction, dtype=reference.dtype, device=reference.device),
        torch.as_tensor(metadata.axis.origin, dtype=reference.dtype, device=reference.device),
    )


@LOSSES.register_module()
class SymmetryAwareCorrespondenceLoss(nn.Module):
    def __init__(
        self,
        robust_type: str = "smooth_l1",
        beta: float = 0.01,
        use_shared_symmetry_element: bool = True,
    ) -> None:
        super().__init__()
        if robust_type not in {"smooth_l1", "l1", "l2"}:
            raise ValueError("robust_type must be smooth_l1, l1 or l2")
        if not use_shared_symmetry_element:
            raise ValueError("per-point symmetry-sector selection is forbidden")
        self.robust_type = robust_type
        self.beta = float(beta)
        self.use_shared_symmetry_element = True

    def _error(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = torch.broadcast_tensors(prediction, target)
        if self.robust_type == "smooth_l1":
            return torch.nn.functional.smooth_l1_loss(
                prediction, target, beta=self.beta, reduction="none"
            ).sum(-1)
        delta = prediction - target
        if self.robust_type == "l1":
            return delta.abs().sum(-1)
        return delta.square().sum(-1)

    def forward(
        self,
        prediction_points_O: Tensor,
        target_points_O: Tensor,
        valid_mask: Tensor,
        symmetry_metadata: Sequence[Any],
        effective_symmetry_groups: Sequence[Any],
        correspondence_confidence: Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | dict[str, Any]:
        if prediction_points_O.shape != target_points_O.shape:
            raise ValueError("prediction and target correspondence shapes disagree")
        if valid_mask.shape != prediction_points_O.shape[:2]:
            raise ValueError("valid_mask must have shape [B,N]")
        if len(symmetry_metadata) != len(prediction_points_O):
            raise ValueError("symmetry metadata batch length mismatch")
        weights = valid_mask.to(prediction_points_O.dtype)
        if correspondence_confidence is not None:
            weights = weights * correspondence_confidence.to(weights).clamp_min(0.0)
        sample_losses = []
        selected_indices = []
        selected_targets = []
        for index in range(len(prediction_points_O)):
            prediction = prediction_points_O[index]
            target = target_points_O[index]
            sample_weights = weights[index]
            denominator = sample_weights.sum().clamp_min(1e-8)
            group = parse_rotation_group(effective_symmetry_groups[index])
            axis, origin = _axis(symmetry_metadata[index], target)
            axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
            if isinstance(group, SO2Group):
                predicted_relative = prediction - origin
                target_relative = target - origin
                predicted_axial = torch.sum(predicted_relative * axis, dim=-1)
                target_axial = torch.sum(target_relative * axis, dim=-1)
                predicted_radial = torch.linalg.vector_norm(
                    predicted_relative - predicted_axial.unsqueeze(-1) * axis,
                    dim=-1,
                )
                target_radial = torch.linalg.vector_norm(
                    target_relative - target_axial.unsqueeze(-1) * axis,
                    dim=-1,
                )
                invariant_error = self._error(
                    torch.stack((predicted_axial, predicted_radial), dim=-1),
                    torch.stack((target_axial, target_radial), dim=-1),
                )
                sample_losses.append(
                    torch.sum(invariant_error * sample_weights) / denominator
                )
                selected_indices.append(-1)
                selected_targets.append(target)
                continue
            assert isinstance(group, CyclicGroup)
            rotations = rotation_group_matrices(
                group, axis, dtype=target.dtype, device=target.device
            )
            # One shared canonical convention per sample.  If S is an
            # equivalent object-frame symmetry, q^(S)=S^{-1}q_GT so that
            # (T_GT S) q^(S) = T_GT q_GT.
            inverse_rotations = rotations.transpose(-1, -2)
            equivalent = torch.einsum(
                "gij,nj->gni", inverse_rotations, target - origin
            ) + origin
            errors = self._error(prediction.unsqueeze(0), equivalent)
            aggregate = torch.sum(errors * sample_weights.unsqueeze(0), dim=-1)
            selected = aggregate.argmin()
            sample_losses.append(aggregate[selected] / denominator)
            selected_indices.append(int(selected.detach()))
            selected_targets.append(equivalent[selected])
        loss = torch.stack(sample_losses).mean()
        if not return_diagnostics:
            return loss
        return {
            "loss": loss,
            "selected_shared_symmetry_element": torch.tensor(
                selected_indices, dtype=torch.long, device=prediction_points_O.device
            ),
            "matched_target_points_O": torch.stack(selected_targets),
        }

    def forward_with_diagnostics(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs["return_diagnostics"] = True
        result = self.forward(*args, **kwargs)
        assert isinstance(result, dict)
        return result


__all__ = ["SymmetryAwareCorrespondenceLoss"]
