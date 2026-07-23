from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.registry import LOSSES

from .correspondence_loss import masked_average


@LOSSES.register_module()
class ConsistencyLoss(nn.Module):
    """Keep predicted correspondences consistent with a selected direct pose."""

    def forward(
        self,
        observed_points_C: Tensor,
        correspondence_points_O: Tensor,
        pose_C_from_O: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        transformed = torch.matmul(
            pose_C_from_O[:, None, :3, :3], correspondence_points_O.unsqueeze(-1)
        ).squeeze(-1) + pose_C_from_O[:, None, :3, 3]
        error = torch.linalg.vector_norm(transformed - observed_points_C, dim=-1)
        return masked_average(error, valid_mask)

